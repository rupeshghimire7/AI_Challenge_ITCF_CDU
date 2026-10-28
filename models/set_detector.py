"""ABSA as set prediction, the way DETR does object detection (Carion et al., 2020).

DETR: an image backbone gives features; N learned "object queries" read them through a transformer decoder; each query
outputs one (class, box) or "no object"; training matches queries to the gold objects one-to-one (Hungarian matching),
so the model emits a set with no duplicates and no thresholds per class.

Here: the encoder (bge-small or LFM2.5) gives token states; N queries read them through a small transformer decoder;
each query outputs one opinion  (entity or "no opinion", attribute, star 0-4)  - the E#A#star triple is the "class",
and the query's cross-attention over the words is the "box" (where in the sentence the opinion is, used later as the
evidence highlight). Differences from the classifier heads:
  - entity and attribute are separate outputs, so any of the 22 x 9 = 198 pairs can be predicted, including pairs never
    seen in training (the classifier can only output its training categories);
  - one sentence can hold the same category twice with different polarities (an opinion is a query, not a slot);
  - "no opinion" is a learned class (DETR's no-object, weight 0.1), so empty sentences need no separate gate.

For the shared decoding and metrics, the output is also reduced to the classifier's form: per category, the score of
the best query, p(entity) * p(attribute), and that query's polarity.

Long text: during training every example is one segment (all its windows). At inference an example longer than
`group_windows` windows is split into groups of that many windows, each decoded on its own and merged like the
classifier's windows, so the number of opinions found is not capped by the number of queries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from models.data import BRACKET, N_STARS, STAR_TARGET, Batch
from models.heads import encode, merge_windows
from src.config import ATTRIBUTES, ENTITIES

E_INDEX, A_INDEX = {e: i for i, e in enumerate(ENTITIES)}, {a: i for i, a in enumerate(ATTRIBUTES)}
CATEGORIES = [f"{e}#{a}" for e in ENTITIES for a in ATTRIBUTES]  # all 198, entity-major (the order of p(E) x p(A))


class ABSASetDetector(nn.Module):
    def __init__(self, encoder: nn.Module, n_queries: int = 16, d_model: int = 256, layers: int = 2, heads: int = 8,
                 dropout: float = 0.1, no_opinion_weight: float = 0.1, group_windows: int = 4):
        super().__init__()
        self.encoder, self.n_queries, self.group_windows = encoder, n_queries, group_windows
        self.categories = CATEGORIES
        self.memory = nn.Sequential(nn.Linear(encoder.hidden_size, d_model), nn.LayerNorm(d_model))
        self.queries = nn.Embedding(n_queries, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.entity = nn.Linear(d_model, len(ENTITIES) + 1)  # last class = no opinion
        self.attribute, self.polarity = nn.Linear(d_model, len(ATTRIBUTES)), nn.Linear(d_model, N_STARS)
        w = torch.ones(len(ENTITIES) + 1)
        w[-1] = no_opinion_weight
        self.register_buffer("entity_weight", w)

    def segments(self, owner: list[int]) -> tuple[list[int], list[int], list[int]]:
        """Window -> (segment, slot in the segment), and segment -> example. Training: one segment per example."""
        group = self.group_windows if not self.training else None
        seg, slot, seg_owner, prev, local = [], [], [], -1, 0
        for o in owner:
            local = local + 1 if o == prev else 0
            prev = o
            pos = local % group if group else local
            if pos == 0:
                seg_owner.append(o)
            seg.append(len(seg_owner) - 1)
            slot.append(pos)
        return seg, slot, seg_owner

    def forward(self, batch: Batch) -> dict:
        mem = self.memory(encode(self.encoder, batch))  # [W, T, d]
        seg, slot, seg_owner = self.segments(batch.owner.tolist())
        s, g, (w, t, d) = len(seg_owner), max(slot) + 1, mem.shape
        seg_t, slot_t = torch.tensor(seg, device=mem.device), torch.tensor(slot, device=mem.device)
        memory = mem.new_zeros(s, g, t, d)
        keep = torch.zeros(s, g, t, dtype=torch.bool, device=mem.device)
        memory[seg_t, slot_t] = mem
        keep[seg_t, slot_t] = batch.target_mask
        memory, keep = memory.view(s, g * t, d), keep.view(s, g * t)
        keep[~keep.any(1), 0] = True  # an empty text attends to its first token instead of nothing
        h = self.decoder(self.queries.weight.unsqueeze(0).expand(s, -1, -1).to(memory.dtype), memory,
                         memory_key_padding_mask=~keep)
        return {"entity": self.entity(h), "attribute": self.attribute(h), "polarity": self.polarity(h),
                "seg_owner": torch.tensor(seg_owner, device=mem.device), "n": len(batch.examples)}

    def probs(self, out: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per example and category (198): best query's p(entity) * p(attribute), and that query's polarity."""
        pe = out["entity"].float().softmax(-1)[..., :-1]  # [S, N, 22]; the no-opinion mass is what is left out
        pa, pp = out["attribute"].float().softmax(-1), out["polarity"].float().softmax(-1)
        pc = (pe[..., :, None] * pa[..., None, :]).flatten(2)  # [S, N, 198]
        det, q = pc.max(1)
        pol = pp.gather(1, q[..., None].expand(-1, -1, N_STARS))
        det, pol, _, _ = merge_windows(det, pol, out["seg_owner"], out["n"])
        return det, pol

    def loss(self, out: dict, batch: Batch) -> tuple[torch.Tensor, dict]:
        le, la, lp = (F.log_softmax(out[k].float(), -1) for k in ("entity", "attribute", "polarity"))
        s, n = le.shape[:2]
        dev = le.device
        te = torch.full((s, n), len(ENTITIES), device=dev)  # default target: no opinion
        target = STAR_TARGET.to(dev)
        ta, tp = torch.full((s, n), -100, device=dev), torch.full((s, n), -100, device=dev)
        for i, o in enumerate(out["seg_owner"].tolist()):
            gold = [(E_INDEX[e], A_INDEX[a], int(BRACKET[s])) for e, a, s in batch.examples[o].gold]
            if not gold:
                continue
            g = torch.tensor(gold, device=dev)
            star_ll = lp[i] @ target[g[:, 2]].T  # [N, G] log-likelihood of each opinion's smoothed star target
            cost = -(le[i][:, g[:, 0]] + la[i][:, g[:, 1]] + star_ll)  # [N, G]: how unlikely each query makes each opinion
            r, c = linear_sum_assignment(cost.detach().cpu().numpy())
            r, c = torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)
            te[i, r], ta[i, r], tp[i, r] = g[c, 0], g[c, 1], g[c, 2]
        parts = {"entity": F.nll_loss(le.flatten(0, 1), te.flatten(), weight=self.entity_weight)}
        m = ta >= 0
        if m.any():
            parts["attribute"] = F.nll_loss(la[m], ta[m])
            parts["polarity"] = -(target[tp[m]] * lp[m]).sum(-1).mean()
        return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
