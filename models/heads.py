"""Decision heads: encoder token states -> per-category opinion probability and per-category star distribution (0-4).

encode()          run the encoder over a batch's windows, a bounded number at a time (any input length fits in memory)
merge_windows()   an example's windows -> one prediction: a category's score is its best window's score (an opinion
                  anywhere in a long text counts), and its polarity comes from that same window
ABSAClassifier    pooling + head + losses; base = plan M2, the rest are ladder steps:
  head="linear"   M2: one linear layer to C sigmoid logits (detection), one to C x 5 star logits (polarity, trained on
                  gold categories only, with data.STAR_TARGET's smoothed star targets)
  head="catattn"  L1: one learned query per category, initialised from the encoder's embedding of the category's name,
                  attends over the target tokens; per-category detector + shared star MLP on the attended vector
  factorised      L2: logit(E#A) = s_E + s_A + r_EA, with an L2 penalty on the pair residual r_EA
  gate            L3: sentence-level "has an opinion" logit; p(category) = p(gate) * p(category | opinion)
  loss="asl"      L5: asymmetric loss for detection, focal + class-balanced cross-entropy for polarity
  supcon          L9: supervised contrastive loss on pooled sentence vectors (positives share a category)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.data import BRACKET, N_STARS, STAR_TARGET, Batch, Example, Inputs

CHUNK = 64  # windows per encoder pass


def encode(encoder: nn.Module, batch: Batch) -> torch.Tensor:
    ids, mask, types = batch.input_ids, batch.attention_mask, batch.type_ids
    if len(ids) <= CHUNK:
        return encoder(ids, mask, types)
    return torch.cat([encoder(ids[i : i + CHUNK], mask[i : i + CHUNK], types[i : i + CHUNK]) for i in range(0, len(ids), CHUNK)])


def pool(states: torch.Tensor, target: torch.Tensor, how: str) -> torch.Tensor:
    if how == "cls":
        return states[:, 0]
    m = target.unsqueeze(-1).to(states.dtype)
    return (states * m).sum(1) / m.sum(1).clamp_min(1)


def merge_windows(det, pol, owner, n, gate=None, pooled=None):
    """[W, C] scores, [W, C, 5] stars (and optional [W] gate, [W, H] pooled) -> per example. Max over windows."""
    if det.shape[0] == n:  # one window per example: nothing to merge
        return det, pol, gate, pooled
    c = det.shape[1]
    best = det.new_full((n, c), float("-inf")).scatter_reduce(0, owner[:, None].expand(-1, c), det, "amax")
    win = (det == best[owner]).to(pol.dtype)  # 1 where this window holds the category's best score
    cnt = win.new_zeros(n, c).index_add(0, owner, win)
    pol = pol.new_zeros(n, c, pol.shape[-1]).index_add(0, owner, pol * win[..., None]) / cnt[..., None]
    if gate is not None:
        gate = gate.new_full((n,), float("-inf")).scatter_reduce(0, owner, gate, "amax")
    if pooled is not None:
        k = torch.bincount(owner, minlength=n).clamp_min(1).to(pooled.dtype)
        pooled = pooled.new_zeros(n, pooled.shape[1]).index_add(0, owner, pooled) / k[:, None]
    return best, pol, gate, pooled


class ABSAClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, categories: list[str], head: str = "linear", factorised: bool = False,
                 factor_l2: float = 1e-4, gate: bool = False, loss: str = "bce", supcon: float = 0.0,
                 pol_class_weights=None, dropout: float = 0.1):
        super().__init__()
        h, c = encoder.hidden_size, len(categories)
        self.encoder, self.categories, self.head = encoder, list(categories), head
        self.index = {k: i for i, k in enumerate(self.categories)}
        self.pooling = encoder.spec.pooling
        self.loss_name, self.supcon, self.factor_l2 = loss, supcon, factor_l2
        self.drop = nn.Dropout(dropout)
        if head == "linear":
            self.det, self.pol = nn.Linear(h, c), nn.Linear(h, c * N_STARS)
            nn.init.constant_(self.det.bias, -3.0)  # start near the real positive rate (~2%), not at p = 0.5
        elif head == "catattn":
            self.queries = nn.Parameter(torch.randn(c, h) / math.sqrt(h))  # replaced by init_queries()
            self.temperature = nn.Parameter(torch.tensor(10.0))
            self.det_w, self.det_b = nn.Parameter(torch.zeros(c, h)), nn.Parameter(torch.full((c,), -3.0))
            self.pol_mlp = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, N_STARS))
        else:
            raise ValueError(f"unknown head {head!r}")
        if factorised:
            ents = sorted({k.split("#")[0] for k in self.categories})
            atts = sorted({k.split("#")[1] for k in self.categories})
            self.s_ent, self.s_att = nn.Linear(h, len(ents)), nn.Linear(h, len(atts))
            self.register_buffer("ent_idx", torch.tensor([ents.index(k.split("#")[0]) for k in self.categories]))
            self.register_buffer("att_idx", torch.tensor([atts.index(k.split("#")[1]) for k in self.categories]))
        self.factorised = factorised
        self.gate = nn.Linear(h, 1) if gate else None
        self.register_buffer("cls_w", torch.tensor(pol_class_weights if pol_class_weights is not None else [1.0] * N_STARS))

    @torch.no_grad()
    def init_queries(self, inputs: Inputs) -> None:
        """L1: each category's query starts as the encoder's mean embedding of its name ("battery operation performance")."""
        names = [k.replace("#", " ").replace("_", " ").lower() for k in self.categories]
        was = self.encoder.training
        self.encoder.eval()
        dev = self.queries.device
        b = inputs.batch([Example(f"q{i}", "query", t) for i, t in enumerate(names)]).to(dev)
        vec = pool(encode(self.encoder, b), b.target_mask, "mean").float()
        self.queries.copy_(F.normalize(vec, dim=-1))
        self.encoder.train(was)

    def forward(self, batch: Batch) -> dict:
        states = encode(self.encoder, batch)
        pooled = pool(states, batch.target_mask, self.pooling)
        if self.head == "linear":
            z = self.drop(pooled)
            det, pol = self.det(z), self.pol(z).view(len(z), -1, N_STARS)
        else:
            m = batch.target_mask.clone()
            m[~m.any(1), 0] = True  # an empty text attends to its first token instead of nothing
            keys = F.normalize(states, dim=-1)
            scores = self.temperature * torch.einsum("wth,ch->wct", keys, F.normalize(self.queries, dim=-1).to(keys.dtype))
            attn = scores.masked_fill(~m[:, None, :], float("-inf")).softmax(-1)
            v = self.drop(torch.einsum("wct,wth->wch", attn, states))
            det = (v * self.det_w.to(v.dtype)).sum(-1) + self.det_b
            pol = self.pol_mlp(v)
        if self.factorised:
            z = self.drop(pooled)
            det = det + self.s_ent(z)[:, self.ent_idx] + self.s_att(z)[:, self.att_idx]
        gate = self.gate(self.drop(pooled)).squeeze(-1) if self.gate is not None else None
        det, pol, gate, pooled = merge_windows(det, pol, batch.owner, len(batch.examples), gate, pooled)
        return {"det": det, "pol": pol, "gate": gate, "pooled": pooled}

    def probs(self, out: dict) -> tuple[torch.Tensor, torch.Tensor]:
        det = torch.sigmoid(out["det"].float())
        if out["gate"] is not None:
            det = det * torch.sigmoid(out["gate"].float())[:, None]
        return det, out["pol"].float().softmax(-1)

    def targets(self, examples: list[Example], device) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-hot categories [n, C] and soft star targets [n, C, 5] (STAR_TARGET of the gold sentiment; a category with
        two polarities in one sentence gets half of each). Gold categories outside the label space are ignored."""
        y, yp = torch.zeros(len(examples), len(self.categories)), torch.zeros(len(examples), len(self.categories), N_STARS)
        for i, ex in enumerate(examples):
            for e, a, star in ex.gold:
                j = self.index.get(f"{e}#{a}")
                if j is not None:
                    y[i, j] = 1
                    yp[i, j] += STAR_TARGET[BRACKET[star]]
        return y.to(device), (yp / yp.sum(-1, keepdim=True).clamp_min(1)).to(device)

    def loss(self, out: dict, batch: Batch) -> tuple[torch.Tensor, dict]:
        det = out["det"].float()
        y, yp = self.targets(batch.examples, det.device)
        if out["gate"] is not None:
            logp = F.logsigmoid(out["gate"].float())[:, None] + F.logsigmoid(det)
            log1mp = torch.log1p(-logp.exp().clamp(max=1 - 1e-6))
        else:
            logp, log1mp = F.logsigmoid(det), F.logsigmoid(-det)
        if self.loss_name == "asl":  # gamma+ = 0, gamma- = 4, negatives shifted by 0.05
            pm = (logp.exp() - 0.05).clamp(min=0)
            elem = -(y * logp + (1 - y) * pm.pow(4) * torch.log1p(-pm.clamp(max=1 - 1e-6)))
        else:
            elem = -(y * logp + (1 - y) * log1mp)
        parts = {"det": elem.mean()}
        if out["gate"] is not None:
            has = (y.sum(1) > 0).float()
            parts["gate"] = F.binary_cross_entropy_with_logits(out["gate"].float(), has)
        m = y > 0
        if m.any():
            logq, t = F.log_softmax(out["pol"].float(), -1)[m], yp[m]
            ce = -(t * logq).sum(-1)
            if self.loss_name == "asl":  # focal (gamma 2) with class-balanced weights
                ce = (1 - (t * logq.exp()).sum(-1)).pow(2) * ce * (t @ self.cls_w)
            parts["pol"] = ce.mean()
        if self.supcon > 0 and len(y) > 1:
            z = F.normalize(out["pooled"].float(), dim=-1)
            eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
            logits = (z @ z.T / 0.1).masked_fill(eye, -1e4)
            logprob = logits - logits.logsumexp(1, keepdim=True)
            pos = ((y @ y.T) > 0) & ~eye
            has = pos.any(1)
            if has.any():
                parts["supcon"] = self.supcon * (-(logprob * pos).sum(1)[has] / pos.sum(1)[has]).mean()
        if self.factorised:
            r = self.det.weight if self.head == "linear" else self.det_w
            parts["l2"] = self.factor_l2 * r.pow(2).sum()
        return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
