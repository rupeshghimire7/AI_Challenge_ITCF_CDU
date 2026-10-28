"""Model inference with the selected model: bge_detr at epoch 9 (models/notebooks/ABSA_model_selection_report.md).

bge-small-en-v1.5 reads each sentence with its neighbours as review context and its negation scopes marked; a set head
(16 queries) predicts the sentence's opinions, each as entity, attribute and star 0-4. Only the inference path of
ABSA_experiments_base_tweak.ipynb is kept (no training, no other encoder or head), with the same windowing and arithmetic.
The checkpoint folder holds the weights, the encoder config and the tokenizer, so nothing is downloaded.

Sentence                  a sentence with its previous and next sentence (review context)
with_context(texts)       one review's sentences -> Sentences with their neighbours as context
ABSAModel(folder)         the checkpoint; .predict(sentences) -> detection probabilities [n, 198] and star probabilities
                          [n, 198, 5]; .categories (E#A, entity-major), .tau (detection threshold fitted on validation)
"""

from __future__ import annotations

import json
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import CHECKPOINT, CHECKPOINT_ZIP
from src.nuance_checker import NUANCES
from src.preprocess import normalise

N_STARS = 5
CHUNK = 64  # windows per encoder pass
GROUP = 4  # windows of one long sentence that the set head reads together


@dataclass
class Sentence:
    text: str
    prev: str = ""
    next: str = ""


def with_context(texts: list[str]) -> list[Sentence]:
    return [Sentence(t, texts[k - 1] if k else "", texts[k + 1] if k + 1 < len(texts) else "") for k, t in enumerate(texts)]


# ----------------------------------------------------------------------------------------------- input windows
def negation_spans(text: str) -> list[tuple[int, int]]:
    """Negation scope as character spans: the cue and up to 5 following words, stopping at punctuation or a contrast
    marker ("not bad", "never had a problem", "no issues")."""
    stop = re.compile(r"[.,;:!?]|" + NUANCES["contrast"].pattern, re.I)
    spans = []
    for m in NUANCES["negation"].finditer(text):
        end_m = stop.search(text, m.end())
        words = list(re.finditer(r"\S+", text[m.end() : end_m.start() if end_m else len(text)]))[:5]
        spans.append((m.start(), m.end() + words[-1].end() if words else m.end()))
    return spans


class Inputs:
    """Sentences -> padded windows  [CLS] prev [SEP] target-chunk [SEP] next [SEP],  each context side cut to `ctx_len`
    tokens. A target longer than one window is cut into chunks that overlap by a quarter. Type ids: 0 target, 1 review
    context, 2 target token inside a negation scope. `target` marks the target tokens, the only ones the head reads."""

    def __init__(self, tokenizer_file: Path, max_len: int, ctx_len: int):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_file))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.cls, self.sep, self.pad = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]"))
        self.window = max_len - 2  # [CLS] and the last [SEP]
        self.step = self.window - self.window // 4
        self.ctx_len = ctx_len
        self._cache: dict[str, tuple[list[int], list[int]]] = {}

    def tokens(self, text: str) -> tuple[list[int], list[int]]:
        """Token ids of the normalised text (no special tokens) and their type ids (2 inside a negation scope)."""
        if text in self._cache:
            return self._cache[text]
        clean = normalise(text)
        enc = self.tok.encode(clean, add_special_tokens=False)
        spans = negation_spans(clean)
        types = [2 if any(s < b and a < e for s, e in spans) else 0 for a, b in enc.offsets]
        if len(self._cache) < 100_000:  # bounded: serving never grows it without limit
            self._cache[text] = (enc.ids, types)
        return enc.ids, types

    def windows(self, s: Sentence) -> list[tuple[list[int], list[int], list[bool]]]:
        """(ids, type ids, target mask) of every window of one sentence; an empty sentence still gets one window."""
        ids, types = self.tokens(s.text)
        prev = self.tokens(s.prev)[0][-self.ctx_len :] if s.prev else []
        nxt = self.tokens(s.next)[0][: self.ctx_len] if s.next else []
        head = [self.cls] + (prev + [self.sep] if prev else [])
        tail = ([self.sep] + nxt if nxt else []) + [self.sep]
        head_t = [0] + ([1] * len(prev) + [0] if prev else [])
        tail_t = ([0] + [1] * len(nxt) if nxt else []) + [0]
        starts = [0]
        while starts[-1] + self.window < len(ids):
            starts.append(starts[-1] + self.step)
        out = []
        for a in starts:
            chunk = ids[a : a + self.window]
            out.append((head + chunk + tail, head_t + types[a : a + self.window] + tail_t,
                        [False] * len(head) + [True] * len(chunk) + [False] * len(tail)))
        return out

    def batch(self, sentences: list[Sentence]) -> dict[str, torch.Tensor]:
        rows, owner = [], []
        for i, s in enumerate(sentences):
            for w in self.windows(s):
                rows.append(w)
                owner.append(i)
        n, t = len(rows), max(len(r[0]) for r in rows)
        b = {"ids": torch.full((n, t), self.pad, dtype=torch.long), "mask": torch.zeros((n, t), dtype=torch.long),
             "types": torch.zeros((n, t), dtype=torch.long), "target": torch.zeros((n, t), dtype=torch.bool)}
        for j, (ids, types, target) in enumerate(rows):
            b["ids"][j, : len(ids)] = torch.tensor(ids)
            b["mask"][j, : len(ids)] = 1
            b["types"][j, : len(ids)] = torch.tensor(types)
            b["target"][j, : len(ids)] = torch.tensor(target)
        b["owner"] = torch.tensor(owner, dtype=torch.long)
        return b


# ----------------------------------------------------------------------------------------------- model
class Encoder(nn.Module):
    """bge-small-en-v1.5 (BERT, 12 layers, hidden 384) with the two input tweaks it was trained with: review-context tokens
    get token type 1, and target tokens inside a negation scope get a learned vector added to their embedding."""

    def __init__(self, config_file: Path):
        super().__init__()
        from transformers import BertConfig, BertModel

        cfg = BertConfig.from_json_file(str(config_file))
        cfg._attn_implementation = "sdpa"
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.scope = nn.Embedding(1, cfg.hidden_size)

    def forward(self, ids, mask, types):
        x = self.bert.embeddings.word_embeddings(ids)
        x = x + (types == 2).unsqueeze(-1).to(x.dtype) * self.scope.weight[0]
        return self.bert(inputs_embeds=x, attention_mask=mask, token_type_ids=(types == 1).long()).last_hidden_state


def segments(owner: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Window -> (segment, slot in the segment), and segment -> sentence: a sentence's windows in runs of GROUP."""
    seg, slot, seg_owner, prev, local = [], [], [], -1, 0
    for o in owner:
        local = local + 1 if o == prev else 0
        prev = o
        if local % GROUP == 0:
            seg_owner.append(o)
        seg.append(len(seg_owner) - 1)
        slot.append(local % GROUP)
    return seg, slot, seg_owner


def merge_segments(det, pol, owner, n):
    """Per segment -> per sentence: the max detection probability, and the star probabilities of the segment holding it."""
    if det.shape[0] == n:  # one segment per sentence: nothing to merge
        return det, pol
    c = det.shape[1]
    best = det.new_full((n, c), float("-inf")).scatter_reduce(0, owner[:, None].expand(-1, c), det, "amax")
    win = (det == best[owner]).to(pol.dtype)
    cnt = win.new_zeros(n, c).index_add(0, owner, win)
    return best, pol.new_zeros(n, c, pol.shape[-1]).index_add(0, owner, pol * win[..., None]) / cnt[..., None]


class SetDetector(nn.Module):
    """Encoder + DETR-style set head: 16 learned queries attend over the target tokens (2 decoder layers, width 256), and
    each query predicts one opinion: an entity (or no opinion), an attribute and a star."""

    def __init__(self, encoder_config: Path, n_entities: int, n_attributes: int, n_queries: int, d_model: int = 256,
                 layers: int = 2, heads: int = 8):
        super().__init__()
        self.encoder = Encoder(encoder_config)
        self.memory = nn.Sequential(nn.Linear(self.encoder.bert.config.hidden_size, d_model), nn.LayerNorm(d_model))
        self.queries = nn.Embedding(n_queries, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.entity = nn.Linear(d_model, n_entities + 1)  # last class = no opinion
        self.attribute, self.polarity = nn.Linear(d_model, n_attributes), nn.Linear(d_model, N_STARS)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> per sentence: each E#A category's detection probability [n, E*A] (the best query's p(entity) * p(attribute))
        and that query's star probabilities [n, E*A, 5]."""
        ids, mask, types = b["ids"], b["mask"], b["types"]
        mem = self.memory(torch.cat([self.encoder(ids[i : i + CHUNK], mask[i : i + CHUNK], types[i : i + CHUNK])
                                     for i in range(0, len(ids), CHUNK)]))  # [W, T, d]
        seg, slot, seg_owner = segments(b["owner"].tolist())
        s, g, (_, t, d) = len(seg_owner), max(slot) + 1, mem.shape
        seg_t, slot_t = torch.tensor(seg, device=mem.device), torch.tensor(slot, device=mem.device)
        memory = mem.new_zeros(s, g, t, d)
        keep = torch.zeros(s, g, t, dtype=torch.bool, device=mem.device)
        memory[seg_t, slot_t] = mem
        keep[seg_t, slot_t] = b["target"]
        memory, keep = memory.view(s, g * t, d), keep.view(s, g * t)
        keep[~keep.any(1), 0] = True  # an empty text attends to its first token instead of nothing
        h = self.decoder(self.queries.weight.unsqueeze(0).expand(s, -1, -1).to(memory.dtype), memory,
                         memory_key_padding_mask=~keep)
        pe = self.entity(h).float().softmax(-1)[..., :-1]  # the no-opinion mass is what is left out
        pa, pp = self.attribute(h).float().softmax(-1), self.polarity(h).float().softmax(-1)
        det, q = (pe[..., :, None] * pa[..., None, :]).flatten(2).max(1)  # [S, E*A], entity-major
        pol = pp.gather(1, q[..., None].expand(-1, -1, N_STARS))
        return merge_segments(det, pol, torch.tensor(seg_owner, device=det.device), int(b["owner"].max()) + 1)


# ----------------------------------------------------------------------------------------------- checkpoint
def pick_device(name: str) -> torch.device:
    """`auto`: a GPU only if a kernel really runs on it (an old GPU can be visible yet unsupported by this torch)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (torch.ones(8, device="cuda") @ torch.ones(8, device="cuda")).item()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def unpack(folder: Path) -> None:
    """Take one checkpoint folder (~140 MB) out of the notebook's absa_experiments.zip, which keeps the Colab layout
    result/<run>/checkpoint_<epoch>/ (<run> = the folder's parent name)."""
    prefix = f"result/{folder.parent.name}/{folder.name}/"
    with zipfile.ZipFile(CHECKPOINT_ZIP) as z:
        names = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
        if not names:
            raise FileNotFoundError(f"{folder} has no model.pt and {CHECKPOINT_ZIP} has no {prefix}")
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            with z.open(n) as src, (folder / n[len(prefix) :]).open("wb") as dst:
                shutil.copyfileobj(src, dst)


class ABSAModel:
    def __init__(self, folder: str | Path = CHECKPOINT, device: str = "auto"):
        folder = Path(folder)
        weights = folder / "model.pt"
        if not weights.exists() or weights.stat().st_size < 1024:  # missing, or a Git LFS pointer
            if not CHECKPOINT_ZIP.exists():
                raise FileNotFoundError(f"{weights} has no weights: run `git lfs pull`, or put the notebook's zip at "
                                        f"{CHECKPOINT_ZIP}")
            unpack(folder)
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if (cfg["model"], cfg["variant"]) != ("bge", "detr"):
            raise ValueError(f"{folder} is {cfg['model']}_{cfg['variant']}; this pipeline runs bge_detr only")
        self.folder, self.epoch, self.tau, self.categories = folder, cfg["epoch"], cfg["tau"], cfg["categories"]
        entities = list(dict.fromkeys(c.split("#")[0] for c in self.categories))
        self.device = pick_device(device)
        self.inputs = Inputs(folder / "tokenizer.json", cfg["fixed"]["max_len"], cfg["fixed"]["ctx_len"])
        self.net = SetDetector(folder / "encoder_config.json", len(entities), len(self.categories) // len(entities),
                               cfg["fixed"]["queries"])
        state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        state.pop("entity_weight")  # class weights of the training loss
        # training wrapped the BERT in its tweak module as encoder.base.bert
        self.net.load_state_dict({k.replace("encoder.base.", "encoder.", 1): v for k, v in state.items()})
        self.net.to(self.device).eval()

    @torch.no_grad()
    def predict(self, sentences: list[Sentence], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Detection probabilities [n, C] and star probabilities [n, C, 5], in the order of `sentences`. Batches are
        sorted by length (less padding)."""
        det = np.zeros((len(sentences), len(self.categories)), np.float32)
        pol = np.zeros((len(sentences), len(self.categories), N_STARS), np.float32)
        use_amp = self.device.type == "cuda"
        dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        order = sorted(range(len(sentences)), key=lambda i: len(self.inputs.tokens(sentences[i].text)[0]))
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            b = {k: v.to(self.device) for k, v in self.inputs.batch([sentences[i] for i in idx]).items()}
            with torch.autocast(self.device.type, dtype=dtype, enabled=use_amp):
                d, p = self.net(b)
            det[idx], pol[idx] = d.float().cpu().numpy(), p.float().cpu().numpy()
        return det, pol
