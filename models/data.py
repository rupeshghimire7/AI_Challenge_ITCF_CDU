"""Data helper for the model phase: everything between DATA/final and a model's input tensors, and back to tuples.

make_final()          {train,test}_merged.csv -> {train,test}_final.csv, only the columns the task needs
load_examples()       a *_final.csv -> one Example per sentence, gold = its (entity, attribute, star) tuples
hub_file(), TokenSpec model files from the Hugging Face cache first; how a model frames its input (special tokens)
Inputs                text -> token windows. Text longer than one window is split into overlapping windows and every
                      window is scored (heads.merge_windows joins them): nothing is ever truncated, at any length
long_examples()       consecutive sentences of a review joined into one input: trains the long-text path
star_summary()        5 star probabilities -> star, 90% star interval, final sentiment and its confidence
Scorer                probabilities -> (category, sentiment) tuples and every metric

Polarity is a 0-4 star scale. SemEval's three labels are stored as stars: negative 0, neutral 2, positive 4. Training
targets are smoothed onto the neighbouring stars (STAR_TARGET), so the model gives a full 5-star distribution. Stars
are read back as sentiment by bracket: 0-1 negative, 2 neutral, 3-4 positive; the final sentiment, its confidence, the
predicted star and a 90% star interval come from star_summary().

Final columns (the brief: sentence-level E#A + polarity, "taking into account the context of the review"):
  sentence_id, review_id   a sentence's id (texts repeat across reviews, so text is no key) and its review
  split                train / validation (15% of the train reviews, whole reviews) / test (the official gold set)
  text                 the sentence, already normalised by src.preprocess.normalise
  prev_text, next_text neighbouring sentences of the same review (review context)
  category, attribute  the entity (LAPTOP, BATTERY, ...) and attribute (GENERAL, QUALITY, ...) of one opinion
  polarity             its star: 0 (negative), 2 (neutral), 4 (positive). A sentence without opinions has one row with
                       category, attribute and polarity empty
Run: python models/data.py    (writes the two CSVs and reports/final_data.txt)
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FINAL_DIR, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.nuance_checker import NUANCES  # noqa: E402
from src.preprocess import normalise  # noqa: E402

FINAL_COLUMNS = ["sentence_id", "review_id", "split", "text", "prev_text", "next_text", "category", "attribute", "polarity"]
SENTIMENTS = POLARITIES  # ("negative", "neutral", "positive")
N_STARS = 5
STAR_OF = {"negative": 0, "neutral": 2, "positive": 4}  # SemEval's three labels on the 0-4 scale
BRACKET = np.array([0, 0, 1, 2, 2])  # star -> sentiment index: 0-1 negative, 2 neutral, 3-4 positive
STAR_TARGET = torch.tensor([[0.75, 0.25, 0.0, 0.0, 0.0],  # training target per gold sentiment (ordinal smoothing)
                            [0.0, 0.125, 0.75, 0.125, 0.0],
                            [0.0, 0.0, 0.0, 0.25, 0.75]])
VALIDATION = 15  # percent of train reviews held out for validation
LEVEL = 0.9  # star interval coverage


def validation_review(review_id: str) -> bool:
    """Deterministic, review-grouped: all sentences of a review are on the same side."""
    return int(hashlib.sha1((review_id + "validation").encode()).hexdigest(), 16) % 100 < VALIDATION


# ----------------------------------------------------------------------------------------------- final CSVs
def make_final() -> list[str]:
    """Write {train,test}_final.csv from scripts/prepare_data.py's *_merged.csv and return the printed summary."""
    rep = ["FINAL MODEL DATA (models/data.py)", "", f"columns: {', '.join(FINAL_COLUMNS)}",
           "polarity as stars: negative 0, neutral 2, positive 4", ""]
    for split in ("train", "test"):
        src = FINAL_DIR / f"{split}_merged.csv"
        if not src.exists():
            raise SystemExit(f"{src.relative_to(ROOT)} not found: run python scripts/prepare_data.py first")
        df = pd.read_csv(src, dtype=str, keep_default_na=False)
        bad = sorted(set(df.polarity) - set(STAR_OF) - {""})
        assert not bad, f"{split}: unexpected polarity values {bad}"
        out = pd.DataFrame({
            "sentence_id": df.sentence_id, "review_id": df.review_id,
            "split": df.review_id.map(lambda r: "validation" if validation_review(r) else "train") if split == "train" else "test",
            "text": df.text, "prev_text": df.prev_text, "next_text": df.next_text, "category": df.entity,
            "attribute": df.attribute, "polarity": df.polarity.map(lambda p: str(STAR_OF[p]) if p else "")})[FINAL_COLUMNS]
        has = out.category != ""
        assert not out.duplicated().any(), f"{split}: duplicate opinion rows"
        assert ((out.attribute != "") == has).all() and ((out.polarity != "") == has).all(), f"{split}: half-empty opinion row"
        assert (out.text != "").all() and (out.text.map(normalise) == out.text).all(), f"{split}: empty or unnormalised text"
        assert not set(out.review_id[out.split == "train"]) & set(out.review_id[out.split == "validation"]), "review in two splits"
        out.to_csv(FINAL_DIR / f"{split}_final.csv", index=False)
        rep.append(f"{split}_final.csv")
        for part, g in out.groupby("split", sort=False):
            h = g.category != ""
            n_sent = g.sentence_id.nunique()
            empty = n_sent - g.sentence_id[h].nunique()
            stars = Counter(g.polarity[h])
            rep.append(f"  {part:<10} reviews {g.review_id.nunique():>3}  sentences {n_sent:>4}  opinions {int(h.sum()):>4}  "
                       f"without opinion {empty:>3} ({empty / n_sent:.1%})  categories {(g.category[h] + '#' + g.attribute[h]).nunique():>2}  "
                       f"stars " + " ".join(f"{s}:{stars[s]}" for s in "01234"))
        rep.append("")
    return rep


@dataclass
class Example:
    sid: str
    split: str
    text: str
    prev: str = ""
    next: str = ""
    gold: tuple = ()  # ((entity, attribute, star), ...), unique, sorted

    @property
    def review(self) -> str:
        return self.sid.rsplit(":", 1)[0]

    @property
    def categories(self) -> set[str]:
        return {f"{e}#{a}" for e, a, _ in self.gold}


def load_examples(split: str) -> list[Example]:
    """One Example per sentence of DATA/final/{split}_final.csv (built from *_merged.csv if missing), in file order."""
    path = FINAL_DIR / f"{split}_final.csv"
    if not path.exists():
        make_final()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out: dict[str, Example] = {}
    gold: dict[str, set] = defaultdict(set)
    for r in df.itertuples(index=False):
        if r.sentence_id not in out:
            out[r.sentence_id] = Example(r.sentence_id, r.split, r.text, r.prev_text, r.next_text)
        if r.category:
            gold[r.sentence_id].add((r.category, r.attribute, int(r.polarity)))
    for sid, ex in out.items():
        ex.gold = tuple(sorted(gold[sid]))
    return list(out.values())


def by_review(examples: list[Example]) -> list[list[Example]]:
    """Sentences grouped by review, in sentence order."""
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[ex.review].append(ex)
    return [sorted(g, key=lambda e: int(e.sid.rsplit(":", 1)[1])) for g in groups.values()]


def _joined(part: list[Example], sid: str, run_on: bool) -> Example:
    texts = [re.sub(r"[.!?]+$", "", p.text) if run_on else p.text for p in part]
    gold = tuple(sorted({g for p in part for g in p.gold}))
    return Example(sid, "long", " ".join(texts), part[0].prev, part[-1].next, gold)


def long_examples(examples: list[Example], n: int, rng: random.Random, max_sents: int = 6,
                  max_opinions: int = 16) -> list[Example]:
    """`n` long training inputs: 2..max_sents consecutive sentences of one review joined into one text, labels = the union
    of theirs. Half are run-ons (sentence-final punctuation removed), like an unsplit Amazon paragraph. Built only from
    the `examples` given (the training split), so no validation or test sentence leaks in."""
    reviews = [r for r in by_review(examples) if len(r) >= 2]
    out: list[Example] = []
    while reviews and len(out) < n:
        r = rng.choice(reviews)
        k = rng.randint(2, min(max_sents, len(r)))
        s = rng.randint(0, len(r) - k)
        ex = _joined(r[s : s + k], f"long/{r[s].sid}+{k}", rng.random() < 0.5)
        if len(ex.gold) <= max_opinions:
            out.append(ex)
    return out


# ----------------------------------------------------------------------------------------------- tokens and windows
def hub_file(repo: str, filename: str) -> str:
    """A model file from the local Hugging Face cache; the Hub is contacted only if it was never downloaded."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename)


@dataclass(frozen=True)
class TokenSpec:
    """How one model frames its input. Tokens are named, and their ids are looked up in the model's own tokenizer."""
    repo: str
    prefix: tuple[str, ...]  # before the text ([CLS] / <|startoftext|>)
    suffix: tuple[str, ...]  # after the text ([SEP] / nothing)
    sep: tuple[str, ...]  # between a context sentence and the target sentence (L4)
    pad: str
    max_positions: int  # longest window the model can take (position limit or trained context)
    pooling: str  # "cls" or "mean" (the model's documented sentence pooling, plan M2)


@dataclass
class Batch:
    input_ids: torch.Tensor  # [W, T] one row per window
    attention_mask: torch.Tensor  # [W, T] 1 = real token, 0 = padding
    type_ids: torch.Tensor  # [W, T] 0 = target, 1 = review context, 2 = target token in a negation scope
    target_mask: torch.Tensor  # [W, T] True = a token of the target sentence (specials, padding, context excluded)
    owner: torch.Tensor  # [W] index of the example each window belongs to
    examples: list[Example]

    def to(self, device) -> "Batch":
        for k in ("input_ids", "attention_mask", "type_ids", "target_mask", "owner"):
            setattr(self, k, getattr(self, k).to(device, non_blocking=True))
        return self


class Inputs:
    """Text -> model windows for one model. Every text is normalised first (src.preprocess.normalise, the function
    training data went through), so training and serving see identical strings.

    A window is  prefix [context sep] target-chunk [sep context] suffix,  at most `max_len` tokens plus the context.
    A target longer than one window is cut into chunks that overlap by a quarter, so an opinion phrase cut at a
    boundary is whole in the next window. Special tokens and context never count as target tokens: pooling and the
    heads only read `target_mask`, which is where special tokens are "removed"."""

    def __init__(self, spec: TokenSpec, max_len: int = 56, ctx_len: int = 32, context: bool = False,
                 neg_scope: bool = False, tokenizer_file: str | None = None):
        from tokenizers import Tokenizer

        self.spec, self.context, self.neg_scope, self.ctx_len = spec, context, neg_scope, ctx_len
        self.tokenizer_file = tokenizer_file or hub_file(spec.repo, "tokenizer.json")  # a checkpoint ships its own copy
        self.tok = Tokenizer.from_file(self.tokenizer_file)
        self.tok.no_truncation()
        self.tok.no_padding()
        self.prefix, self.suffix, self.sep = (list(map(self.token_id, spec.prefix)), list(map(self.token_id, spec.suffix)),
                                              list(map(self.token_id, spec.sep)))
        self.pad = self.token_id(spec.pad)
        self.window = max_len - len(self.prefix) - len(self.suffix)
        self.step = self.window - self.window // 4
        longest = max_len + (2 * (ctx_len + len(self.sep)) if context else 0)
        if self.window < 8 or longest > spec.max_positions:
            raise ValueError(f"max_len {max_len} (+ context) must fit {spec.max_positions} positions with room for text")
        self._cache: dict[str, tuple[list[int], list[int]]] = {}

    def token_id(self, token: str) -> int:
        i = self.tok.token_to_id(token)
        if i is None:
            raise KeyError(f"{token!r} is not in the {self.spec.repo} tokenizer")
        return i

    def tokens(self, text: str) -> tuple[list[int], list[int]]:
        """Token ids of the normalised text, without special tokens, and their type ids (2 inside a negation scope)."""
        if text in self._cache:
            return self._cache[text]
        clean = normalise(text)
        enc = self.tok.encode(clean, add_special_tokens=False)
        types = [0] * len(enc.ids)
        if self.neg_scope:
            spans = negation_spans(clean)
            types = [2 if any(s < b and a < e for s, e in spans) else 0 for a, b in enc.offsets]
        if len(self._cache) < 100_000:  # bounded: serving never grows it without limit
            self._cache[text] = (enc.ids, types)
        return enc.ids, types

    def windows(self, ex: Example, drop_prev: bool = False, drop_next: bool = False) -> list[tuple[list, list, list]]:
        """(ids, type ids, target mask) of every window of one example. An empty text still gets one window of special
        tokens only, so every example has an output."""
        ids, types = self.tokens(ex.text)
        prev = self.tokens(ex.prev)[0][-self.ctx_len :] if self.context and ex.prev and not drop_prev else []
        nxt = self.tokens(ex.next)[0][: self.ctx_len] if self.context and ex.next and not drop_next else []
        head = self.prefix + (prev + self.sep if prev else [])
        tail = (self.sep + nxt if nxt else []) + self.suffix
        head_t = [0] * len(self.prefix) + ([1] * len(prev) + [0] * len(self.sep) if prev else [])
        tail_t = ([0] * len(self.sep) + [1] * len(nxt) if nxt else []) + [0] * len(self.suffix)
        starts = [0]
        while starts[-1] + self.window < len(ids):
            starts.append(starts[-1] + self.step)
        out = []
        for s in starts:
            chunk, chunk_t = ids[s : s + self.window], types[s : s + self.window]
            out.append((head + chunk + tail, head_t + chunk_t + tail_t,
                        [False] * len(head) + [True] * len(chunk) + [False] * len(tail)))
        return out

    def batch(self, examples: list[Example], rng: random.Random | None = None, ctx_dropout: float = 0.0) -> Batch:
        """Padded tensors for a list of examples. With `rng` (training), each context side is emptied with probability
        `ctx_dropout` (L4 context dropout), so the model also works when a sentence comes without its review."""
        rows, owner = [], []
        for i, ex in enumerate(examples):
            dp = rng is not None and rng.random() < ctx_dropout
            dn = rng is not None and rng.random() < ctx_dropout
            for w in self.windows(ex, dp, dn):
                rows.append(w)
                owner.append(i)
        n, t = len(rows), max((len(r[0]) for r in rows), default=1)
        ids = torch.full((n, t), self.pad, dtype=torch.long)
        mask = torch.zeros((n, t), dtype=torch.long)
        types = torch.zeros((n, t), dtype=torch.long)
        target = torch.zeros((n, t), dtype=torch.bool)
        for j, (i_, t_, m_) in enumerate(rows):
            ids[j, : len(i_)] = torch.tensor(i_)
            mask[j, : len(i_)] = 1
            types[j, : len(i_)] = torch.tensor(t_)
            target[j, : len(i_)] = torch.tensor(m_)
        return Batch(ids, mask, types, target, torch.tensor(owner, dtype=torch.long), examples)

    def n_tokens(self, ex: Example) -> int:
        return len(self.tokens(ex.text)[0])


def negation_spans(text: str) -> list[tuple[int, int]]:
    """D6.3 negation scope as character spans: the cue (src.nuance_checker's negation pattern) and up to 5 following
    words, stopping at punctuation or a contrast marker ("not bad", "never had a problem", "no issues")."""
    stop = re.compile(r"[.,;:!?]|" + NUANCES["contrast"].pattern, re.I)
    spans = []
    for m in NUANCES["negation"].finditer(text):
        end_m = stop.search(text, m.end())
        limit = end_m.start() if end_m else len(text)
        words = list(re.finditer(r"\S+", text[m.end() : limit]))[:5]
        spans.append((m.start(), m.end() + words[-1].end() if words else m.end()))
    return spans


# ----------------------------------------------------------------------------------------------- stars and metrics
_INTERVALS = np.array([(lo, hi) for lo in range(N_STARS) for hi in range(lo, N_STARS)])


def star_summary(p: np.ndarray) -> dict[str, np.ndarray]:
    """Star probabilities [..., 5] -> per prediction:
      expected   expected star (0-4, a float)
      lo, hi     90% star interval: the narrowest run of stars holding >= 90% of the probability
      sentiment  final sentiment index (SENTIMENTS): the bracket (0-1 negative, 2 neutral, 3-4 positive) holding the
                 interval when it sits inside one bracket; otherwise the bracket with the most probability, except that
                 an interval spanning negative and positive stars with no bracket above 50% is a mixed opinion: neutral
      confidence probability of the final sentiment's bracket
      star       the predicted star (0 1 2 3 4): the expected star rounded and kept inside the final sentiment's
                 bracket, so star and sentiment always agree (1 = leaning negative, 3 = leaning positive)
      certain    the 90% interval sits inside one bracket"""
    p = np.asarray(p, dtype=np.float64)
    expected = (p * np.arange(N_STARS)).sum(-1)
    cum = np.concatenate([np.zeros(p.shape[:-1] + (1,)), p.cumsum(-1)], -1)
    mass = cum[..., _INTERVALS[:, 1] + 1] - cum[..., _INTERVALS[:, 0]]  # [..., 15]
    width = _INTERVALS[:, 1] - _INTERVALS[:, 0]
    k = np.argmax((mass >= LEVEL - 1e-6) * 100.0 - width * 10.0 + mass, -1)  # narrowest, then most probable
    lo, hi = _INTERVALS[k, 0], _INTERVALS[k, 1]
    brackets = np.stack([p[..., 0] + p[..., 1], p[..., 2], p[..., 3] + p[..., 4]], -1)
    certain = BRACKET[lo] == BRACKET[hi]
    sentiment = np.where(certain, BRACKET[lo], brackets.argmax(-1))
    sentiment = np.where(~certain & (brackets.max(-1) < 0.5) & (lo <= 1) & (hi >= 3), 1, sentiment)
    confidence = np.take_along_axis(brackets, sentiment[..., None], -1)[..., 0]
    star = np.clip(np.rint(expected), np.array([0, 2, 3])[sentiment], np.array([1, 2, 4])[sentiment]).astype(int)
    return {"expected": expected, "star": star, "lo": lo, "hi": hi, "sentiment": sentiment, "confidence": confidence,
            "certain": certain}


def _prf(tp: float, n_pred: float, n_gold: float) -> tuple[float, float, float]:
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


class Scorer:
    """Metrics of category probabilities `det` [n, C] and star probabilities `pol` [n, C, 5] for `examples`.

    detect_*     category detection P/R/F1 over unique (sentence, category): the official SemEval-2016 Slot1
    entity_f1, attribute_f1   the same, for the entity part and the attribute part alone
    sentiment_*  final sentiment accuracy and macro-F1 on the gold categories (official Slot3, phase B)
    joint_*      P/R/F1 over (sentence, category, sentiment): the headline metric
    empty_acc    share of opinion-less sentences given no opinion
    star_mae     mean |predicted star - gold star| on the gold categories (gold stars are 0, 2, 4)
    interval_cover   share of gold stars inside the 90% star interval
    confidence   mean confidence of the final sentiment on the gold categories
    """

    TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
    KEYS = ["detect_p", "detect_r", "detect_f1", "entity_f1", "attribute_f1", "sentiment_acc", "sentiment_macro_f1",
            "joint_p", "joint_r", "joint_f1", "empty_acc", "star_mae", "interval_cover", "confidence"]

    def __init__(self, examples: list[Example], categories: list[str]):
        self.examples, self.categories = examples, list(categories)
        self.index = {c: i for i, c in enumerate(self.categories)}
        self.gold_cats = [ex.categories for ex in examples]
        self.gold = [{(f"{e}#{a}", int(BRACKET[s]), s) for e, a, s in ex.gold} for ex in examples]

    def decode(self, det: np.ndarray, summary: dict, tau: float) -> list[set[tuple[str, int]]]:
        """Predicted (category, sentiment index) per example: every category with probability >= tau."""
        return [{(self.categories[j], int(summary["sentiment"][i, j])) for j in np.flatnonzero(det[i] >= tau)}
                for i in range(len(det))]

    def score(self, det: np.ndarray, pol: np.ndarray, tau: float, summary: dict | None = None) -> dict:
        summary = summary if summary is not None else star_summary(pol)
        pred = self.decode(det, summary, tau)
        c = np.zeros((4, 3))  # tp, n_pred, n_gold for: category, entity, attribute, joint
        y_true, y_pred, err, cover, conf = [], [], [], [], []
        empty = []
        for i, (pi, gi) in enumerate(zip(pred, self.gold)):
            pc, gc = {k for k, _ in pi}, self.gold_cats[i]
            for row, f in enumerate((lambda k: k, lambda k: k.split("#")[0], lambda k: k.split("#")[1])):
                a, b = {f(k) for k in pc}, {f(k) for k in gc}
                c[row] += (len(a & b), len(a), len(b))
            gt = {(k, s) for k, s, _ in gi}
            c[3] += (len(pi & gt), len(pi), len(gt))
            if not gi:
                empty.append(not pi)
            for k, s, star in gi:
                j = self.index.get(k)  # a category outside the model's label space counts as wrong
                y_true.append(s)
                y_pred.append(int(summary["sentiment"][i, j]) if j is not None else -1)
                if j is not None:
                    err.append(abs(int(summary["star"][i, j]) - star))
                    cover.append(summary["lo"][i, j] <= star <= summary["hi"][i, j])
                    conf.append(float(summary["confidence"][i, j]))
        out = {"n": len(det), "tau": float(tau)}
        for name, row in (("detect", 0), ("joint", 3)):
            out[f"{name}_p"], out[f"{name}_r"], out[f"{name}_f1"] = _prf(*c[row])
        out["entity_f1"], out["attribute_f1"] = _prf(*c[1])[2], _prf(*c[2])[2]
        out["sentiment_acc"] = float(np.mean(np.equal(y_true, y_pred))) if y_true else 0.0
        out["sentiment_macro_f1"] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)) if y_true else 0.0
        out["empty_acc"] = float(np.mean(empty)) if empty else 0.0
        out["star_mae"] = float(np.mean(err)) if err else 0.0
        out["interval_cover"] = float(np.mean(cover)) if cover else 0.0
        out["confidence"] = float(np.mean(conf)) if conf else 0.0
        return out

    def best_tau(self, det: np.ndarray, pol: np.ndarray) -> float:
        """The global detection threshold with the best joint F1 (fitted on validation, never on test)."""
        summary = star_summary(pol)
        return float(max(self.TAUS, key=lambda t: self.score(det, pol, t, summary)["joint_f1"]))


if __name__ == "__main__":
    report = make_final()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "final_data.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"saved {(FINAL_DIR / 'train_final.csv').relative_to(ROOT)}, {(FINAL_DIR / 'test_final.csv').relative_to(ROOT)}, "
          f"{(REPORT_DIR / 'final_data.txt').relative_to(ROOT)}")
