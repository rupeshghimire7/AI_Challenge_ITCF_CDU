"""Post-processing: model probabilities -> the outputs the brief asks for.

star_summary(pol)                       star probabilities -> sentiment, star, 90% star interval, confidence
decode(det, pol, categories, floor)     per sentence, every category with detection probability >= floor, highest first,
                                        with its sentiment, star and confidence
entity_checks(entity, ...)              the checks below: is the opinion's entity explicit in the sentence, or NULL?
finalise(sentence, context, tau, ...)   the rules below, on one sentence (preprocess() record, its `gate` from
                                        src.domain_gate, its decoded `opinions`) -> its output tuples, at least one
rating(t), stars(t), face(t)            one tuple's display: 1-5 rating (star + 1), "★★★★☆", and a face for the CLI and
                                        CSV; empty without an opinion (stars only where there is a rating)
confidence_label(t)                     "high" / "medium" / "low" from the sentiment confidence and the 90% star interval
brief(tuples)                           the brief's format: {LAPTOP#OPERATION_PERFORMANCE, negative}, {NULL#GENERAL,
                                        negative}; a tuple without an opinion has no polarity: {LAPTOP#NULL}, {NULL#NULL}
opinion_rows(records), aspect_table(rows)   one row per tuple; the aspect-sentiment table of each product

Output rules (finalise). The brief: "When there is no explicit mention of the entity or attribute, E or A takes the value
NULL"; its examples: "Do not buy it!" -> {NULL#GENERAL, negative}, "It is the worst laptop ever." -> {LAPTOP#GENERAL,
negative}.
  off-domain       the gates rejected the sentence -> {NULL#NULL}, no polarity
  opinions         the model's opinions with detection probability >= tau (fitted on validation in training)
  second threshold a sentence that mentions a laptop part but has no opinion >= tau takes the opinions >= TAU_LOW
                   ("The hinge broke after two weeks." scores LAPTOP#QUALITY 0.82, just under tau 0.85)
  merged           opinions found by merging a {NULL#NULL} sentence with a neighbour (src.main), see `merged_opinions`
  no opinion       each entity the sentence mentions with attribute NULL ("Apple mac mini" -> {LAPTOP#NULL},
                   {COMPANY#NULL}); mentioning none -> {NULL#NULL}
  part re-map      (REMAP, off by default) the model's LAPTOP#A when the sentence does not name the laptop but names
                   exactly one part P by its word list, and the model has no opinion on P: P#A ("The hinge broke" ->
                   HARDWARE#QUALITY). Off because the gold data files broken parts under LAPTOP#QUALITY, as the model
                   learned: the re-map costs about 1 joint-F1 point
  entity checks    an opinion keeps its entity E when E is named in the sentence, or when the sentence mentions some
                   laptop part explicitly AND one of: E is mentioned in a neighbouring sentence (review context), E is
                   the whole LAPTOP, E is among the sentence's 3 closest labels. Otherwise the entity is NULL:
                   "Do not buy it!" mentions nothing -> NULL#GENERAL; '"premium" build' mentions build -> LAPTOP#QUALITY
Each tuple keeps `category` (the system's E#A, which the aspect table and the metrics use), `model_category` (what the
model said), `source`, `checks` (the entity checks, for the app's verbose panel) and `note` (why there is no opinion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POLARITIES

N_STARS = 5
BRACKET = np.array([0, 0, 1, 2, 2])  # star -> sentiment index: 0-1 negative, 2 neutral, 3-4 positive
LEVEL = 0.9  # star interval coverage
NULL_ASPECT = "NULL#NULL"
TAU_LOW = 0.8  # second threshold, fitted on validation by scripts/evaluate_system.py
REMAP = False  # part re-map: off, it costs about 1 joint-F1 point on validation and test (scripts/evaluate_system.py)
WHOLE = ("LAPTOP", "COMPANY")  # entities that are not a part of the laptop
_INTERVALS = np.array([(lo, hi) for lo in range(N_STARS) for hi in range(lo, N_STARS)])


def star_summary(p: np.ndarray) -> dict[str, np.ndarray]:
    """Star probabilities [..., 5] -> per prediction:
      expected   expected star (0-4, a float)
      lo, hi     90% star interval: the narrowest run of stars holding >= 90% of the probability
      sentiment  final sentiment index (POLARITIES): the bracket (0-1 negative, 2 neutral, 3-4 positive) holding the
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


def decode(det: np.ndarray, pol: np.ndarray, categories: list[str], floor: float) -> list[list[dict]]:
    """Per sentence, every category with detection probability (`score`) >= floor, highest first."""
    s = star_summary(pol)
    out = []
    for i in range(len(det)):
        ops = [{"category": categories[j], "sentiment": POLARITIES[int(s["sentiment"][i, j])], "star": int(s["star"][i, j]),
                "expected_star": round(float(s["expected"][i, j]), 2), "star_low": int(s["lo"][i, j]),
                "star_high": int(s["hi"][i, j]), "confidence": round(float(s["confidence"][i, j]), 3),
                "certain": bool(s["certain"][i, j]), "score": round(float(det[i, j]), 3)}
               for j in np.flatnonzero(det[i] >= floor)]
        out.append(sorted(ops, key=lambda o: -o["score"]))
    return out


def _no_opinion(aspect: str, note: str) -> dict:
    return {"aspect": aspect, "category": "", "model_category": "", "sentiment": "", "source": "", "checks": [],
            "note": note}


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": passed, "detail": detail}


def entity_checks(entity: str, mentions: dict[str, str], context: set[str], top_labels: list[str]) -> tuple[bool, list]:
    """-> (keep the entity, the checks). Rule at the top of this file."""
    checks = [
        _check("named", entity in mentions, mentions.get(entity, f"{entity} is not mentioned in the sentence")),
        _check("explicit target", bool(mentions), "mentions " + ", ".join(mentions) if mentions else
               "the sentence mentions no laptop part, only pronouns or opinion words"),
        _check("review context", entity in context, f"{entity} is {'' if entity in context else 'not '}mentioned in a "
               "neighbouring sentence"),
        _check("whole laptop", entity == "LAPTOP", "every part belongs to the laptop" if entity == "LAPTOP" else
               f"{entity} is one part, not the whole laptop"),
        _check("label similarity", entity in top_labels, "closest labels: " + (", ".join(top_labels) or "not computed")),
    ]
    keep = checks[0]["passed"] or (checks[1]["passed"] and any(c["passed"] for c in checks[2:]))
    return keep, checks


def finalise(sentence: dict, context: set[str], tau: float, tau_low: float | None = TAU_LOW,
             remap: bool = REMAP) -> list[dict]:
    """The output rules (top of this file) on one sentence -> its output tuples, at least one. `context`: the entities
    the neighbouring sentences of the same review mention."""
    gate, merged = sentence["gate"], sentence.get("merged_opinions")
    if not gate["about_laptop"] and not merged:
        return [_no_opinion(NULL_ASPECT, gate["reason"])]
    mentions = gate["mentions"]
    ops = merged or [o | {"source": "model"} for o in sentence["opinions"] if o["score"] >= tau]
    if not ops and mentions and tau_low is not None:
        ops = [o | {"source": f"second threshold {tau_low:.2f}"} for o in sentence["opinions"] if tau_low <= o["score"]]
    if not ops:
        return [_no_opinion(f"{e}#NULL", "no opinion") for e in mentions] or [_no_opinion(NULL_ASPECT, "no opinion")]
    parts = [e for e in sentence["entities"] if e not in WHOLE]  # named by the word lists
    have = {o["category"].split("#")[0] for o in ops}
    out = []
    for o in ops:
        entity, attribute = o["category"].split("#")
        category, extra = o["category"], []
        if remap and entity == "LAPTOP" and "LAPTOP" not in mentions and len(parts) == 1 and parts[0] not in have:
            entity, category = parts[0], f"{parts[0]}#{attribute}"
            extra = [_check("part re-map", True, f"the sentence names one part, {parts[0]}, and the model has no opinion "
                            f"on it: {o['category']} becomes {category}")]
        keep, checks = entity_checks(entity, mentions, context, gate["top_labels"])
        out.append(o | {"aspect": category if keep else f"NULL#{attribute}", "category": category,
                        "model_category": o["category"], "checks": extra + checks, "note": ""})
    return out


FACES = {"positive": "🙂", "neutral": "😐", "negative": "🙁"}
HIGH, MEDIUM = 0.8, 0.6  # sentiment confidence for "high" / "medium"


def rating(t: dict) -> int | None:
    return int(t["star"]) + 1 if t.get("sentiment") else None


def stars(t: dict) -> str:
    r = rating(t)
    return "★" * r + "☆" * (N_STARS - r) if r else ""


def face(t: dict) -> str:
    return FACES.get(t.get("sentiment", ""), "")


def confidence_label(t: dict) -> str:
    """high: confidence >= HIGH and the 90% star interval inside one sentiment; medium: confidence >= MEDIUM; else low."""
    if not t.get("sentiment"):
        return ""
    c = t["confidence"]
    return "high" if c >= HIGH and t["certain"] else "medium" if c >= MEDIUM else "low"


def brief(tuples: list[dict]) -> str:
    # dict.fromkeys de-duplicates: two implicit entities with the same attribute both become NULL#<attribute>
    return ", ".join(dict.fromkeys(f"{{{t['aspect']}, {t['sentiment']}}}" if t["sentiment"] else f"{{{t['aspect']}}}"
                                   for t in tuples))


KEY_COLUMNS = ["product", "review_id", "sentence_index", "sentence"]
OPINION_COLUMNS = ["aspect", "category", "model_category", "sentiment", "star", "expected_star", "star_low", "star_high",
                   "confidence", "certain", "score", "source", "note"]


def opinion_rows(records: list[dict]) -> pd.DataFrame:
    """Records (KEY_COLUMNS + `tuples` from finalise) -> one row per tuple, so every sentence is in the output."""
    rows = [{k: r[k] for k in KEY_COLUMNS} | t for r in records for t in r["tuples"]]
    stars = {"star": "Int64", "star_low": "Int64", "star_high": "Int64"}  # stay integers next to the no-opinion rows
    return pd.DataFrame(rows, columns=KEY_COLUMNS + OPINION_COLUMNS).astype(stars)


def aspect_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per product and category (the model's E#A: an implicit "it" is still the laptop): mentions (sentence opinions) by
    sentiment, the reviews that mention it, and a rating on 1-5 stars: 1 + the mean over those reviews of each review's
    mean expected star (0-4), so a review that mentions the battery in five sentences votes once. Most-mentioned first."""
    op = rows[rows.category.fillna("") != ""]
    keys = ["product", "category"]
    per_review = op.groupby([*keys, "review_id"]).expected_star.mean()
    t = pd.crosstab([op["product"], op.category], op.sentiment).reindex(columns=list(POLARITIES), fill_value=0)
    t.insert(0, "mentions", t.sum(1))
    t["reviews"] = per_review.groupby(keys).size()
    t["rating"] = (1 + per_review.groupby(keys).mean()).round(1)
    t["stars"] = [("★" * int(r + 0.5)).ljust(5, "☆") for r in t.rating]
    return t.reset_index().sort_values(["product", "mentions", "rating"], ascending=[True, False, False],
                                       ignore_index=True)
