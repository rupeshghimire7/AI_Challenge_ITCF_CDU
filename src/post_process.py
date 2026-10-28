"""Post-processing: model probabilities -> the outputs the brief asks for.

star_summary(pol)                   star probabilities -> sentiment, star, 90% star interval, confidence (as in training)
decode(det, pol, categories, tau)   per sentence, the model's opinions: every category with detection probability >= tau,
                                    with its sentiment, star and confidence
finalise(opinions, sentence)        the rules below, on one sentence (preprocess() record + its `gate` from
                                    src.domain_gate) -> its output tuples
rating(t), stars(t), face(t)        one tuple's display: 1-5 rating (star + 1), "★★★★☆" and 🙂 / 😐 / 🙁; empty without an
                                    opinion (stars only where there is a rating)
confidence_label(t)                 "high" / "medium" / "low" from the sentiment confidence and the 90% star interval
brief(tuples)                       the brief's format: {LAPTOP#OPERATION_PERFORMANCE, negative}, {NULL#GENERAL, negative};
                                    a tuple without an opinion has no polarity: {LAPTOP#NULL}, {NULL#NULL}
opinion_rows(records)               sentence records -> one row per tuple
aspect_table(rows)                  the aspect-sentiment table of each product: mentions by sentiment and a 1-5 star rating

Domain gates: src.domain_gate runs them before the model (only a sentence about the laptop reaches it). The model was
trained on laptop reviews only and reads any opinion as a laptop opinion, so a sentence the gates reject is {NULL#NULL}.

Output rules (finalise); the brief: "When there is no explicit mention of the entity or attribute, E or A takes the value
NULL".
  off-domain  a gate says the sentence is not about the laptop -> {NULL#NULL}, no polarity
  no opinion  nothing above the threshold: each entity the sentence names, with attribute NULL and no polarity
              ("Apple mac mini" -> {LAPTOP#NULL}, {COMPANY#NULL}); naming none -> {NULL#NULL}
  implicit    an opinion whose entity the sentence does not name gets the aspect NULL#<attribute> ("Do not buy it!" ->
              {NULL#GENERAL, negative}); `category` keeps the model's E#A, which the metrics and the table use
Every tuple has a `note`: empty for an opinion, otherwise why the sentence has none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POLARITIES

N_STARS = 5
BRACKET = np.array([0, 0, 1, 2, 2])  # star -> sentiment index: 0-1 negative, 2 neutral, 3-4 positive
LEVEL = 0.9  # star interval coverage
NULL_ASPECT = "NULL#NULL"
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


def decode(det: np.ndarray, pol: np.ndarray, categories: list[str], tau: float) -> list[list[dict]]:
    """Per sentence, its opinions, highest detection probability (`score`) first."""
    s = star_summary(pol)
    out = []
    for i in range(len(det)):
        ops = [{"category": categories[j], "sentiment": POLARITIES[int(s["sentiment"][i, j])], "star": int(s["star"][i, j]),
                "expected_star": round(float(s["expected"][i, j]), 2), "star_low": int(s["lo"][i, j]),
                "star_high": int(s["hi"][i, j]), "confidence": round(float(s["confidence"][i, j]), 3),
                "certain": bool(s["certain"][i, j]), "score": round(float(det[i, j]), 3)}
               for j in np.flatnonzero(det[i] >= tau)]
        out.append(sorted(ops, key=lambda o: -o["score"]))
    return out


def _no_opinion(aspect: str, note: str) -> dict:
    return {"aspect": aspect, "category": "", "sentiment": "", "note": note}


def finalise(opinions: list[dict], sentence: dict) -> list[dict]:
    """The output rules (top of this file) on one sentence -> its output tuples, at least one."""
    if not sentence["gate"]["about_laptop"]:
        return [_no_opinion(NULL_ASPECT, sentence["gate"]["reason"])]
    if not opinions:
        return [_no_opinion(f"{e}#NULL", "no opinion") for e in sentence["entities"]] or [_no_opinion(NULL_ASPECT, "no opinion")]
    named = set(sentence["entities"])
    return [o | {"aspect": o["category"] if o["category"].split("#")[0] in named else "NULL#" + o["category"].split("#")[1],
                 "note": ""} for o in opinions]


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
OPINION_COLUMNS = ["aspect", "category", "sentiment", "star", "expected_star", "star_low", "star_high", "confidence",
                   "certain", "score", "note"]


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
