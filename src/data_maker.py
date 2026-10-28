"""Load the prepared data in DATA/final: one row per opinion (long format), or one record per sentence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

BOOL_COLUMNS = ("out_of_scope", "polarity_conflict", "has_opinion", "elongated")
INT_COLUMNS = ("fold", "sentence_index", "review_n_sentences", "n_opinions")
OPINION_COLUMNS = ("aspect_category", "entity", "attribute", "polarity", "polarity_conflict")


def load_long(path: str | Path) -> pd.DataFrame:
    """Read a DATA/final CSV (one row per opinion; a sentence without opinions has one row with an empty category).
    Empty fields stay "" (not NaN); booleans and integers get their types back."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in BOOL_COLUMNS:
        if c in df:
            df[c] = df[c].map({"True": True, "False": False}).astype(bool)
    for c in INT_COLUMNS:
        if c in df:
            df[c] = pd.to_numeric(df[c].replace("", None)).astype("Int64")
    return df


def to_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """One record per sentence, in file order: the sentence-level columns plus `opinions`, a list of (category, polarity)."""
    ops: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sid, cat, pol in zip(df.sentence_id, df.aspect_category, df.polarity):
        if cat:
            ops[sid].append((cat, pol))
    out = df.drop_duplicates("sentence_id").drop(columns=[c for c in OPINION_COLUMNS if c in df]).reset_index(drop=True)
    out["opinions"] = [ops.get(s, []) for s in out.sentence_id]
    return out
