"""Stage 2 of data preparation: clean DATA/final/{train,test}_merged.csv into {train,test}_merged_cleaned.csv.

Steps, in order (counted in the report; every removed sentence is listed in reports/removed_sentences.csv):
C1 Normalise text with src.preprocess.normalise (D2.4, the function serving uses): text, context and ACOS terms.
   src.nuance_checker counts the artefacts repaired; `elongated` = a letter run was capped ("soooo" -> "sooo").
C2 Out-of-scope opinions. ACOS's Out_Of_Scope#* opinions are dropped (D2.2). A sentence left without opinions becomes
   an opinion-less sentence with out_of_scope=True, which is exactly what SemEval's OutOfScope attribute means: the
   sentence holds an opinion, but about something outside the laptop inventory (another product, a comparison, a
   brand in general), so its gold label set is empty.
C3 Same text, different labels, same split. SemEval vs ACOS: the ACOS copy is dropped (SemEval is the target
   annotation scheme). ACOS vs ACOS: every copy is dropped (no review context can explain the difference, so one of
   the labels is noise and we cannot tell which). SemEval vs SemEval: both kept, because SemEval labels depend on the
   review context. Copies that C2 made identical are kept once.
C4 Train/test leakage (D2.5). Character 5-gram Jaccard between every train and every test sentence, computed exactly
   (MinHash would only approximate it). Counts are reported at exact / >= 0.9 / >= 0.8; train sentences at >= 0.8 are
   removed. Test sentences are never removed for leakage, so T1 stays the official SemEval test set.
C5 Derived columns: polarity_conflict (one category, two polarities in one sentence; both rows kept, D2.3),
   has_opinion, n_opinions, fold (D3.1, SemEval train only; -1 = not in CV: ACOS auxiliary rows and all test rows),
   nuances (D5.1 text slices from src.nuance_checker, "|"-joined).
C6 Integrity checks; the script stops if any fails.

Outputs: DATA/final/{train,test}_merged_cleaned.csv, DATA/final/label_space.json,
         DATA/final/reports/clean_data.txt (the printed report), DATA/final/reports/removed_sentences.csv
Run:     python scripts/clean_data.py      (after prepare_data.py)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTRIBUTES, ENTITIES, FINAL_DIR, OUT_OF_SCOPE, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.data_maker import load_long, to_sentences  # noqa: E402
from src.nuance_checker import ARTEFACTS, artefacts, nuances  # noqa: E402
from src.preprocess import match_key, normalise  # noqa: E402

OUT = FINAL_DIR
LEAK_THRESHOLD, LEAK_REPORTED = 0.8, (1.0, 0.9, 0.8)
COLUMNS = ["sentence_id", "source", "split", "fold", "review_id", "sentence_index", "review_n_sentences", "text", "aspect_category",
           "entity", "attribute", "polarity", "has_opinion", "n_opinions", "out_of_scope", "polarity_conflict", "elongated",
           "nuances", "prev_text", "next_text", "aspect_terms", "opinion_terms", "text_raw"]
STEPS: list[list] = []  # [step, split, source, sentences, rows, what]
REMOVED: list[dict] = []  # one record per removed sentence


def log(step: str, split: str, df: pd.DataFrame, mask: pd.Series, what: str) -> None:
    for src, n_rows in Counter(df.source[mask]).items():
        STEPS.append([step, split, src, df.sentence_id[mask & (df.source == src)].nunique(), n_rows, what])


def remove(df: pd.DataFrame, ids: dict[str, tuple[str, str, float | str]], step: str, split: str) -> pd.DataFrame:
    """Drop whole sentences. ids: sentence_id -> (reason, matched sentence_id, score)."""
    mask = df.sentence_id.isin(ids)
    for sid, text, src in df.loc[mask, ["sentence_id", "text", "source"]].drop_duplicates("sentence_id").itertuples(index=False):
        reason, other, score = ids[sid]
        REMOVED.append({"step": step, "split": split, "source": src, "sentence_id": sid, "reason": reason, "matched_sentence_id": other,
                        "similarity": score, "text": text})
    log(step, split, df, mask, "sentence removed: " + "; ".join(sorted({r for r, _, _ in ids.values()})))
    return df[~mask].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- C1 normalisation
def normalise_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    found = {t: artefacts(t) for t in df.text.unique()}
    for name in ARTEFACTS:
        log("C1", split, df, df.text.map(lambda t: name in found[t]), f"normalised: {name}")
    df["elongated"] = df.text.map(lambda t: "elongation" in found[t])
    for col in ("text", "prev_text", "next_text", "aspect_terms", "opinion_terms"):
        df[col] = df[col].map(normalise)
    if (df.text == "").any():
        sys.exit(f"C1: {int((df.text == '').sum())} {split} rows have an empty text after normalisation")
    return df


# ----------------------------------------------------------------------------------------------- C2 out of scope
def resolve_out_of_scope(df: pd.DataFrame, split: str) -> pd.DataFrame:
    order = {sid: i for i, sid in enumerate(df.sentence_id.drop_duplicates())}
    oos = df.entity == OUT_OF_SCOPE
    keep = df[~oos]
    emptied = set(df.sentence_id[oos]) - set(keep.sentence_id)
    mixed = df.sentence_id.isin(set(df.sentence_id[oos]) - emptied)
    log("C2", split, df, oos & ~df.sentence_id.isin(emptied) & mixed, "Out_Of_Scope opinion dropped; the sentence keeps its other opinions")
    log("C2", split, df, oos & df.sentence_id.isin(emptied), "Out_Of_Scope opinion dropped; sentence kept without opinions, out_of_scope=True")
    blank = {"aspect_category": "", "entity": "", "attribute": "", "polarity": "", "aspect_terms": "", "opinion_terms": "", "out_of_scope": True}
    added = df[oos & df.sentence_id.isin(emptied)].drop_duplicates("sentence_id").assign(**blank)
    out = pd.concat([keep, added], ignore_index=True)
    return out.iloc[np.argsort(out.sentence_id.map(order).to_numpy(), kind="stable")].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- C3 label conflicts
def resolve_same_text(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, int]:
    s = to_sentences(df)
    s["key"] = [match_key(t) or t for t in s.text]  # a text without letters or digits (":(") compares verbatim
    s["labels"] = [(frozenset(o), oos) for o, oos in zip(s.opinions, s.out_of_scope)]
    drop: dict[str, tuple[str, str, float | str]] = {}
    context_dependent = 0
    for _, g in s.groupby("key", sort=False):
        if len(g) < 2:
            continue
        first = g.sentence_id.iloc[0]
        if g.labels.nunique() == 1:  # identical: only possible after C2 emptied an ACOS sentence
            drop |= {sid: ("identical copy after Out_Of_Scope removal", first, 1.0) for sid in g.sentence_id.iloc[1:]}
            continue
        sem, acos = g[g.source == "semeval16"], g[g.source == "acos"]
        if len(sem):
            drop |= {sid: ("same text as a SemEval sentence with different labels (SemEval kept)", sem.sentence_id.iloc[0], 1.0)
                     for sid in acos.sentence_id}
            context_dependent += len(sem) > 1 and sem.labels.nunique() > 1
        else:
            drop |= {sid: ("ACOS copies of one text disagree on labels (all copies dropped)", ";".join(acos.sentence_id.drop(sid_idx)), 1.0)
                     for sid_idx, sid in zip(acos.index, acos.sentence_id)}
    return remove(df, drop, "C3", split), context_dependent


# ----------------------------------------------------------------------------------------------- C4 leakage
def similarity(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For every train sentence: the best character 5-gram Jaccard against any test sentence, and that sentence's index.
    Texts are compared as match_key with space padding; texts too short for a 5-gram only match exactly."""
    k_tr = [f" {match_key(t) or t} " for t in train.text]
    k_te = [f" {match_key(t) or t} " for t in test.text]
    vec = CountVectorizer(analyzer="char", ngram_range=(5, 5), lowercase=False, binary=True, dtype=np.int32).fit(k_tr + k_te)
    a, b = vec.transform(k_tr), vec.transform(k_te)
    inter = (a @ b.T).tocoo()
    na, nb = np.asarray(a.sum(1)).ravel(), np.asarray(b.sum(1)).ravel()
    jac = coo_matrix((inter.data / (na[inter.row] + nb[inter.col] - inter.data), (inter.row, inter.col)), shape=inter.shape).tocsr()
    best = jac.max(axis=1).toarray().ravel()
    arg = np.where(best > 0, np.asarray(jac.argmax(axis=1)).ravel(), -1)
    exact = {k: j for j, k in enumerate(k_te)}
    for i, k in enumerate(k_tr):
        if k in exact:
            best[i], arg[i] = 1.0, exact[k]
    return best, arg


def remove_leakage(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    tr, te = to_sentences(train), to_sentences(test)
    best, arg = similarity(tr, te)
    table = []
    for src in ("semeval16", "acos"):
        for tsrc in ("semeval16", "acos"):
            m = (tr.source == src).to_numpy() & (arg >= 0) & (te.source.to_numpy()[np.maximum(arg, 0)] == tsrc)
            table.append([src, tsrc, *(int((m & (best >= th - 1e-9)).sum()) for th in LEAK_REPORTED)])
    leak = best >= LEAK_THRESHOLD - 1e-9
    ids = {sid: ("exact copy of a test sentence" if s >= 1 - 1e-9 else f"near-duplicate of a test sentence (Jaccard >= {LEAK_THRESHOLD})",
                 te.sentence_id.iloc[j], round(float(s), 3)) for sid, s, j in zip(tr.sentence_id[leak], best[leak], arg[leak])}
    return remove(train, ids, "C4", "train"), table


# ----------------------------------------------------------------------------------------------- C5 derived columns
def cv_fold(review_id: str, k: int = 5, salt: str = "absa-exp") -> int:
    """D3.1: fold = SHA-1(review_id + salt) mod k. Deterministic, so a review's sentences always share a fold."""
    return int(hashlib.sha1((review_id + salt).encode("utf-8")).hexdigest(), 16) % k


def derive(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    has = df.aspect_category != ""
    n_pol = df[has].groupby(["sentence_id", "aspect_category"]).polarity.transform("nunique")
    df["polarity_conflict"] = False
    df.loc[n_pol.index, "polarity_conflict"] = (n_pol > 1).to_numpy()
    log("C5", split, df, df.polarity_conflict, "flagged polarity_conflict (one category, two polarities; both rows kept)")
    n_ops = Counter(df.sentence_id[has])
    df["n_opinions"] = df.sentence_id.map(lambda sid: n_ops.get(sid, 0)).astype(int)
    df["has_opinion"] = df.n_opinions > 0
    is_cv = (df.source == "semeval16") & (df.split == "train")
    df["fold"] = [cv_fold(r) if cv else -1 for r, cv in zip(df.review_id, is_cv)]
    tags = {t: "|".join(nuances(t)) for t in df.text.unique()}
    df["nuances"] = df.text.map(tags)
    return df[COLUMNS]


# ----------------------------------------------------------------------------------------------- C6 integrity
def check(frames: dict[str, pd.DataFrame], merged_test: pd.DataFrame) -> list[tuple[str, bool]]:
    tr, te = frames["train"], frames["test"]
    checks = [("no sentence_id in both train and test", not set(tr.sentence_id) & set(te.sentence_id))]
    for split, df in frames.items():
        has = df.aspect_category != ""
        s = to_sentences(df)
        labelled = s.text.map(match_key) + "\t" + s.opinions.map(lambda o: str(sorted(o))) + s.out_of_scope.astype(str)
        checks += [
            (f"{split}: unique (sentence, category, polarity) rows", not df.duplicated(["sentence_id", "aspect_category", "polarity"]).any()),
            (f"{split}: categories inside the 22 x 9 inventory", df.entity[has].isin(ENTITIES).all() and df.attribute[has].isin(ATTRIBUTES).all()),
            (f"{split}: valid polarity on opinion rows, none on opinion-less rows", df.polarity[has].isin(POLARITIES).all() and (df.polarity[~has] == "").all()),
            (f"{split}: one row per opinion-less sentence", not (df[~has].groupby("sentence_id").size() > 1).any() and not (has & ~df.has_opinion).any()),
            (f"{split}: no sentence repeated with the same labels", not labelled.duplicated().any()),
            (f"{split}: no empty text", not (df.text == "").any()),
            (f"{split}: no formatting artefact left (src.nuance_checker)", not any(artefacts(t) for t in s.text)),
        ]
    best, _ = similarity(to_sentences(tr), to_sentences(te))
    checks.append((f"no train sentence within Jaccard {LEAK_THRESHOLD} of a test sentence", not (best >= LEAK_THRESHOLD - 1e-9).any()))
    cv = tr.source == "semeval16"
    checks.append(("folds: 0-4 on SemEval train, -1 elsewhere", tr.fold[cv].between(0, 4).all() and (tr.fold[~cv] == -1).all() and (te.fold == -1).all()))
    t1, t1_was = te[te.source == "semeval16"], merged_test[merged_test.source == "semeval16"]
    checks.append(("T1 = the official 808 sentences / 801 opinions", t1.sentence_id.nunique() == 808 == t1_was.sentence_id.nunique()
                   and int((t1.aspect_category != "").sum()) == 801))
    return [(name, bool(ok)) for name, ok in checks]


# ----------------------------------------------------------------------------------------------- report
def table(headers, rows) -> list[str]:
    """Plain-text table: numeric columns right-aligned, text left-aligned."""
    cells = [list(map(str, headers)), *([str(c) for c in r] for r in rows)]
    width = [max(map(len, col)) for col in zip(*cells)]
    right = [all(re.fullmatch(r"-?[\d.,]+%?", c) for c in col[1:]) for col in zip(*cells)]
    line = lambda r: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(r, width, right)).rstrip()  # noqa: E731
    return [line(cells[0]), line(["-" * w for w in width]), *map(line, cells[1:]), ""]


def summary(frames: dict[str, pd.DataFrame]) -> list[list]:
    rows = []
    for split, df in frames.items():
        for src in ("semeval16", "acos"):
            d = df[df.source == src]
            s = d.drop_duplicates("sentence_id")
            has = d.aspect_category != ""
            rows.append([split, src, len(s), int(s.out_of_scope.sum()), len(s) - d.sentence_id[has].nunique(), int(has.sum()),
                         d[has].drop_duplicates(["sentence_id", "aspect_category"]).shape[0], d.aspect_category[has].nunique()])
    return rows


def label_space(frames: dict[str, pd.DataFrame]) -> dict:
    counts: dict[str, Counter] = {}
    for split, df in frames.items():
        for src in ("semeval16", "acos"):
            d = df[(df.source == src) & (df.aspect_category != "")].drop_duplicates(["sentence_id", "aspect_category"])
            counts[f"{src}_{split}"] = Counter(d.aspect_category)
    s1, merged = counts["semeval16_train"], counts["semeval16_train"] + counts["acos_train"]
    cats = sorted(set().union(*counts.values()), key=lambda c: (-s1[c], -merged[c], c))
    return {
        "note": "Counts are sentences containing the category (unique sentence x category), from the *_merged_cleaned.csv files.",
        "entities": list(ENTITIES), "attributes": list(ATTRIBUTES), "polarities": list(POLARITIES),
        "n_possible_categories": len(ENTITIES) * len(ATTRIBUTES),
        "semeval16_train_categories": [c for c in cats if s1[c]],
        "merged_train_categories": [c for c in cats if merged[c]],
        "rare_in_semeval16_train_lt5": [c for c in cats if 0 < s1[c] < 5],
        "unseen_in_semeval16_train": {k: sorted(c for c in counts[k] if not s1[c]) for k in ("semeval16_test", "acos_test")},
        "unseen_in_merged_train": {k: sorted(c for c in counts[k] if not merged[c]) for k in ("semeval16_test", "acos_test")},
        "counts": {c: {k: counts[k][c] for k in counts} for c in cats},
    }


def main() -> None:
    t0 = time.time()
    merged = {s: load_long(OUT / f"{s}_merged.csv") for s in ("train", "test")}
    frames, context_dependent = {}, 0
    for split, df in merged.items():
        df = normalise_frame(df, split)
        df = resolve_out_of_scope(df, split)
        df, n = resolve_same_text(df, split)
        frames[split], context_dependent = df, context_dependent + n
    frames["train"], leak_table = remove_leakage(frames["train"], frames["test"])
    frames = {split: derive(df, split) for split, df in frames.items()}
    checks = check(frames, merged["test"])
    if not all(ok for _, ok in checks):
        sys.exit("C6 integrity check failed:\n  " + "\n  ".join(name for name, ok in checks if not ok))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_csv(OUT / f"{split}_merged_cleaned.csv", index=False)
    space = label_space(frames)
    (OUT / "label_space.json").write_text(json.dumps(space, indent=1), encoding="utf-8")
    pd.DataFrame(REMOVED).to_csv(REPORT_DIR / "removed_sentences.csv", index=False)

    tr = frames["train"][frames["train"].source == "semeval16"]
    folds = [[k, tr.review_id[tr.fold == k].nunique(), tr.sentence_id[tr.fold == k].nunique(), int(((tr.fold == k) & (tr.aspect_category != "")).sum())]
             for k in range(5)]
    head = ["split", "source", "sentences", "out of scope", "without opinion", "opinion rows", "sentence x category", "categories"]
    rep = ["STAGE 2: CLEANING (scripts/clean_data.py)", "", "1. Before (*_merged.csv)"]
    rep += table(head, summary(merged)) + ["   After (*_merged_cleaned.csv)"] + table(head, summary(frames))
    rep += ["2. Steps"]
    rep += table(["step", "split", "source", "sentences", "rows", "what"], STEPS)
    rep += [f"C3 SemEval sentences sharing a text but not labels (kept): {context_dependent}", "",
            "3. Train/test leakage (C4, D2.5): train sentences by best character 5-gram Jaccard to any test sentence"]
    rep += table(["train source", "best match in", "exact", ">= 0.9", ">= 0.8"], leak_table)
    rep += [f"removed: >= {LEAK_THRESHOLD}; every removed sentence: reports/removed_sentences.csv", "",
            "4. CV folds (D3.1, SemEval train)"]
    rep += table(["fold", "reviews", "sentences", "opinion rows"], folds)
    rep += ["5. Label space (label_space.json)",
            f"possible categories: {space['n_possible_categories']}",
            f"SemEval train categories: {len(space['semeval16_train_categories'])}",
            f"merged train categories (SemEval + ACOS): {len(space['merged_train_categories'])}",
            f"SemEval train categories with < 5 sentences: {len(space['rare_in_semeval16_train_lt5'])}",
            *(f"{k} categories unseen in SemEval train: {len(v)} {v}" for k, v in space["unseen_in_semeval16_train"].items()),
            *(f"{k} categories unseen in merged train: {len(v)} {v}" for k, v in space["unseen_in_merged_train"].items()), "",
            "6. Integrity checks (C6)"]
    rep += table(["check", "result"], [[name, "ok" if ok else "FAILED"] for name, ok in checks])
    (REPORT_DIR / "clean_data.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'clean_data.txt').relative_to(ROOT)}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
