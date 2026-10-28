"""Data preparation: parse SemEval-2016 Task 5 English laptops (DATA/HF), merge its two subtasks per split, clean, and
write DATA/final/train_merged.csv and DATA/final/test_merged.csv.

Inputs (plan D1, DATA/HF)
* SB1  labels per sentence: S1 train, T1 test (the task's labels).
* SB2  labels per review, on exactly the same sentences (S3); its polarity can also be "conflict".

Steps, in order (counted in the report; every removed sentence is listed in reports/removed_sentences.csv)
P1 Strict parsing (D2.1): every structural assumption is asserted, review / sentence / opinion counts must equal the
   published numbers, and every category must be in the 22 x 9 inventory, or the script stops. HTML entities are
   decoded (the test XML is double-escaped).
P2 Merge SB1 + SB2: SB2 must reuse SB1's sentences verbatim; every sentence row gets its review's SB2 labels in
   `review_opinions` ("CATEGORY:polarity|..."; empty = the review has no SB2 annotation, 55 train reviews).
P3 Normalise text and context with src.preprocess.normalise (D2.4, the function serving uses). src.nuance_checker
   counts the artefacts repaired; `elongated` = a letter run was capped ("soooo" -> "sooo").
P4 Exact duplicates (train): sentences with the same text (match_key) and the same labels are kept once. Same text with
   different labels is kept: SemEval labels depend on the review context.
P5 Train/test leakage (D2.5): character 5-gram Jaccard between every train and every test sentence, computed exactly.
   Counts are reported at exact / >= 0.9 / >= 0.8; train sentences at >= 0.8 are removed. Test sentences are never
   removed, so T1 stays the official test set.
P6 Derived columns: polarity_conflict (one category, two polarities in one sentence; both rows kept, D2.3),
   has_opinion, n_opinions, fold (D3.1, train only; -1 = test), nuances (D5.1 text slices from src.nuance_checker,
   "|"-joined).
P7 Integrity checks; the script stops if any fails.

Outputs: DATA/final/{train,test}_merged.csv (one row per opinion; a sentence without opinions has one row with an empty
         category), DATA/final/reports/prepare_data.txt (the printed report), reports/removed_sentences.csv
Run:     python scripts/prepare_data.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTRIBUTES, DATA_DIR, ENTITIES, FINAL_DIR, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.nuance_checker import ARTEFACTS, artefacts, nuances  # noqa: E402
from src.preprocess import match_key, normalise, unescape_html  # noqa: E402

RAW, OUT = DATA_DIR / "HF", FINAL_DIR
SB1 = {"train": "ABSA16_Laptops_Train_SB1_v2.xml", "test": "EN_LAPT_SB1_TEST_.xml.gold"}
SB2 = {"train": "ABSA16_Laptops_Train_English_SB2.xml", "test": "EN_LAPT_SB2_TEST.xml.gold"}
# Published sizes (Pontiki et al. 2016, Table 1): (reviews, sentences, opinions).
PUBLISHED = {("SB1", "train"): (450, 2500, 2909), ("SB1", "test"): (80, 808, 801),
             ("SB2", "train"): (395, None, 2082), ("SB2", "test"): (80, None, 545)}
LEAK_THRESHOLD, LEAK_REPORTED = 0.8, (1.0, 0.9, 0.8)
COLUMNS = ["sentence_id", "split", "fold", "review_id", "sentence_index", "review_n_sentences", "text", "aspect_category",
           "entity", "attribute", "polarity", "has_opinion", "n_opinions", "out_of_scope", "polarity_conflict", "elongated",
           "nuances", "prev_text", "next_text", "review_opinions", "text_raw"]
STEPS: list[list] = []  # [step, split, sentences, what]
REMOVED: list[dict] = []  # one record per removed sentence


def category(raw: str) -> str:
    """Anything outside the 22 x 9 inventory raises, so no label is ever silently dropped (unknown-category audit)."""
    entity, _, attribute = raw.partition("#")
    if entity in ENTITIES and attribute in ATTRIBUTES:
        return raw
    raise ValueError(f"category {raw!r} is not in the SemEval-2016 laptop inventory")


# ----------------------------------------------------------------------------------------------- P1 parsing
def read_sb1(split: str) -> list[dict]:
    """One record per sentence; opinions = [(category, polarity)]. prev/next_text = neighbouring sentences of the same
    review (context for plan L4)."""
    root = ET.parse(RAW / SB1[split]).getroot()
    assert root.tag == "Reviews", root.tag
    units = []
    for review in root.findall("Review"):
        rid, sents = review.get("rid"), review.findall("sentences/sentence")
        assert rid and sents, f"review without id or sentences in {SB1[split]}"
        texts = [unescape_html(s.findtext("text")) for s in sents]
        for k, s in enumerate(sents):
            assert s.get("id") == f"{rid}:{k}", f"sentence id {s.get('id')} is not {rid}:{k}"
            assert set(s.attrib) <= {"id", "OutOfScope"} and s.get("OutOfScope") in (None, "TRUE"), s.attrib
            assert s.findtext("text") is not None, s.get("id")
            opinions = []
            for o in s.findall("Opinions/Opinion"):
                assert set(o.attrib) == {"category", "polarity"} and o.get("polarity") in POLARITIES, o.attrib
                opinions.append((category(o.get("category")), o.get("polarity")))
            assert len(set(opinions)) == len(opinions), f"{s.get('id')}: the same opinion twice"
            units.append({"sentence_id": s.get("id"), "split": split, "review_id": rid, "sentence_index": k,
                          "review_n_sentences": len(sents), "text": texts[k], "text_raw": s.findtext("text"),
                          "out_of_scope": s.get("OutOfScope") == "TRUE", "prev_text": texts[k - 1] if k else "",
                          "next_text": texts[k + 1] if k + 1 < len(sents) else "", "opinions": opinions})
    if n := sum(u["text"] != u["text_raw"] for u in units):
        STEPS.append(["P1", split, n, "HTML entities decoded (the test XML is double-escaped: &amp;#34; -> \")"])
    return units


def read_sb2(split: str, sb1: list[dict]) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Review-level labels: review_id -> [(category, polarity)], polarity incl. 'conflict'. Asserts that SB2 reuses
    SB1's sentences verbatim. Returns the labels and the raw opinion count (for the published-count check)."""
    sb1_text = {u["sentence_id"]: u["text_raw"] for u in sb1}
    labels, n_raw = {}, 0
    for review in ET.parse(RAW / SB2[split]).getroot().findall("Review"):
        rid = review.get("rid")
        for s in review.findall("sentences/sentence"):
            assert sb1_text.get(s.get("id")) == s.findtext("text"), f"SB2 sentence {s.get('id')} differs from SB1"
        ops = []
        for o in review.findall("Opinions/Opinion"):
            assert o.get("polarity") in (*POLARITIES, "conflict"), o.attrib
            n_raw += 1
            key = (category(o.get("category")), o.get("polarity"))
            if key in ops:
                STEPS.append(["P2", split, 1, f"identical review-level opinion collapsed (review {rid}, {key[0]} {key[1]})"])
                continue
            ops.append(key)
        assert ops, f"SB2 review {rid} has no opinions"  # so an empty review_opinions always means "not in SB2"
        labels[rid] = ops
    return labels, n_raw


def check_published(name: str, split: str, got: tuple) -> list:
    want = PUBLISHED[name, split]
    if not all(w is None or w == g for w, g in zip(want, got)):
        sys.exit(f"D2.1 count check failed for {name} {split}: got {got}, published {want}")
    return [f"{name} {split}", *("" if w is None else g for w, g in zip(want, got)), "ok"]


# ----------------------------------------------------------------------------------------------- P3 normalisation
def normalise_units(units: list[dict], split: str) -> None:
    found = [artefacts(u["text"]) for u in units]
    for name in ARTEFACTS:
        if n := sum(name in f for f in found):
            STEPS.append(["P3", split, n, f"normalised: {name}"])
    for u, f in zip(units, found):
        u["elongated"] = "elongation" in f
        for col in ("text", "prev_text", "next_text"):
            u[col] = normalise(u[col])
        if not u["text"]:
            sys.exit(f"P3: {u['sentence_id']} has an empty text after normalisation")


def remove(units: list[dict], ids: dict[str, tuple[str, str, float]], step: str, split: str) -> list[dict]:
    """Drop whole sentences. ids: sentence_id -> (reason, matched sentence_id, similarity)."""
    for u in units:
        if u["sentence_id"] in ids:
            reason, other, score = ids[u["sentence_id"]]
            REMOVED.append({"step": step, "split": split, "sentence_id": u["sentence_id"], "reason": reason,
                            "matched_sentence_id": other, "similarity": score, "text": u["text"]})
    for reason, n in Counter(r for r, _, _ in ids.values()).items():
        STEPS.append([step, split, n, f"sentence removed: {reason}"])
    return [u for u in units if u["sentence_id"] not in ids]


# ----------------------------------------------------------------------------------------------- P4 duplicates
def labels_key(u: dict) -> tuple:
    return match_key(u["text"]) or u["text"], frozenset(u["opinions"]), u["out_of_scope"]  # ":(" compares verbatim


def drop_exact_duplicates(units: list[dict]) -> list[dict]:
    first: dict[tuple, str] = {}
    drop = {}
    for u in units:
        k = labels_key(u)
        if k in first:
            drop[u["sentence_id"]] = ("exact duplicate (same text, same labels)", first[k], 1.0)
        else:
            first[k] = u["sentence_id"]
    return remove(units, drop, "P4", "train")


# ----------------------------------------------------------------------------------------------- P5 leakage
def similarity(train: list[str], test: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For every train text: the best character 5-gram Jaccard against any test text, and that text's index. Texts are
    compared as match_key with space padding; texts too short for a 5-gram only match exactly."""
    k_tr = [f" {match_key(t) or t} " for t in train]
    k_te = [f" {match_key(t) or t} " for t in test]
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


def remove_leakage(train: list[dict], test: list[dict]) -> tuple[list[dict], list[int]]:
    best, arg = similarity([u["text"] for u in train], [u["text"] for u in test])
    counts = [int((best >= th - 1e-9).sum()) for th in LEAK_REPORTED]
    ids = {u["sentence_id"]: ("exact copy of a test sentence" if s >= 1 - 1e-9 else
                              f"near-duplicate of a test sentence (Jaccard >= {LEAK_THRESHOLD})", test[j]["sentence_id"], round(float(s), 3))
           for u, s, j in zip(train, best, arg) if s >= LEAK_THRESHOLD - 1e-9}
    return remove(train, ids, "P5", "train"), counts


# ----------------------------------------------------------------------------------------------- P6 derived columns
def cv_fold(review_id: str, k: int = 5, salt: str = "absa-exp") -> int:
    """D3.1: fold = SHA-1(review_id + salt) mod k. Deterministic, so a review's sentences always share a fold."""
    return int(hashlib.sha1((review_id + salt).encode("utf-8")).hexdigest(), 16) % k


def to_rows(units: list[dict], split: str, sb2: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for u in units:
        cats = Counter(c for c, _ in u["opinions"])
        base = {c: u.get(c) for c in COLUMNS} | {
            "fold": cv_fold(u["review_id"]) if split == "train" else -1, "has_opinion": bool(u["opinions"]),
            "n_opinions": len(u["opinions"]), "nuances": "|".join(nuances(u["text"])),
            "review_opinions": "|".join(f"{c}:{p}" for c, p in sb2.get(u["review_id"], [])),
            "aspect_category": "", "entity": "", "attribute": "", "polarity": "", "polarity_conflict": False}
        if not u["opinions"]:
            rows.append(base)
        for cat, pol in u["opinions"]:
            ent, _, att = cat.partition("#")
            rows.append(base | {"aspect_category": cat, "entity": ent, "attribute": att, "polarity": pol, "polarity_conflict": cats[cat] > 1})
    df = pd.DataFrame(rows, columns=COLUMNS)
    if n := df[df.polarity_conflict].sentence_id.nunique():
        STEPS.append(["P6", split, n, "flagged polarity_conflict (one category, two polarities; both rows kept)"])
    return df


# ----------------------------------------------------------------------------------------------- P7 integrity
def check(frames: dict[str, pd.DataFrame], sb2: dict[str, dict]) -> list[tuple[str, bool]]:
    tr, te = frames["train"], frames["test"]
    checks = [("no sentence_id in both train and test", not set(tr.sentence_id) & set(te.sentence_id))]
    for split, df in frames.items():
        has = df.aspect_category != ""
        s = df.drop_duplicates("sentence_id")
        checks += [
            (f"{split}: unique (sentence, category, polarity) rows", not df.duplicated(["sentence_id", "aspect_category", "polarity"]).any()),
            (f"{split}: categories inside the 22 x 9 inventory", df.entity[has].isin(ENTITIES).all() and df.attribute[has].isin(ATTRIBUTES).all()),
            (f"{split}: valid polarity on opinion rows, none on opinion-less rows", df.polarity[has].isin(POLARITIES).all() and (df.polarity[~has] == "").all()),
            (f"{split}: one row per opinion-less sentence", not (df[~has].groupby("sentence_id").size() > 1).any() and not (has & ~df.has_opinion).any()),
            (f"{split}: no empty text", not (df.text == "").any()),
            (f"{split}: no formatting artefact left (src.nuance_checker)", not any(artefacts(t) for t in s.text)),
            (f"{split}: review_opinions = SB2 labels of the review", (s.review_opinions != "").sum() == s.review_id.isin(sb2[split]).sum()),
        ]
    s = tr.drop_duplicates("sentence_id")
    ops = tr.groupby("sentence_id", sort=False).apply(lambda g: frozenset(zip(g.aspect_category, g.polarity)), include_groups=False)
    keys = [(match_key(t) or t, ops[sid], oos) for sid, t, oos in zip(s.sentence_id, s.text, s.out_of_scope)]
    checks.append(("train: no sentence repeated with the same labels", len(set(keys)) == len(keys)))
    best, _ = similarity(s.text.tolist(), te.drop_duplicates("sentence_id").text.tolist())
    checks.append((f"no train sentence within Jaccard {LEAK_THRESHOLD} of a test sentence", not (best >= LEAK_THRESHOLD - 1e-9).any()))
    checks.append(("folds: 0-4 on train, -1 on test", tr.fold.between(0, 4).all() and (te.fold == -1).all()))
    checks.append(("T1 = the official 808 sentences / 801 opinions", te.sentence_id.nunique() == 808 and int((te.aspect_category != "").sum()) == 801))
    return [(name, bool(ok)) for name, ok in checks]


# ----------------------------------------------------------------------------------------------- report
def table(headers, rows) -> list[str]:
    """Plain-text table: numeric columns right-aligned, text left-aligned."""
    cells = [list(map(str, headers)), *([str(c) for c in r] for r in rows)]
    width = [max(map(len, col)) for col in zip(*cells)]
    right = [all(re.fullmatch(r"-?[\d.,]+%?", c) for c in col[1:]) for col in zip(*cells)]
    line = lambda r: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(r, width, right)).rstrip()  # noqa: E731
    return [line(cells[0]), line(["-" * w for w in width]), *map(line, cells[1:]), ""]


def summary(frames: dict[str, pd.DataFrame], sb2: dict[str, dict]) -> list[list]:
    rows = []
    for split, df in frames.items():
        s = df.drop_duplicates("sentence_id")
        has = df.aspect_category != ""
        rev = s.review_id.unique()
        rows.append([split, len(rev), len(s), int(s.out_of_scope.sum()), int((~s.has_opinion).sum()), int(has.sum()),
                     df[has].drop_duplicates(["sentence_id", "aspect_category"]).shape[0], df.aspect_category[has].nunique(),
                     sum(r in sb2[split] for r in rev), sum(len(sb2[split].get(r, [])) for r in rev)])
    return rows


def main() -> None:
    t0 = time.time()
    sb1 = {s: read_sb1(s) for s in SB1}
    sb2, counts = {}, []
    for split in SB1:
        sb2[split], n_sb2 = read_sb2(split, sb1[split])
        counts.append(check_published("SB1", split, (len({u["review_id"] for u in sb1[split]}), len(sb1[split]),
                                                      sum(len(u["opinions"]) for u in sb1[split]))))
        counts.append(check_published("SB2", split, (len(sb2[split]), None, n_sb2)))
    sb2_pol = {s: Counter(p for ops in sb2[s].values() for _, p in ops) for s in SB2}

    for split, units in sb1.items():
        normalise_units(units, split)
    sb1["train"] = drop_exact_duplicates(sb1["train"])
    sb1["train"], leak = remove_leakage(sb1["train"], sb1["test"])
    frames = {split: to_rows(units, split, sb2[split]) for split, units in sb1.items()}
    checks = check(frames, sb2)
    if not all(ok for _, ok in checks):
        sys.exit("P7 integrity check failed:\n  " + "\n  ".join(name for name, ok in checks if not ok))

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_csv(OUT / f"{split}_merged.csv", index=False)
    pd.DataFrame(REMOVED, columns=["step", "split", "sentence_id", "reason", "matched_sentence_id", "similarity", "text"]).to_csv(
        REPORT_DIR / "removed_sentences.csv", index=False)

    tr, te = frames["train"], frames["test"]
    folds = [[k, tr.review_id[tr.fold == k].nunique(), tr.sentence_id[tr.fold == k].nunique(), int(((tr.fold == k) & (tr.aspect_category != "")).sum())]
             for k in range(5)]
    cat_tr = Counter(tr[tr.aspect_category != ""].drop_duplicates(["sentence_id", "aspect_category"]).aspect_category)
    unseen = sorted(set(te.aspect_category[te.aspect_category != ""]) - set(cat_tr))
    md5 = {f: hashlib.md5((RAW / f).read_bytes()).hexdigest()[:12] for f in [*SB1.values(), *SB2.values()]}

    rep = ["DATA PREPARATION: SemEval-2016 laptops SB1 + SB2 (scripts/prepare_data.py)", "", "1. Raw files"]
    rep += table(["file", "md5"], [[f"DATA/HF/{k}", v] for k, v in md5.items()])
    rep += ["2. Counts vs the published numbers (D2.1)"]
    rep += table(["subtask", "reviews", "sentences", "opinions", "check"], counts)
    rep += ["3. Steps"]
    rep += table(["step", "split", "sentences", "what"], STEPS)
    rep += ["4. Train/test leakage (P5, D2.5): train sentences by best character 5-gram Jaccard to any test sentence"]
    rep += table(["exact", ">= 0.9", ">= 0.8"], [leak])
    rep += [f"removed: >= {LEAK_THRESHOLD}; every removed sentence: reports/removed_sentences.csv", "", "5. Output (*_merged.csv)"]
    rep += table(["split", "reviews", "sentences", "out of scope", "without opinion", "opinion rows", "sentence x category",
                  "categories", "reviews with SB2 labels", "SB2 opinions"], summary(frames, sb2))
    rep += ["SB2 review-level polarity: " + "; ".join(f"{s} " + ", ".join(f"{p} {c[p]}" for p in (*POLARITIES, "conflict")) for s, c in sb2_pol.items()), "",
            "6. CV folds (D3.1, train)"]
    rep += table(["fold", "reviews", "sentences", "opinion rows"], folds)
    rep += ["7. Label space",
            f"possible categories: {len(ENTITIES) * len(ATTRIBUTES)}; in train: {len(cat_tr)}; train categories with < 5 sentences: "
            f"{sum(v < 5 for v in cat_tr.values())}",
            f"test categories unseen in train: {len(unseen)} {unseen}", "", "8. Integrity checks (P7)"]
    rep += table(["check", "result"], [[name, "ok" if ok else "FAILED"] for name, ok in checks])
    (REPORT_DIR / "prepare_data.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'prepare_data.txt').relative_to(ROOT)}, {(OUT / 'train_merged.csv').relative_to(ROOT)}, "
          f"{(OUT / 'test_merged.csv').relative_to(ROOT)}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
