"""Check the raw corpora in DATA/ for duplicated content, then clean them into DATA/final/*.csv.

Inputs
* DATA/HF      SemEval-2016 Task 5, English laptops (XML). SB1 = labels per sentence, SB2 = labels per review.
* DATA/GITHUB  Laptop-ACOS (TSV): a sentence, then one tab-separated quad per opinion:
               "aspect_span category sentiment opinion_span" (token spans, -1,-1 = implicit; sentiment 0/1/2).

Part 1 (check): is any file a byte-identical copy of another (md5)? Which rows are duplicates, i.e. have the same
text and the same aspect_category, within and across files? Text is compared as lowercase letters and digits only,
so "I love it!" matches ACOS's tokenised "i love it !". Also: which sentences do files share regardless of labels?

Part 2 (clean): one CSV per input file in DATA/final/, one row per unique (text, aspect_category, polarity).
A text with no opinion keeps one row with empty label columns: those rows are the negatives for category detection.
Duplicate rows are flagged in `duplicate_in`, not removed. Both parts are summarised in DATA/final/datacheck_report.md.

Run:  python scripts/datacheck.py
"""

from __future__ import annotations

import hashlib
import html
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from absa_lab.data import POLARITIES, canonical, norm_text  # noqa: E402

RAW, OUT = ROOT / "DATA", ROOT / "DATA" / "final"

# output name -> (path under DATA/, format, split). The names are reused in the report and in the `duplicate_in` column.
FILES = {
    "semeval16_train": ("HF/ABSA16_Laptops_Train_SB1_v2.xml", "sb1", "train"),
    "semeval16_test": ("HF/EN_LAPT_SB1_TEST_.xml.gold", "sb1", "test"),
    "semeval16_review_train": ("HF/ABSA16_Laptops_Train_English_SB2.xml", "sb2", "train"),
    "semeval16_review_test": ("HF/EN_LAPT_SB2_TEST.xml.gold", "sb2", "test"),
    "acos_train": ("GITHUB/laptop_quad_train.tsv", "acos", "train"),
    "acos_test": ("GITHUB/laptop_quad_test.tsv", "acos", "test"),
}
FOLDER = {n: rel.split("/")[0] for n, (rel, _, _) in FILES.items()}
ACOS_POLARITY = {"0": "negative", "1": "neutral", "2": "positive"}
COLUMNS = ["text", "aspect_category", "polarity", "source", "split", "review_id", "sentence_id", "out_of_scope", "polarity_conflict",
           "aspect_terms", "opinion_terms", "duplicate_in", "text_raw"]
LOG: Counter = Counter()  # (file, cleaning step) -> count


# ----------------------------------------------------------------------------------------------- reading
def read_semeval(path: Path, level: str) -> list[dict]:
    """One unit per sentence (SB1), or per review (SB2: its text is the review's sentences joined)."""
    units = []
    for r in ET.parse(path).getroot().iter("Review"):
        sents = [(s.get("id"), s.find("text").text or "", s.get("OutOfScope") == "TRUE",
                  [(o.get("category"), o.get("polarity"), "", "") for o in s.iter("Opinion")]) for s in r.iter("sentence")]
        if level == "sentence":
            units += [{"review_id": r.get("rid"), "sentence_id": sid, "text_raw": t, "sentences": [(sid, t)], "out_of_scope": oos, "opinions": ops}
                      for sid, t, oos, ops in sents]
        else:
            ops = r.find("Opinions")
            units.append({"review_id": r.get("rid"), "sentence_id": "", "text_raw": " ".join(t for _, t, _, _ in sents),
                          "sentences": [(sid, t) for sid, t, _, _ in sents], "out_of_scope": False,
                          "opinions": [(o.get("category"), o.get("polarity"), "", "") for o in (ops if ops is not None else [])]})
    return units


_QUAD = re.compile(r"(-?\d+),(-?\d+) (\S+#\S+) ([012]) (-?\d+),(-?\d+)")


def _span(toks, a, b):
    return "NULL" if a == "-1" else " ".join(toks[int(a) : int(b)])  # NULL = implicit, not written in the sentence


def read_acos(path: Path, split: str) -> list[dict]:
    units = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text, *quads = line.split("\t")
        toks, ops = text.split(" "), []
        for q in quads:
            m = _QUAD.fullmatch(q)
            if not m:
                raise ValueError(f"{path.name}:{i + 1}: malformed quad {q!r}")
            ops.append((m[3], m[4], _span(toks, m[1], m[2]), _span(toks, m[5], m[6])))
        sid = f"acos_{split}_{i}"  # ACOS has no ids; line number within the file
        units.append({"review_id": "", "sentence_id": sid, "text_raw": text, "sentences": [(sid, text)], "out_of_scope": False, "opinions": ops})
    return units


def read_all() -> dict[str, list[dict]]:
    return {n: read_acos(RAW / rel, split) if fmt == "acos" else read_semeval(RAW / rel, "review" if fmt == "sb2" else "sentence")
            for n, (rel, fmt, split) in FILES.items()}


# ----------------------------------------------------------------------------------------------- cleaning
def clean_text(t: str, name: str) -> str:
    u = html.unescape(t)  # the SemEval test XML is double-escaped, so "&quot;" survives XML parsing
    if u != t:
        LOG[name, "HTML entities unescaped (&quot; &amp; &gt;)"] += 1
    n = unicodedata.normalize("NFKC", u)
    if n != u:
        LOG[name, "Unicode NFKC (no-break space -> space)"] += 1
    w = re.sub(r"\s+", " ", n).strip()
    if w != n:
        LOG[name, "whitespace collapsed / trimmed"] += 1
    return w


_CONTRACTIONS = {"s", "t", "m", "ll", "re", "ve", "d"}
_ATTACH_LEFT = re.compile(r"[.,!?;:%)\]}]+")
_ATTACH_RIGHT = {"(", "[", "{", "$"}


def detokenise(t: str) -> str:
    """Undo ACOS's unambiguous tokenisation: "i ' m" -> "i'm", "fast ." -> "fast.", "$ 12" -> "$12", "` ` x ` `" -> '"x"'.
    Hyphens and slashes stay spaced ("laptop - fast" may be a dash). Lowercasing cannot be undone."""
    toks = t.replace("` `", '"').split(" ")
    out, glue, in_quote = [], False, False
    for i, tok in enumerate(toks):
        if tok == "'" and out and i + 1 < len(toks) and toks[i + 1] in _CONTRACTIONS:
            out[-1] += tok
            glue = True
            continue
        if out and (glue or _ATTACH_LEFT.fullmatch(tok) or (tok == '"' and in_quote)):
            out[-1] += tok
        else:
            out.append(tok)
        glue = tok in _ATTACH_RIGHT or (tok == '"' and not in_quote)
        in_quote ^= tok == '"'
    return " ".join(out)


def unit_rows(u: dict, name: str, fmt: str) -> list[dict]:
    """Rows for one text unit: unique (category, polarity), with ACOS aspect/opinion terms merged in aligned ' | ' lists."""
    merged: dict[tuple[str, str], list[tuple[str, str]]] = {}
    n_oos = 0
    for cat, pol, asp, opn in u["opinions"]:
        if cat.startswith("Out_Of_Scope"):
            LOG[name, "Out_Of_Scope#* opinion dropped (sentence flagged out_of_scope if nothing else is left)"] += 1
            n_oos += 1
            continue
        c = canonical(cat)
        if c is None:
            LOG[name, f"opinion with category outside the 22x9 scheme dropped ({cat})"] += 1
            continue
        if c != cat:
            LOG[name, f"category renamed {cat} -> {c}"] += 1
        if fmt == "acos":
            pols = [ACOS_POLARITY[pol]]
        elif pol == "conflict":
            pols = ["positive", "negative"]
            LOG[name, "'conflict' polarity split into a positive and a negative row (polarity_conflict=True)"] += 1
        elif pol in POLARITIES:
            pols = [pol]
        else:
            raise ValueError(f"{name}: unknown polarity {pol!r}")
        for p in pols:
            terms = merged.setdefault((c, p), [])
            if (asp, opn) in terms:
                LOG[name, "exact duplicate opinion removed"] += 1
                continue
            if terms:
                LOG[name, "opinion merged into the row with the same category and polarity (terms joined with ' | ')"] += 1
            terms.append((asp, opn))

    text = clean_text(u["text_raw"], name)
    if fmt == "acos":
        d = detokenise(text)
        if d != text:
            LOG[name, "tokenisation spacing undone (contractions, punctuation, quotes)"] += 1
        text = d
    base = {"text": text, "aspect_category": "", "polarity": "", "source": "acos" if fmt == "acos" else "semeval16",
            "split": FILES[name][2], "review_id": u["review_id"], "sentence_id": u["sentence_id"],
            "out_of_scope": u["out_of_scope"] or (n_oos > 0 and not merged), "polarity_conflict": False,
            "aspect_terms": "", "opinion_terms": "", "duplicate_in": "", "text_raw": u["text_raw"]}
    if not merged:
        return [base]
    conflict = {c for c, k in Counter(c for c, _ in merged).items() if k > 1}
    rows = []
    for (c, p), terms in merged.items():
        row = base | {"aspect_category": c, "polarity": p, "polarity_conflict": c in conflict}
        if fmt == "acos":
            row |= {"aspect_terms": " | ".join(detokenise(x) for x, _ in terms), "opinion_terms": " | ".join(detokenise(y) for _, y in terms)}
        rows.append(row)
    return rows


def clean(units: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    return {n: pd.DataFrame([r for u in units[n] for r in unit_rows(u, n, fmt)], columns=COLUMNS) for n, (_, fmt, _) in FILES.items()}


def row_key(text: str, category: str) -> tuple[str, str] | None:
    """Duplicate key: same text (lowercase letters and digits only, since ACOS is lowercased and tokenised) and same
    aspect_category. An empty category is a value too, so two no-opinion copies of a sentence are duplicates."""
    t = norm_text(text)
    return (t, category) if t else None


def flag_duplicates(frames: dict[str, pd.DataFrame]) -> dict:
    """Fill `duplicate_in`: files holding a row with the same key in a *different* text unit (sentence or review), own
    file included. Two rows of one sentence with opposite polarities are not duplicates; that is `polarity_conflict`.
    Returns key -> {(file, unit id): polarities}."""
    owners: dict = defaultdict(lambda: defaultdict(set))
    keyed = {}
    for name, df in frames.items():
        unit = df.sentence_id.where(df.sentence_id != "", df.review_id)
        keyed[name] = list(zip(map(row_key, df.text, df.aspect_category), unit))
        for (k, u), p in zip(keyed[name], df.polarity):
            if k:
                owners[k][name, u].add(p)
    for name, df in frames.items():
        df["duplicate_in"] = ["; ".join(n for n in FILES if k and any(f == n and v != u for f, v in owners[k])) for k, u in keyed[name]]
    return owners


# ----------------------------------------------------------------------------------------------- checking
def md(headers, rows) -> list[str]:
    cell = lambda x: str(x).replace("|", "\\|")  # noqa: E731
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers), *("| " + " | ".join(map(cell, r)) + " |" for r in rows), ""]


def check(units: dict[str, list[dict]], index: dict[str, Counter], frames: dict[str, pd.DataFrame], owners: dict) -> list[str]:
    names = list(FILES)
    rep = ["## 1. Files", ""]
    md5 = {n: hashlib.md5((RAW / rel).read_bytes()).hexdigest() for n, (rel, _, _) in FILES.items()}
    rows = []
    for n, (rel, fmt, _) in FILES.items():
        pol = Counter(p for u in units[n] for _, p, _, _ in u["opinions"])
        if fmt == "acos":
            pol = Counter({ACOS_POLARITY[k]: v for k, v in pol.items()})
        rows.append([n, f"`DATA/{rel}`", "TSV" if fmt == "acos" else "XML", "review" if fmt == "sb2" else "sentence", len(units[n]),
                     sum(len(u["opinions"]) for u in units[n]), ", ".join(f"{k} {v}" for k, v in sorted(pol.items())), md5[n][:12]])
    rep += md(["name", "path", "format", "labels per", "units", "opinions", "polarities", "md5"], rows)

    dup = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if md5[a] == md5[b]]
    rep += ["## 2. Byte-identical files", "", ("; ".join(f"`{a}` = `{b}`" for a, b in dup) if dup else
            "None. Every file has a different md5, so no file is a renamed copy of another."), ""]

    # ------------------------------------------------------------ duplicate rows: same text + same aspect_category
    dups = {k: us for k, us in owners.items() if len(us) > 1}
    per_file = {k: Counter(f for f, _ in us) for k, us in dups.items()}
    differ = sum(len({frozenset(p) for p in us.values()}) > 1 for us in dups.values())
    text_of = {}
    for df in frames.values():
        for t, c in zip(df.text, df.aspect_category):
            text_of.setdefault(row_key(t, c), t)

    def side(k, folder):
        by_file = defaultdict(set)
        for (f, _), p in owners[k].items():
            if FOLDER[f] == folder:
                by_file[f] |= p
        return by_file

    def fmt_side(by_file):
        return "; ".join(f"{f}: {'/'.join(sorted(p - {''})) or 'no opinion'}" for f, p in by_file.items())

    cross = sorted((k for k, c in per_file.items() if len({FOLDER[f] for f in c}) > 1), key=lambda k: (k[1], k[0]))
    agree = sum(set().union(*side(k, "HF").values()) == set().union(*side(k, "GITHUB").values()) for k in cross)
    rep += ["## 3. Duplicate rows: same text and same aspect_category", "",
            "Two rows are duplicates when they have the same text and the same `aspect_category` but come from different sentences "
            "(or reviews). Text is compared ignoring case, spacing and punctuation, because ACOS is lowercased and tokenised: an exact "
            "comparison would miss every ACOS match. Rows without an opinion count too (empty category = same category). Two rows of "
            "one sentence with opposite polarities are not duplicates; they are flagged `polarity_conflict`. Every duplicate row is "
            "flagged in the `duplicate_in` column of the output CSVs; nothing is removed.", "",
            f"**DATA/GITHUB vs DATA/HF: {len(cross)} duplicate (text, aspect_category) pairs; the polarity is the same on both sides for {agree}.**", ""]
    rep += md(["text", "aspect_category", "DATA/HF (file: polarity)", "DATA/GITHUB (file: polarity)"],
              [[text_of[k], k[1] or "(no opinion)", fmt_side(side(k, "HF")), fmt_side(side(k, "GITHUB"))] for k in cross])
    rep += ["**Every pair of files.** Cell = distinct (text, aspect_category) pairs that occur in both files. "
            "Diagonal = pairs that occur in more than one sentence (or review) of the same file.", ""]
    rep += md(["", *names], [[a, *[sum(1 for c in per_file.values() if (c[a] > 1 if a == b else c[a] and c[b])) for b in names]] for a in names])
    rep += [f"Overall, {len(dups)} (text, aspect_category) pairs occur more than once; {differ} of them carry different polarities in different copies.", ""]

    # ------------------------------------------------------------ same sentences regardless of labels
    rep += ["## 4. Same sentences, ignoring labels (why SB2 is not new data)", "",
            "Section 3 compares rows, and SB2 rows hold whole reviews, so it cannot see that SB2 reuses SB1's sentences. "
            "Here each file is reduced to its sentences (SB2 through the sentences of its reviews). "
            "Cell = distinct sentences in both files; diagonal = sentences repeated inside the file.", ""]
    rep += md(["", *names], [[a, *[sum(1 for v in index[a].values() if v > 1) if a == b else len(index[a].keys() & index[b].keys()) for b in names]] for a in names])
    for sp in ("train", "test"):
        s1, s2 = units[f"semeval16_{sp}"], units[f"semeval16_review_{sp}"]
        p1 = {(u["review_id"], *p) for u in s1 for p in u["sentences"]}
        p2 = {(u["review_id"], *p) for u in s2 for p in u["sentences"]}
        r1, r2 = {u["review_id"] for u in s1}, {u["review_id"] for u in s2}
        c1 = {}
        for u in s1:
            c1.setdefault(u["review_id"], set()).update(c for c, _, _, _ in u["opinions"])
        rel = Counter()
        for u in s2:
            c2 = {c for c, _, _, _ in u["opinions"]}
            ref = c1.get(u["review_id"], set())
            rel["identical" if c2 == ref else "SB2 adds categories" if c2 > ref else "SB2 drops categories" if c2 < ref else "different"] += 1
        rep += [f"**SemEval {sp}, SB1 vs SB2.** SB2 has {len(r2)} reviews, {len(r2 & r1)} of them in SB1 (SB1 has {len(r1)}). "
                f"{len(p2 & p1)} of {len(p2)} SB2 sentences are exact copies of SB1 sentences (same review id, sentence id and text). "
                f"Only the labels differ: SB2 labels the whole review. Review category set vs the union of SB1 sentence categories: "
                + ", ".join(f"{k} {v}" for k, v in rel.most_common()) + ".", ""]
    return rep


# ----------------------------------------------------------------------------------------------- main
def main():
    units = read_all()
    frames = clean(units)
    owners = flag_duplicates(frames)
    index = {n: Counter(k for u in units[n] for _, t in u["sentences"] if (k := norm_text(clean_text(t, "_")))) for n in FILES}
    check_rep = check(units, index, frames, owners)

    OUT.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for name, df in frames.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        key = df.sentence_id.where(df.sentence_id != "", df.review_id)
        units_with = key[df.aspect_category != ""].nunique()
        out_rows.append([f"`DATA/final/{name}.csv`", len(df), key.nunique(), units_with, key.nunique() - units_with,
                         key[df.out_of_scope].nunique(), int(df.polarity_conflict.sum()), int((df.duplicate_in != "").sum())])

    clean_rep = ["## 5. Cleaning applied", ""]
    clean_rep += md(["file", "step", "count"], [[f, s, c] for (f, s), c in sorted(LOG.items()) if f != "_"])  # "_" = lookup keys only
    clean_rep += ["Labels kept as given otherwise. Polarity is always negative / neutral / positive. "
                  "`aspect_terms` and `opinion_terms` (ACOS only) are aligned lists: item i of one pairs with item i of the other; NULL = implicit.", ""]
    clean_rep += ["## 6. Output", ""]
    clean_rep += md(["file", "rows", "text units", "with opinion", "without opinion", "out of scope", "polarity_conflict rows", "duplicate rows"], out_rows)
    clean_rep += ["Columns: " + ", ".join(f"`{c}`" for c in COLUMNS), ""]

    report = ["# Data check report", "", f"Generated by `scripts/datacheck.py` on {time.strftime('%Y-%m-%d %H:%M')}.", "", *check_rep, *clean_rep]
    (OUT / "datacheck_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
