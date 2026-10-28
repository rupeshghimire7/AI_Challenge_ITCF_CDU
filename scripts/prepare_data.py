"""Stage 1 of data preparation: parse the raw corpora, harmonise them into one schema, and merge them into
DATA/final/train_merged.csv and DATA/final/test_merged.csv with exact duplicates excluded.

Inputs (plan D1)
* DATA/HF      SemEval-2016 Task 5, English laptops. SB1 = labels per sentence (S1 train, T1 test); SB2 = labels per
               review, on exactly the same sentences (S3, kept separately for the review-level metric).
* DATA/GITHUB  Laptop-ACOS (S4 train, T2 test), TSV: a lowercased, tokenised sentence, then one tab-separated quad per
               opinion "aspect_span category sentiment opinion_span" (token spans; -1,-1 = implicit; sentiment 0/1/2).

What this stage does
1. Strict parsing (D2.1): every structural assumption is asserted, and sentence / review / opinion counts must equal
   the published numbers, or the script stops.
2. Harmonisation: one row per (sentence, category, polarity), the same columns for both sources. Categories are
   canonicalised (D2.2: upper case, FANS&COOLING -> FANS_COOLING); ACOS polarity 0/1/2 -> negative/neutral/positive;
   ACOS text is de-tokenised ("it ' s" -> "it's"); HTML entities are decoded. ACOS's Out_Of_Scope#* opinions are kept
   here and resolved in stage 2.
3. Structural repairs: identical opinions inside a sentence are collapsed; ACOS opinions with the same category and
   polarity become one row whose aspect / opinion terms are aligned " | " lists; an ACOS sentence whose annotation was
   split over two consecutive lines is merged back into one sentence.
4. Exact duplicates excluded: within each split, sentences with the same text (compared with match_key) and the same
   labels are kept once (SemEval before ACOS, then file order); `removed_copies` lists the ids that were dropped.
   Same text with different labels is not a duplicate: stage 2 (clean_data.py) resolves it, together with
   train/test leakage.

Outputs: DATA/final/{train,test}_merged.csv, DATA/final/sb2_reviews_{train,test}.csv,
         DATA/final/reports/prepare_data.txt (the printed report)
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTRIBUTES, DATA_DIR, ENTITIES, FINAL_DIR, OUT_OF_SCOPE, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.preprocess import match_key, unescape_html  # noqa: E402

RAW, OUT = DATA_DIR, FINAL_DIR

SB1 = {"train": "HF/ABSA16_Laptops_Train_SB1_v2.xml", "test": "HF/EN_LAPT_SB1_TEST_.xml.gold"}
SB2 = {"train": "HF/ABSA16_Laptops_Train_English_SB2.xml", "test": "HF/EN_LAPT_SB2_TEST.xml.gold"}
ACOS = {"train": "GITHUB/laptop_quad_train.tsv", "test": "GITHUB/laptop_quad_test.tsv"}
# Published sizes (Pontiki et al. 2016, Table 1; Cai et al. 2021, Table 2): (reviews, sentences, opinions).
PUBLISHED = {("semeval16", "train"): (450, 2500, 2909), ("semeval16", "test"): (80, 808, 801),
             ("sb2", "train"): (395, None, 2082), ("sb2", "test"): (80, None, 545),
             ("acos", "train"): (None, 2934, 4172), ("acos", "test"): (None, 816, 1161)}
ACOS_POLARITY = {"0": "negative", "1": "neutral", "2": "positive"}
COLUMNS = ["sentence_id", "source", "split", "review_id", "sentence_index", "review_n_sentences", "text", "aspect_category",
           "entity", "attribute", "polarity", "out_of_scope", "aspect_terms", "opinion_terms", "prev_text", "next_text",
           "removed_copies", "text_raw"]
ENTITY_ALIASES = {"FANS&COOLING": "FANS_COOLING", "HARD_DISK": "HARD_DISC"}
LOG: Counter = Counter()  # (source/split, step) -> count


def canonical_category(raw: str) -> str:
    """D2.2: 'fans&cooling#general' -> 'FANS_COOLING#GENERAL'; 'Out_Of_Scope#X' -> 'OUT_OF_SCOPE#X'. Anything outside the
    inventory raises, so no label is ever silently dropped (unknown-category audit)."""
    entity, sep, attribute = raw.strip().upper().partition("#")
    entity = ENTITY_ALIASES.get(entity, entity)
    if sep and attribute in ATTRIBUTES and (entity in ENTITIES or entity == OUT_OF_SCOPE):
        return f"{entity}#{attribute}"
    raise ValueError(f"category {raw!r} is not in the SemEval-2016 laptop inventory")


def split_category(category: str) -> tuple[str, str]:
    entity, _, attribute = category.partition("#")
    return entity, attribute


# ----------------------------------------------------------------------------------------------- SemEval (XML)
def read_sb1(split: str) -> list[dict]:
    """One unit per sentence. prev/next_text = neighbouring sentences of the same review (context for plan L4)."""
    root = ET.parse(RAW / SB1[split]).getroot()
    assert root.tag == "Reviews", root.tag
    units = []
    for review in root.findall("Review"):
        rid, sents = review.get("rid"), review.findall("sentences/sentence")
        assert rid and sents, f"review without id or sentences in {SB1[split]}"
        texts = [unescape_html(s.findtext("text")) for s in sents]
        if n_decoded := sum(t != s.findtext("text") for t, s in zip(texts, sents)):
            LOG[f"semeval16/{split}", "HTML entities decoded (the test XML is double-escaped: &amp;#34; -> \")"] += n_decoded
        for k, s in enumerate(sents):
            assert s.get("id") == f"{rid}:{k}", f"sentence id {s.get('id')} is not {rid}:{k}"
            assert set(s.attrib) <= {"id", "OutOfScope"} and s.get("OutOfScope") in (None, "TRUE"), s.attrib
            assert s.findtext("text") is not None, s.get("id")
            opinions = []
            for o in s.findall("Opinions/Opinion"):
                assert set(o.attrib) == {"category", "polarity"} and o.get("polarity") in POLARITIES, o.attrib
                opinions.append((canonical_category(o.get("category")), o.get("polarity"), "", ""))
            units.append({"sentence_id": f"semeval16/{split}/{s.get('id')}", "source": "semeval16", "split": split, "review_id": rid,
                          "sentence_index": k, "review_n_sentences": len(sents), "text": texts[k], "text_raw": s.findtext("text"),
                          "out_of_scope": s.get("OutOfScope") == "TRUE", "prev_text": texts[k - 1] if k else "",
                          "next_text": texts[k + 1] if k + 1 < len(sents) else "", "opinions": opinions})
    return units


def read_sb2(split: str, sb1: list[dict]) -> pd.DataFrame:
    """Review-level labels (one row per review x category x polarity; 'conflict' is SB2's fourth polarity). Asserts that
    SB2 reuses SB1's sentences verbatim, which is why it is not merged as extra data."""
    sb1_text = {(u["review_id"], u["sentence_id"].rsplit("/", 1)[1]): u["text_raw"] for u in sb1}
    rows = []
    for review in ET.parse(RAW / SB2[split]).getroot().findall("Review"):
        rid, sents = review.get("rid"), review.findall("sentences/sentence")
        for s in sents:
            assert sb1_text.get((rid, s.get("id"))) == s.findtext("text"), f"SB2 sentence {s.get('id')} differs from SB1"
        seen = set()
        for o in review.findall("Opinions/Opinion"):
            assert o.get("polarity") in (*POLARITIES, "conflict"), o.attrib
            key = (canonical_category(o.get("category")), o.get("polarity"))
            if key in seen:
                LOG[f"sb2/{split}", "identical review-level opinion collapsed"] += 1
                continue
            seen.add(key)
            rows.append({"review_id": rid, "review_n_sentences": len(sents), "aspect_category": key[0], "entity": split_category(key[0])[0],
                         "attribute": split_category(key[0])[1], "polarity": key[1],
                         "review_text": " ".join(unescape_html(s.findtext("text")) for s in sents)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- ACOS (TSV)
_QUAD = re.compile(r"(-?\d+),(-?\d+) (\S+#\S+) ([012]) (-?\d+),(-?\d+)")
_CONTRACTIONS = {"s", "t", "m", "ll", "re", "ve", "d"}
_ATTACH_LEFT = re.compile(r"[.,!?;:%)\]}]+")
_ATTACH_RIGHT = {"(", "[", "{", "$"}
_CURLY = {"“": ('"', True), "”": ('"', False), "‘": ("'", True), "’": ("'", False)}  # quote char, opens
_STARS = re.compile(r"\*+")
_WORD = re.compile(r"[a-z0-9]+")
# Hyphen joining is ambiguous after tokenisation ("usb - c" is a word, "terrible - i use it" is a dash), so a pair is
# joined only on evidence: the compound is written hyphenated in SemEval, a side holds a digit (model numbers,
# ranges), a known prefix / suffix, or the same pair recurs in ACOS. Never next to a function word.
_PREFIXES = {"re", "non", "anti", "pre", "multi", "ultra", "mid", "semi", "self", "e", "co", "bi", "dual", "quad", "over",
             "under", "sub", "post", "inter", "x"}
_SUFFIXES = {"like", "based", "free", "friendly", "made", "end", "level", "related", "oriented", "sized", "style", "type",
             "in", "up", "out", "fi", "lit", "core"}
_FUNCTION = {"i", "it", "its", "the", "a", "an", "this", "that", "these", "those", "and", "but", "or", "however", "there",
             "you", "we", "they", "he", "she", "my", "your", "our", "their", "will", "is", "was", "had", "has", "have", "too",
             "once", "just", "such", "not", "no", "which", "what", "when", "if", "then", "also", "very", "as", "to", "for",
             "of", "with", "at", "by", "from", "on"}


def _join_hyphen(left: str, right: str, compounds: set[str], pairs: Counter) -> bool:
    if f"{left}-{right}" in compounds:
        return True
    if left in _FUNCTION or right in _FUNCTION:
        return False
    return (any(c.isdigit() for c in left + right) or left in _PREFIXES or right in _SUFFIXES or pairs[left, right] >= 2)


def detokenise(text: str, compounds: set[str], pairs: Counter) -> str:
    """Undo ACOS's tokenisation where it is unambiguous: contractions ("i ' m" -> "i'm"), punctuation spacing
    ("fast ." -> "fast."), currency / percent ("$ 12" -> "$12"), quotes ("` ` x ` `" -> '"x"', "“ x ”" -> '"x"',
    "' x '" -> "'x'"), *emphasis*, split dashes and asterisks ("- -" -> "--", "* * *" -> "***"), slashes and digit
    colons ("i / o" -> "i/o", "16 : 9" -> "16:9"), and evidence-backed hyphen compounds. Lowercasing cannot be undone
    and is left as is."""
    text = re.sub(r"(?<=[a-z]) n ' t(?= |$)", "n't", text.replace("` `", '"'))  # PTB-style "is n ' t" -> "isn't"
    text = re.sub(r"(?<!\S)([-*/])(?: \1)+(?!\S)", lambda m: m[0].replace(" ", ""), text)
    toks = text.split(" ")
    joined: list[str] = []  # pass 1: infix joins, so "usb - c" is one token before the spacing rules run
    for i, tok in enumerate(toks):
        if joined and joined[-1] in "-/:" and len(joined) >= 2 and _WORD.fullmatch(tok):
            left = re.split(r"[-/:]", joined[-2])[-1]
            infix = joined[-1]
            if _WORD.fullmatch(left) and (infix == "/" or (infix == ":" and left.isdigit() and tok.isdigit())
                                          or (infix == "-" and _join_hyphen(left, tok, compounds, pairs))):
                joined[-2:] = [f"{joined[-2]}{infix}{tok}"]
                continue
        joined.append(tok)
    def contraction(i: int) -> bool:  # "don ' t", "acer ' s", "manufacturer ' s/model's"
        return 0 < i < len(joined) - 1 and joined[i + 1].split("/")[0] in _CONTRACTIONS

    n_squote = sum(1 for i, t in enumerate(joined) if t == "'" and not contraction(i))
    n_star = sum(1 for t in joined if _STARS.fullmatch(t))
    out: list[str] = []
    glue, in_dq, in_sq, in_star = False, False, False, False
    for i, tok in enumerate(joined):
        if tok == "'" and out and contraction(i):
            out[-1] += tok
            glue = True
            continue
        if tok in _CURLY:  # curly quotes face one way: "“ new ”" -> '"new"', "15 ”" -> '15"'
            tok, opens = _CURLY[tok]
            quote, closing = True, not opens
        elif _STARS.fullmatch(tok):  # "* love *" -> "*love*" when asterisks pair up; a lone one is a bullet or footnote
            quote = n_star % 2 == 0
            closing = quote and in_star
        else:  # single quotes only when they pair up, or close a curly "‘"
            quote = tok == '"' or (tok == "'" and (n_squote % 2 == 0 or in_sq))
            closing = quote and (in_dq if tok == '"' else in_sq)
        if out and (glue or _ATTACH_LEFT.fullmatch(tok) or closing):
            out[-1] += tok
        else:
            out.append(tok)
        glue = tok in _ATTACH_RIGHT or (quote and not closing)
        if quote:
            if tok == '"':
                in_dq = not closing
            elif tok == "'":
                in_sq = not closing
            else:
                in_star = not closing
    return " ".join(out)


def read_acos(split: str, compounds: set[str], pairs: Counter) -> list[dict]:
    units = []
    for i, line in enumerate((RAW / ACOS[split]).read_text(encoding="utf-8").splitlines()):
        text, *quads = line.split("\t")
        toks = text.split(" ")
        assert text.strip() and quads, f"{ACOS[split]}:{i + 1}: empty sentence or no quad"
        opinions = []
        for q in quads:
            m = _QUAD.fullmatch(q)
            assert m, f"{ACOS[split]}:{i + 1}: malformed quad {q!r}"
            spans = [(int(m[1]), int(m[2])), (int(m[5]), int(m[6]))]
            assert all(a == b == -1 or 0 <= a < b <= len(toks) for a, b in spans), f"{ACOS[split]}:{i + 1}: span out of range"
            terms = ["NULL" if a == -1 else detokenise(" ".join(toks[a:b]), compounds, pairs) for a, b in spans]  # NULL = implicit
            opinions.append((canonical_category(m[3]), ACOS_POLARITY[m[4]], *terms))
            if canonical_category(m[3]) != m[3]:
                LOG[f"acos/{split}", f"category renamed {m[3]} -> {canonical_category(m[3])}"] += 1
        units.append({"sentence_id": f"acos/{split}/{i}", "source": "acos", "split": split, "review_id": "", "sentence_index": i,
                      "review_n_sentences": None, "text": detokenise(text, compounds, pairs), "text_raw": text, "out_of_scope": False,
                      "prev_text": "", "next_text": "", "opinions": opinions})
        if units[-1]["text"] != text:
            LOG[f"acos/{split}", "tokenisation spacing undone"] += 1
    return units


def hyphen_evidence(sb1_units: list[dict]) -> tuple[set[str], Counter]:
    """Hyphenated words as SemEval writes them (incl. the sub-pairs of chains like 2-in-1), and ACOS's spaced pairs."""
    compounds = set()
    for u in sb1_units:
        for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", u["text"].lower()):
            parts = w.split("-")
            compounds |= {w} | {f"{a}-{b}" for a, b in zip(parts, parts[1:])}
    pairs = Counter()
    for split in ACOS:
        for line in (RAW / ACOS[split]).read_text(encoding="utf-8").splitlines():
            pairs.update(re.findall(r"(?<!\S)([a-z0-9]+) - (?=([a-z0-9]+)(?!\S))", line.split("\t")[0]))
    return compounds, pairs


# ----------------------------------------------------------------------------------------------- harmonise and merge
def collapse_opinions(unit: dict) -> list[tuple[str, str, str, str]]:
    """Unique (category, polarity) per sentence; ACOS terms of merged opinions are kept as aligned ' | ' lists."""
    merged: dict[tuple[str, str], list[tuple[str, str]]] = {}
    tag = f"{unit['source']}/{unit['split']}"
    for cat, pol, asp, opn in unit["opinions"]:
        terms = merged.setdefault((cat, pol), [])
        if (asp, opn) in terms:
            LOG[tag, "identical opinion inside a sentence collapsed"] += 1
            continue
        if terms:
            LOG[tag, "opinion merged into the row with the same category and polarity (terms joined with ' | ')"] += 1
        terms.append((asp, opn))
    return [(c, p, " | ".join(a for a, _ in t) if unit["source"] == "acos" else "", " | ".join(o for _, o in t) if unit["source"] == "acos" else "")
            for (c, p), t in merged.items()]


def merge_split_lines(units: list[dict]) -> list[dict]:
    """ACOS occasionally annotates one sentence over two consecutive lines (same text, disjoint opinions): merge them."""
    out: list[dict] = []
    for u in units:
        prev = out[-1] if out else None
        if prev and u["source"] == prev["source"] == "acos" and match_key(u["text"]) == match_key(prev["text"]):
            prev["opinions"] += u["opinions"]
            LOG[f"acos/{u['split']}", f"sentence annotated over two lines merged ({prev['sentence_id']} + {u['sentence_id']})"] += 1
            continue
        out.append(u)
    return out


def drop_exact_duplicates(units: list[dict]) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Within one split: same match_key and the same labels (opinion set and out-of-scope flag) -> keep the first unit.
    `units` must be in priority order (SemEval first). Returns kept units and (removed id, kept id, text) records."""
    first: dict[tuple, dict] = {}
    kept, removed = [], []
    for u in units:
        k = (match_key(u["text"]), frozenset((c, p) for c, p, _, _ in u["opinions"]), u["out_of_scope"])
        if k in first:
            keeper = first[k]
            keeper["removed_copies"].append(u["sentence_id"])
            removed.append((u["sentence_id"], keeper["sentence_id"], u["text"]))
            LOG[f"{u['source']}/{u['split']}", f"exact duplicate of a {keeper['source']} {keeper['split']} sentence removed"] += 1
            continue
        u["removed_copies"] = []
        first[k] = u
        kept.append(u)
    return kept, removed


def to_rows(units: list[dict]) -> pd.DataFrame:
    rows = []
    for u in units:
        base = {c: u.get(c, "") for c in COLUMNS} | {"removed_copies": ";".join(u["removed_copies"]), "aspect_category": "",
                                                     "entity": "", "attribute": "", "polarity": "", "aspect_terms": "", "opinion_terms": ""}
        if not u["opinions"]:
            rows.append(base)
        for cat, pol, asp, opn in u["opinions"]:
            ent, att = split_category(cat)
            rows.append(base | {"aspect_category": cat, "entity": ent, "attribute": att, "polarity": pol, "aspect_terms": asp, "opinion_terms": opn})
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["review_n_sentences"] = df["review_n_sentences"].astype("Int64")
    return df


# ----------------------------------------------------------------------------------------------- report
def table(headers, rows) -> list[str]:
    """Plain-text table: numeric columns right-aligned, text left-aligned."""
    cells = [list(map(str, headers)), *([str(c) for c in r] for r in rows)]
    width = [max(map(len, col)) for col in zip(*cells)]
    right = [all(re.fullmatch(r"-?[\d.,]+%?", c) for c in col[1:]) for col in zip(*cells)]
    line = lambda r: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(r, width, right)).rstrip()  # noqa: E731
    return [line(cells[0]), line(["-" * w for w in width]), *map(line, cells[1:]), ""]


def check_published(name: str, split: str, reviews: int | None, sentences: int | None, opinions: int) -> list:
    want = PUBLISHED[name, split]
    got = (reviews, sentences, opinions)
    ok = all(w is None or w == g for w, g in zip(want, got))
    if not ok:
        sys.exit(f"D2.1 count check failed for {name} {split}: got {got}, published {want}")
    return [f"{name} {split}", *("" if g is None else g for g in got), "ok"]


def main() -> None:
    t0 = time.time()
    sb1 = {s: read_sb1(s) for s in SB1}
    compounds, pairs = hyphen_evidence(sb1["train"] + sb1["test"])
    acos = {s: read_acos(s, compounds, pairs) for s in ACOS}
    sb2 = {s: read_sb2(s, sb1[s]) for s in SB2}

    counts = []
    for split in ("train", "test"):
        counts.append(check_published("semeval16", split, len({u["review_id"] for u in sb1[split]}), len(sb1[split]),
                                      sum(len(u["opinions"]) for u in sb1[split])))
        counts.append(check_published("acos", split, None, len(acos[split]), sum(len(u["opinions"]) for u in acos[split])))
        n_sb2 = len(ET.parse(RAW / SB2[split]).getroot().findall("Review/Opinions/Opinion"))
        counts.append(check_published("sb2", split, sb2[split].review_id.nunique(), None, n_sb2))

    frames, removed_all, summary = {}, [], []
    for split in ("train", "test"):
        units = sb1[split] + merge_split_lines(acos[split])  # priority order: SemEval first
        for u in units:
            u["opinions"] = collapse_opinions(u)
        kept, removed = drop_exact_duplicates(units)
        removed_all += [(split, *r) for r in removed]
        frames[split] = to_rows(kept)
        for src in ("semeval16", "acos"):
            n_in = sum(u["source"] == src for u in units)
            n_out = sum(u["source"] == src for u in kept)
            summary.append([split, src, n_in, n_in - n_out, n_out, int(((frames[split].source == src) & (frames[split].aspect_category != "")).sum())])

    OUT.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_csv(OUT / f"{split}_merged.csv", index=False)
        sb2[split].to_csv(OUT / f"sb2_reviews_{split}.csv", index=False)

    # Informational only: identical texts across train and test. Stage 2 removes them (with near-duplicates) as leakage.
    test_keys = {match_key(t) for t in frames["test"].text}
    overlap = frames["train"].drop_duplicates("sentence_id")
    overlap = overlap[overlap.text.map(match_key).isin(test_keys)]
    md5 = {rel: hashlib.md5((RAW / rel).read_bytes()).hexdigest()[:12] for rel in [*SB1.values(), *SB2.values(), *ACOS.values()]}

    rep = ["STAGE 1: PARSE, HARMONISE, MERGE (scripts/prepare_data.py)", "", "1. Raw files"]
    rep += table(["file", "md5"], [[f"DATA/{k}", v] for k, v in md5.items()])
    rep += ["2. Counts vs the published numbers (D2.1)"]
    rep += table(["corpus", "reviews", "sentences", "opinions", "check"], counts)
    rep += ["3. Harmonisation and structural repairs"]
    rep += table(["corpus", "step", "count"], [[f, s, c] for (f, s), c in sorted(LOG.items())])
    rep += [f"hyphen evidence: {len(compounds)} hyphenated words in SemEval; {sum(v >= 2 for v in pairs.values())} recurring ACOS 'x - y' pairs", "",
            "4. Exact duplicates removed (same split, same text, same labels)"]
    rep += table(["split", "removed", "kept copy", "text"], removed_all)
    rep += ["5. Output"]
    rep += table(["split", "source", "sentences in", "duplicates removed", "sentences out", "opinion rows"], summary)
    rep += [f"rows: train {len(frames['train'])}, test {len(frames['test'])}",
            f"train sentences with an identical text in test (removed in stage 2): {len(overlap)}", ""]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "prepare_data.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'prepare_data.txt').relative_to(ROOT)}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
