"""Check the system's text preprocessing (src/preprocess.py, src/nuance_checker.py) on fixed cases and on the corpus.

1. Unit cases: normalisation, sentence splitting, match_key, artefact and nuance detection. The script stops on the
   first failing case.
2. Corpus (needs DATA/final/*_merged_cleaned.csv from clean_data.py):
   - normalise() is idempotent on every cleaned sentence and leaves no artefact on the raw source text;
   - the splitter against SemEval's gold segmentation (each review's gold sentences joined with a space, split again);
   - preprocess() throughput on whole SemEval reviews.

Output: DATA/final/reports/preprocess.txt (the printed report)
Run:    python scripts/preprocess.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FINAL_DIR, REPORT_DIR, ROOT  # noqa: E402
from src.data_maker import load_long, to_sentences  # noqa: E402
from src.nuance_checker import ARTEFACTS, NUANCES, artefacts, nuances  # noqa: E402
from src.preprocess import match_key, normalise, preprocess, split_sentences  # noqa: E402

NORMALISE = {  # raw -> normalised
    "And he said&quot; this": 'And he said" this',
    "my 52&amp;#34; TV": 'my 52" TV',
    "Do not\xa0 buy it!": "Do not buy it!",
    "  It’s “great”—really  ": "It's \"great\" - really",
    "soooooooooo big!!!!!!": "sooo big!!!",
    "FOooooooRRREVER....": "FOoooRRREVER...",
    "Works<br />well<br/>": "Works well",
    "zero​width \U0001F44D\U0001F3FD emoji": "zerowidth \U0001F44D\U0001F3FD emoji",
    "$1000000 and 10000mAh": "$1000000 and 10000mAh",
    "I LOVE it :)": "I LOVE it :)",
    "- light and fast": "light and fast",
    "• 2) * Blue screen": "Blue screen",
    "+ cheap, > 8 hours": "+ cheap, > 8 hours",
    "*love* it, -5 stars, 3.5 GHz": "*love* it, -5 stars, 3.5 GHz",
}
SPLIT = {  # review -> sentences
    "Great screen. The battery lasts 5 hrs. Keyboard is ok.": ["Great screen.", "The battery lasts 5 hrs.", "Keyboard is ok."],
    "Fast, e.g. boots in 8 sec. and the 15.6\" screen is nice!": ["Fast, e.g. boots in 8 sec. and the 15.6\" screen is nice!"],
    "great laptop. fast and light!! would buy again": ["great laptop.", "fast and light!!", "would buy again"],
    "It works great :) The keys are soft... but quiet.": ["It works great :)", "The keys are soft... but quiet."],
    "Pros:<br>- light<br>- cheap<br><br>Cons: slow": ["Pros:", "light", "cheap", "Cons: slow"],
    "Mr. Jobs would cry. Really?! Yes.": ["Mr. Jobs would cry.", "Really?!", "Yes."],
    "Do not buy it! !!": ["Do not buy it! !!"],
    "I hate Windows 8. The keys are fine.": ["I hate Windows 8.", "The keys are fine."],
    "": [],
}
ARTEFACT_CASES = {  # one positive example per artefact; each clean text must have none
    "html_entity": "screen &amp; keys", "html_tag": "fast<br>quiet", "unicode_compat": "ＳＳＤ is fast",
    "typographic_punct": "it’s fine", "invisible_char": "zero​width", "irregular_space": "too  many spaces",
    "list_marker": "- light", "elongation": "sooooo good", "punct_run": "wow!!!!!",
}
NUANCE_CASES = {  # (positive, negative) per nuance
    "negation": ("The keys don't stick.", "The keys are great."),
    "contrast": ("Fast, but loud.", "Fast and quiet."),
    "comparison": ("Better than my old Dell.", "A good laptop."),
    "sarcasm_cand": ('Payed $2200 for a "premium" laptop', "Payed $2200 for a laptop"),
    "numeric": ("Battery lasts 8 hours.", "Battery lasts all day."),
    "question": ("Is it worth the money", "Would definitely recommend"),
    "conditional": ("I wish it had more ports.", "It has many ports."),
}


def unit_cases() -> int:
    n = 0

    def expect(ok: bool, what: str) -> None:
        nonlocal n
        if not ok:
            sys.exit(f"unit case failed: {what}")
        n += 1

    for raw, want in NORMALISE.items():
        got = normalise(raw)
        expect(got == want, f"normalise({raw!r}) = {got!r}, want {want!r}")
        expect(normalise(got) == got, f"normalise not idempotent on {raw!r}")
        expect(not artefacts(got), f"artefacts left in normalise({raw!r}): {artefacts(got)}")
    for raw, want in SPLIT.items():
        expect(split_sentences(raw) == want, f"split_sentences({raw!r}) = {split_sentences(raw)!r}")
    expect(match_key("I LOVE it!") == match_key("i love it !") == "i love it", "match_key")
    expect(set(ARTEFACT_CASES) == set(ARTEFACTS) and set(NUANCE_CASES) == set(NUANCES), "every artefact and nuance has a case")
    for name, text in ARTEFACT_CASES.items():
        expect(name in artefacts(text), f"artefact {name} not found in {text!r}")
        expect(not artefacts(normalise(text)), f"normalise left {artefacts(normalise(text))} in {text!r}")
    for name, (pos, neg) in NUANCE_CASES.items():
        expect(name in nuances(pos) and name not in nuances(neg), f"nuance {name}: {nuances(pos)} / {nuances(neg)}")
    rec = preprocess("Sooooo fast!!!! But the fan is loud.")
    expect(rec == [{"text": "Sooo fast!!!", "artefacts": ["elongation", "punct_run"], "nuances": []},
                   {"text": "But the fan is loud.", "artefacts": [], "nuances": ["contrast"]}], f"preprocess record {rec}")
    return n


def corpus_checks(sents: pd.DataFrame) -> list[str]:
    out = [f"cleaned sentences: {len(sents)}"]
    out.append(f"normalise idempotent: {int((sents.text.map(normalise) == sents.text).sum())}/{len(sents)}")
    left = sents.text_raw.map(lambda t: artefacts(normalise(t)))
    out.append(f"raw source text with an artefact left after normalise: {int((left.map(len) > 0).sum())}/{len(sents)}")

    sem = sents[sents.source == "semeval16"]
    gold = sem.groupby(["split", "review_id"], sort=False).text.agg(list)  # cleaning may have removed a few sentences
    tp = n_pred = n_gold = exact = 0
    for sentences in gold:
        pred = split_sentences(" ".join(sentences))
        b_gold = {sum(len(s) + 1 for s in sentences[: i + 1]) for i in range(len(sentences) - 1)}
        b_pred = {sum(len(s) + 1 for s in pred[: i + 1]) for i in range(len(pred) - 1)}
        tp, n_pred, n_gold = tp + len(b_gold & b_pred), n_pred + len(b_pred), n_gold + len(b_gold)
        exact += pred == sentences
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    out += ["", f"splitter vs SemEval gold boundaries ({len(gold)} reviews)",
            f"precision {p:.3f}  recall {r:.3f}  F1 {2 * p * r / max(p + r, 1e-9):.3f}  reviews split exactly: {exact}/{len(gold)}"]

    reviews = sem.groupby(["split", "review_id"], sort=False).text_raw.agg(" ".join).tolist()
    t0 = time.perf_counter()
    n = sum(len(preprocess(r)) for r in reviews)
    dt = time.perf_counter() - t0
    out += ["", f"preprocess() throughput: {len(reviews)} reviews -> {n} sentences in {dt:.2f}s "
            f"({n / dt:,.0f} sentences/s, {1000 * dt / len(reviews):.2f} ms per review, single CPU thread)"]
    return out


def main() -> None:
    rep = ["PREPROCESSING CHECK (scripts/preprocess.py: src/preprocess.py, src/nuance_checker.py)", ""]
    rep.append(f"unit cases passed: {unit_cases()}")
    rep.append(f"artefacts checked: {', '.join(ARTEFACTS)}")
    rep += [f"nuances checked: {', '.join(NUANCES)}", ""]
    paths = [FINAL_DIR / f"{s}_merged_cleaned.csv" for s in ("train", "test")]
    if all(p.exists() for p in paths):
        rep += corpus_checks(pd.concat([to_sentences(load_long(p)) for p in paths], ignore_index=True))
    else:
        rep.append("corpus checks skipped: run prepare_data.py and clean_data.py first")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "preprocess.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'preprocess.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
