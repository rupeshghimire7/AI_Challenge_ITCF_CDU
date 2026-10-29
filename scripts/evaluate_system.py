"""End-to-end check of the system (src.main.Pipeline: preprocess -> domain gates -> bge_detr -> post-processing).

Laptop sets: DATA/final/train_final.csv (split == validation: the 15% of train reviews the model was not trained on) and
test_final.csv (the official SemEval-2016 gold set). The gold sentences are read in review order, so the model sees its
review context. Score: joint F1 of (E#A, sentiment) against the gold, with the system's E#A (`category`).

1. Fit the second threshold (post_process.TAU_LOW) on validation: joint F1 for each candidate.
2. Ablation on validation and test: each post-processing rule added in turn, and the gates switched off.
3. Laptop test sentences the gates reject although the model had an opinion for them; merged sentences.
4. Restaurant test (SemEval-2016 restaurants SB1 test, DATA/HF/EN_REST_SB1_TEST.xml.gold; skipped if absent).
5. Edge cases: hand-written inputs and the output tuples they get.

Output: DATA/final/reports/evaluate_system.txt (the printed report)
Run:    python scripts/evaluate_system.py
"""

from __future__ import annotations

import copy
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import post_process  # noqa: E402
from src.config import DATA_DIR, FINAL_DIR, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.main import Pipeline  # noqa: E402
from src.post_process import BRACKET, brief  # noqa: E402

GRID = [None, 0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82]
EDGE_CASES = ["", "   ", "!!!", "????", "12345", "Hello.", "Thanks everyone!", "ok", "The pizza was cold.",
              "I love my new phone.", "restaurant is good and has good food", "The weather is great today.",
              "My printer keeps jamming.", "The waiter was rude.", "Great.", "Do not buy it!", "It is the worst laptop ever.",
              "Apple mac mini", "It works.", "Battery life is terrible but the screen is gorgeous.",
              'Yeah right, "premium" build. The hinge broke after two weeks. Not bad for the price though.',
              "Hello everyone! Apple mac mini. The trackpad is smooth and the 16GB of RAM handles everything.",
              "Called tech support about the screen. Gave me the runaround for an entire day and treated me like an imbecile.",
              "The fan is noisy.", "This laptop is way better than my old phone.", "The keyboard isn't bad at all.",
              "SOOOOO SLOOOOW!!!!!", "&lt;br&gt;The fan is noisy&lt;br&gt;", "The   screen​ is   amazing 😍",
              "Is the battery worth it?", "If only the trackpad were bigger.", "a" * 2000, "laptop " * 300]


def gold(split: str) -> pd.DataFrame:
    df = pd.read_csv(FINAL_DIR / f"{split}_final.csv", dtype=str, keep_default_na=False)
    if split == "train":
        df = df[df.split == "validation"].copy()
    df["sentence_index"] = df.sentence_id.str.split(":").str[1].astype(int)
    return df


def reviews_of(g: pd.DataFrame) -> pd.DataFrame:
    r = (g.drop_duplicates("sentence_id").sort_values(["review_id", "sentence_index"])
         .groupby("review_id", sort=False).text.apply(list).reset_index())  # the gold sentences, not re-split
    return pd.DataFrame({"product": "laptop", "review_id": r.review_id, "text": r.text})


def joint_f1(records: list[dict], g: pd.DataFrame) -> tuple[float, float, float]:
    truth = {(r.review_id, r.sentence_index, f"{r.category}#{r.attribute}", POLARITIES[BRACKET[int(r.polarity)]])
             for r in g.itertuples() if r.category}
    pred = {(str(r["review_id"]), r["sentence_index"], t["category"], t["sentiment"])
            for r in records for t in r["tuples"] if t["sentiment"]}
    tp = len(truth & pred)
    p, rc = tp / max(len(pred), 1), tp / max(len(truth), 1)
    return p, rc, 2 * p * rc / max(p + rc, 1e-9)


def main() -> None:
    pipe = Pipeline()
    sets = {s: gold(s) for s in ("train", "test")}
    rep = ["SYSTEM EVALUATION (scripts/evaluate_system.py: src.main.Pipeline)", ""]
    analysed, t0 = {}, time.perf_counter()
    for s, g in sets.items():
        analysed[s] = pipe.analyse(reviews_of(g), presplit=True)
    t1 = time.perf_counter()
    n = sum(len(a) for a in analysed.values())
    rep.append(f"{n} sentences (validation {len(analysed['train'])}, test {len(analysed['test'])}): preprocessing, gates "
               f"and model in {t1 - t0:.1f} s ({(t1 - t0) / n:.3f} s per sentence)")

    def score(split: str, recs: list[dict] | None = None, **kw) -> tuple[float, float, float]:
        return joint_f1(pipe.finish(copy.deepcopy(recs or analysed[split]), **kw), sets[split])

    def cells(scores: list[tuple[float, float, float]]) -> str:
        return "".join(f"{f'{f:.1%} ({p:.1%} / {r:.1%})':>24}" for p, r, f in scores)

    rep += ["", f"1. second threshold, fitted on validation (part re-map {'on' if post_process.REMAP else 'off'}, merge off)",
            f"   {'TAU_LOW':<10}{'joint P':>9}{'joint R':>9}{'joint F1':>10}"]
    fits = {}
    for t in GRID:
        fits[t] = score("train", tau_low=t, remap=post_process.REMAP, merge=False)
        rep.append(f"   {'off' if t is None else f'{t:.2f}':<10}" + "".join(f"{x:>9.1%}" for x in fits[t][:2])
                   + f"{fits[t][2]:>10.1%}")
    best = max(GRID, key=lambda t: (round(fits[t][2], 4), t or 1.0))  # ties: the higher threshold
    rep.append(f"   best: {best}; src/post_process.py has TAU_LOW = {post_process.TAU_LOW}"
               + ("" if best == post_process.TAU_LOW else "  <- update it"))

    tl = post_process.TAU_LOW
    variants = [("model only (tau, entity NULL rule)", dict(tau_low=None, remap=False, merge=False)),
                (f"+ second threshold {tl}", dict(tau_low=tl, remap=False, merge=False)),
                ("+ merge of NULL#NULL sentences (default)", dict(tau_low=tl, remap=False, merge=True)),
                ("default + part re-map (REMAP option)", dict(tau_low=tl, remap=True, merge=True))]
    rep += ["", "2. ablation, joint F1 (P / R)", f"   {'':<42}{'validation':>24}{'test':>24}"]
    for label, kw in variants:
        rep.append(f"   {label:<42}" + cells([score(s, **kw) for s in ("train", "test")]))
    run = pipe.gate.run
    pipe.gate.run = lambda recs: [g | {"about_laptop": True} for g in run(recs)]  # gates off
    off = {s: pipe.analyse(reviews_of(sets[s]), presplit=True) for s in sets}
    pipe.gate.run = run
    rep.append(f"   {'default, gates off':<42}" + cells([score(s, off[s]) for s in ("train", "test")]))

    on, off_t = pipe.finish(analysed["test"]), pipe.finish(off["test"])
    lost = [r for r, o in zip(on, off_t) if not r["gate"]["about_laptop"] and any(t["sentiment"] for t in o["tuples"])]
    rep += ["", f"3. laptop test: the gates reject {sum(not r['gate']['about_laptop'] for r in on)}/{len(on)} sentences; "
            f"the model had an opinion for {len(lost)} of them:"]
    rep += [f"     {r['gate']['reason'][:48]:<50}{r['sentence'][:85]}" for r in lost[:20]] or ["     none"]
    merged = [r for r in on if r["merged"]]
    rep += ["", f"   merged sentences on laptop test: {len(merged)}"]
    rep += [f"     {brief(r['tuples'])[:48]:<50}{r['sentence'][:85]}" for r in merged[:10]]

    xml = DATA_DIR / "HF" / "EN_REST_SB1_TEST.xml.gold"
    if xml.exists():
        texts = [s.findtext("text") or "" for s in ET.parse(xml).getroot().iter("sentence")]
        rest = pipe.run(pd.DataFrame({"product": "restaurant", "review_id": range(len(texts)), "text": texts}))
        leaked = [r for r in rest if any(t["sentiment"] for t in r["tuples"])]
        n_off = sum(not r["gate"]["about_laptop"] for r in rest)
        rep += ["", f"4. restaurant test: {len(rest)} sentences; rejected by the gates {n_off} ({n_off / len(rest):.1%}); "
                f"given a laptop opinion {len(leaked)} ({len(leaked) / len(rest):.1%})"]
    else:
        rep += ["", f"4. restaurant test: skipped, {xml.relative_to(ROOT)} not found (SemEval-2016 restaurants SB1 test)"]

    rep += ["", "5. edge cases"]
    for k, text in enumerate(EDGE_CASES):
        recs = pipe.run(pd.DataFrame({"product": "x", "review_id": [k], "text": [text]}))
        rep.append(f"   {text if len(text) <= 70 else f'{text[:40]}... ({len(text)} chars)'!r}")
        if not recs:
            rep.append("     -> (no sentence)")
        for r in recs:
            note = (f"  [merged with the {r['merged']['with']} sentence]" if r["merged"] else "" if r["gate"]["about_laptop"]
                    else f"  [{r['gate']['reason'][:60]}]")
            rep.append(f"     {r['sentence'][:50]!r:<54} -> {brief(r['tuples'])}{note}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluate_system.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'evaluate_system.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
