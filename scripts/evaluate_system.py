"""End-to-end check of the system (src.main.Pipeline: preprocess -> domain gates -> bge_detr -> post-processing).

1. Laptop test (DATA/final/test_final.csv, SemEval-2016; the gold sentences in review order, so the model sees context):
   joint F1 of (E#A, sentiment) with the gates, and with every sentence sent to the model (gates off).
2. Restaurant test (SemEval-2016 restaurants SB1 test, DATA/HF/EN_REST_SB1_TEST.xml.gold; skipped if absent):
   sentences the system wrongly gives a laptop opinion.
3. Edge cases: hand-written inputs and the output tuples they get.

Output: DATA/final/reports/evaluate_system.txt (the printed report)
Run:    python scripts/evaluate_system.py
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, FINAL_DIR, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.main import Pipeline  # noqa: E402
from src.post_process import BRACKET, brief  # noqa: E402

EDGE_CASES = ["", "   ", "!!!", "????", "12345", "Hello.", "Thanks everyone!", "ok", "The pizza was cold.",
              "I love my new phone.", "restaurant is good and has good food", "The weather is great today.",
              "My printer keeps jamming.", "The waiter was rude.", "Great.", "Do not buy it!", "Apple mac mini",
              "It works.", "Battery life is terrible but the screen is gorgeous.", "This laptop is way better than my old phone.",
              "I watch movies on it all the time and the speakers are loud.", "The keyboard isn't bad at all.",
              "Yeah right, \"premium\" build.", "SOOOOO SLOOOOW!!!!!", "&lt;br&gt;The fan is noisy&lt;br&gt;",
              "The   screen​ is   amazing 😍", "Is the battery worth it?", "If only the trackpad were bigger.",
              "a" * 2000, "laptop " * 300]


def gold() -> pd.DataFrame:
    df = pd.read_csv(FINAL_DIR / "test_final.csv", dtype=str, keep_default_na=False)
    df["sentence_index"] = df.sentence_id.str.split(":").str[1].astype(int)
    return df


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
    g = gold()
    reviews = (g.drop_duplicates("sentence_id").sort_values(["review_id", "sentence_index"])
               .groupby("review_id", sort=False).text.apply(list).reset_index())  # the gold sentences, not re-split
    reviews = pd.DataFrame({"product": "laptop", "review_id": reviews.review_id, "text": reviews.text})
    rep = ["SYSTEM EVALUATION (scripts/evaluate_system.py: src.main.Pipeline)", ""]

    t0 = time.perf_counter()
    on = pipe.run(reviews, presplit=True)
    t1 = time.perf_counter()
    gate_run = pipe.gate.run
    pipe.gate.run = lambda recs: [{"about_laptop": True, "reason": "", "gates": []} for _ in recs]  # gates off
    off = pipe.run(reviews, presplit=True)
    pipe.gate.run = gate_run
    rep += [f"1. laptop test: {len(reviews)} reviews, {len(on)} sentences in {t1 - t0:.1f} s",
            f"   passed the gates: {sum(r['gate']['about_laptop'] for r in on)}/{len(on)}", "",
            f"   {'':<12}{'joint P':>9}{'joint R':>9}{'joint F1':>10}"]
    for name, recs in (("gates on", on), ("gates off", off)):
        p, r, f = joint_f1(recs, g)
        rep.append(f"   {name:<12}{p:>9.1%}{r:>9.1%}{f:>10.1%}")
    lost = [r for r in on if not r["gate"]["about_laptop"] and any(t["sentiment"] for t in
            next(x for x in off if (x["review_id"], x["sentence_index"]) == (r["review_id"], r["sentence_index"]))["tuples"])]
    rep += ["", "   rejected by a gate although the model had an opinion for it:"]
    rep += [f"     {r['gate']['reason'][:45]:<46} {r['sentence'][:90]}" for r in lost[:20]] or ["     none"]

    xml = DATA_DIR / "HF" / "EN_REST_SB1_TEST.xml.gold"
    if xml.exists():
        texts = [s.findtext("text") or "" for s in ET.parse(xml).getroot().iter("sentence")]
        rest = pipe.run(pd.DataFrame({"product": "restaurant", "review_id": range(len(texts)), "text": texts}))
        leaked = [r for r in rest if any(t["sentiment"] for t in r["tuples"])]
        n_off = sum(not r["gate"]["about_laptop"] for r in rest)
        rep += ["", f"2. restaurant test: {len(rest)} sentences; rejected by the gates {n_off} ({n_off / len(rest):.1%}); "
                f"given a laptop opinion {len(leaked)} ({len(leaked) / len(rest):.1%})", "   examples given a laptop opinion:"]
        rep += [f"     {brief(r['tuples'])[:50]:<52}{r['sentence'][:80]}" for r in leaked[:15]]
    else:
        rep += ["", f"2. restaurant test: skipped, {xml.relative_to(ROOT)} not found (SemEval-2016 restaurants SB1 test)"]

    rep += ["", "3. edge cases"]
    for k, text in enumerate(EDGE_CASES):
        recs = pipe.run(pd.DataFrame({"product": "x", "review_id": [k], "text": [text]}))
        shown = text if len(text) <= 60 else f"{text[:40]}... ({len(text)} chars)"
        if not recs:
            rep.append(f"   {shown!r:<64} -> (no sentence)")
        for r in recs:
            note = "" if r["gate"]["about_laptop"] else f"  [{r['gate']['reason'][:50]}]"
            rep.append(f"   {shown!r:<64} -> {brief(r['tuples'])}{note}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluate_system.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {(REPORT_DIR / 'evaluate_system.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
