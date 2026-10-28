"""ABSA pipeline: reviews -> each sentence's opinions in the brief's format -> the aspect-sentiment table per product.

Flow per review: src/preprocess.py (normalise, split into sentences, aspect cues) -> src/domain_gate.py (the gates: only
a sentence about the laptop goes on) -> src/model_inference.py (bge_detr at epoch 9, the model chosen in
models/notebooks/ABSA_model_selection_report.md) -> src/post_process.py (output tuples, stars, faces, the aspect table).
The Streamlit app (src/ui_streamlit.py) runs the same Pipeline.

Run from the repo root:
    python -m src.main --text "The So called laptop Runs to Slow and I hate it!" --text "Do not buy it!"
    python -m src.main --file reviews.txt --product "Apple Mac mini"        one review per line, all of one product
    python -m src.main --csv reviews.csv --text-column review --product-column asin --out output/
    python -m src.main --csv reviews.csv --text-column review --product-column asin --product B0074703CM
                                                                            only that product's reviews
Output:
    printed     every sentence with its {E#A, sentiment} tuples, then the aspect-sentiment table of each product
    --out DIR   DIR/opinions.csv: one row per opinion with star, 90% star interval, confidence and detection score
                (a sentence without opinions: one row with an empty category); DIR/aspect_table.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import CHECKPOINT
from src.domain_gate import DomainGate
from src.model_inference import ABSAModel, with_context
from src.post_process import aspect_table, brief, decode, finalise, opinion_rows
from src.preprocess import normalise, preprocess, sentence_record


def read_reviews(a: argparse.Namespace) -> pd.DataFrame:
    """-> one row per review: product, review_id (its line / row number), text."""
    if a.csv:
        d = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
        for c in (a.text_column, a.product_column):
            if c and c not in d.columns:
                sys.exit(f"{a.csv} has no column {c!r}; columns: {list(d.columns)}")
        d = pd.DataFrame({"product": d[a.product_column] if a.product_column else a.product or "reviews",
                          "review_id": d.index, "text": d[a.text_column]})
        if a.product_column and a.product:
            d = d[d["product"].str.casefold() == a.product.casefold()]
            if d.empty:
                sys.exit(f"no reviews of {a.product!r} in {a.csv}")
        return d
    texts = a.text or [line for line in Path(a.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame({"product": a.product or "reviews", "review_id": range(len(texts)), "text": texts})


class Pipeline:
    """The model and the gates, loaded once. run(reviews) -> one record per sentence: product, review_id,
    sentence_index, sentence, the preprocess() fields, `gate`, `opinions` (the model's) and `tuples` (the output).
    presplit=True: each review's `text` is a list of sentences, kept as they are (evaluation on gold sentences)."""

    def __init__(self, checkpoint: str | Path = CHECKPOINT, device: str = "auto"):
        self.model = ABSAModel(checkpoint, device)
        self.gate = DomainGate(self.model.device)

    def run(self, reviews: pd.DataFrame, presplit: bool = False) -> list[dict]:
        records, sentences = [], []
        for r in reviews.itertuples(index=False):
            recs = ([sentence_record(t, c) for t in r.text if (c := normalise(t))] if presplit else preprocess(r.text))
            for k, (rec, s) in enumerate(zip(recs, with_context([x["text"] for x in recs]))):
                records.append({"product": r.product, "review_id": r.review_id, "sentence_index": k, "sentence": s.text}
                               | rec)
                sentences.append(s)
        for rec, g in zip(records, self.gate.run(records)):
            rec["gate"] = g
        keep = [i for i, r in enumerate(records) if r["gate"]["about_laptop"]]  # only these reach the model
        for r in records:
            r["opinions"] = []
        if keep:
            det, pol = self.model.predict([sentences[i] for i in keep])
            for i, ops in zip(keep, decode(det, pol, self.model.categories, self.model.tau)):
                records[i]["opinions"] = ops
        for r in records:
            r["tuples"] = finalise(r["opinions"], r)
        return records


def main() -> None:
    ap = argparse.ArgumentParser(description="ABSA pipeline with the selected model (see the top of this file).")
    given = ap.add_mutually_exclusive_group(required=True)
    given.add_argument("--text", action="append", help="a review; repeat the flag for several")
    given.add_argument("--file", help="text file, one review per line")
    given.add_argument("--csv", help="CSV with one review per row")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--product-column", help="CSV column with the product name / id (one table per product)")
    ap.add_argument("--product", help="product name for --text / --file; with --product-column, keep only this product")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out", help="folder for opinions.csv and aspect_table.csv")
    a = ap.parse_args()

    reviews = read_reviews(a)
    t0 = time.perf_counter()
    pipe = Pipeline(a.checkpoint, a.device)
    t1 = time.perf_counter()
    records = pipe.run(reviews)
    t2 = time.perf_counter()
    m = pipe.model
    print(f"bge_detr epoch {m.epoch}, threshold {m.tau:.2f}, {m.device.type}: loaded in {t1 - t0:.1f} s; {len(reviews)} "
          f"reviews, {len(records)} sentences ({sum(r['gate']['about_laptop'] for r in records)} passed the gates) in "
          f"{t2 - t1:.2f} s", file=sys.stderr)

    review = None
    for r in records:
        if (r["product"], r["review_id"]) != review:
            review = (r["product"], r["review_id"])
            print(f"\n[{r['product']}, review {r['review_id']}]")
        note = "" if r["gate"]["about_laptop"] else f"   [gate: {r['gate']['reason']}]"
        print(f"  {r['sentence']}\n    -> {brief(r['tuples'])}{note}")

    rows = opinion_rows(records)
    table = aspect_table(rows)
    if table.empty:
        print("\nAspect-sentiment table: no opinions found")
    for product, t in table.groupby("product", sort=False):
        print(f"\nAspect-sentiment table: {product}")
        print(t.drop(columns="product").to_string(index=False))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / "opinions.csv", index=False)
        table.to_csv(out / "aspect_table.csv", index=False)
        print(f"\nwrote {out / 'opinions.csv'} ({len(rows)} rows) and {out / 'aspect_table.csv'} ({len(table)} rows)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
