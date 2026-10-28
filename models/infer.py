"""Use any saved checkpoint: analyse text, score the gold test set, or compare every run.

  python models/infer.py --checkpoint result/bge_tweaked/checkpoint_5 --text "Great screen, but the battery dies fast."
  python models/infer.py --checkpoint result/bge_tweaked/checkpoint_5 --file reviews.txt     (one review per line)
  python models/infer.py --checkpoint result/bge_base/checkpoint_5 --test                    (DATA/final/test_final.csv)
  python models/infer.py --table                                                              (every run in result/)

Text of any length works: it is split into sentences (src.preprocess, the training normaliser), each sentence is read
with its neighbours as review context, and a sentence longer than one window is scored window by window.
Each opinion: category (E#A), star 0-4 (rounded expected star), 90% star interval, final sentiment (negative /
neutral / positive, from the star brackets 0-1 / 2 / 3-4 and the interval, see data.star_summary), its confidence,
and the detection score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import Example, Scorer, load_examples, star_summary  # noqa: E402
from models.train import (BASES, COLUMNS, METRIC_HEAD, build, describe_opinion, make_inputs, metric_rows,  # noqa: E402
                          pick_device, predict, table)
from src.config import ROOT  # noqa: E402
from src.preprocess import split_sentences  # noqa: E402


def load_checkpoint(folder: str | Path, device="cpu"):
    """A checkpoint folder -> (model, inputs, config). Needs only the folder: no download, no training data."""
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    v = cfg["settings"]
    encoder = BASES[cfg["model"]].from_config(folder)
    model = build(cfg["model"], v, cfg["categories"], encoder, wrap=cfg["wrapped"])
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    inputs = make_inputs(cfg["model"], v, str(folder / "tokenizer.json"))
    return model.to(device).eval(), inputs, cfg


def analyse(review: str, model, inputs, cfg: dict, device) -> list[dict]:
    """Review text -> one record per sentence with its opinions."""
    sents = split_sentences(review)
    if not sents:
        return []
    examples = [Example(f"s:{k}", "input", t, sents[k - 1] if k else "", sents[k + 1] if k + 1 < len(sents) else "")
                for k, t in enumerate(sents)]
    det, pol = predict(model, inputs, examples, device)
    s = star_summary(pol)
    out = []
    for i, ex in enumerate(examples):
        ops = [describe_opinion(model.categories[j], s, i, j, float(det[i, j])) for j in np.flatnonzero(det[i] >= cfg["tau"])]
        out.append({"sentence": ex.text, "opinions": sorted(ops, key=lambda o: -o["score"])})
    return out


def show(records: list[dict]) -> None:
    for r in records:
        tuples = ", ".join(f"{{{o['category']}, {o['sentiment']}}}" for o in r["opinions"]) or "(no opinion)"
        print(f"{r['sentence']}\n  -> {tuples}")
        for o in r["opinions"]:
            lo, hi = o["star_interval_90"]
            print(f"     {o['category']:<36} star {o['star']} (expected {o['expected_star']:.2f}, 90% interval {lo}-{hi})  "
                  f"{o['sentiment']} {100 * o['confidence']:.0f}%{'' if o['certain'] else ' (uncertain)'}  "
                  f"detection {o['score']:.2f}")
    print(json.dumps(records, ensure_ascii=False))


def evaluate(folder: Path, device) -> None:
    model, inputs, cfg = load_checkpoint(folder, device)
    test = load_examples("test")
    det, pol = predict(model, inputs, test, device)
    m = Scorer(test, model.categories).score(det, pol, cfg["tau"])
    print(f"{folder}: {cfg['model']}_{cfg['variant']} epoch {cfg['epoch']}, threshold {cfg['tau']:.2f} (from validation)")
    print("\n".join(table(METRIC_HEAD, metric_rows({"test": m}))))


def tabulate(out: str) -> None:
    """Every run in result/: the epoch with the best validation joint F1 and the last epoch, test metrics."""
    rows, keep = [], []
    for path in sorted((ROOT / out).glob("*/metrics.csv")):
        with path.open(encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        val = {int(r["epoch"]): float(r["joint_f1"]) for r in data if r["split"] == "validation"}
        if not val:
            continue
        best, last = max(val, key=val.get), max(val)
        for tag, ep in (("best val", best), ("last", last)):
            t = next(r for r in data if r["split"] == "test" and int(r["epoch"]) == ep)
            rows.append([path.parent.name, f"{ep} ({tag})", f"{100 * val[ep]:.1f}",
                         *(f"{float(t[k]):.2f}" if k == "star_mae" else f"{100 * float(t[k]):.1f}" for k, _ in COLUMNS)])
            keep.append({"run": path.parent.name, "epoch": ep, "which": tag, "val_joint_f1": val[ep], **{k: t[k] for k, _ in COLUMNS}})
    if not rows:
        sys.exit(f"no runs in {out}/: train first (models/train.py)")
    print("TEST SET (official SemEval-2016 gold), %; star MAE in stars")
    print("\n".join(table(["run", "epoch", "val joint F1", *(h for _, h in COLUMNS)], rows)))
    with (ROOT / out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0]))
        w.writeheader()
        w.writerows(keep)
    print(f"saved {out}/summary.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="a result/{model}_{variant}/checkpoint_{epoch} folder")
    p.add_argument("--text", help="review text (any length)")
    p.add_argument("--file", help="text file, one review per line")
    p.add_argument("--test", action="store_true", help="score the checkpoint on DATA/final/test_final.csv")
    p.add_argument("--table", action="store_true", help="compare every run in --out")
    p.add_argument("--out", default="result")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.table:
        return tabulate(a.out)
    if not a.checkpoint:
        p.error("--checkpoint is required (or use --table)")
    device = pick_device(a.device)
    if a.test:
        return evaluate(Path(a.checkpoint), device)
    texts = [a.text] if a.text is not None else []
    if a.file:
        texts += [line for line in Path(a.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        p.error("give --text, --file or --test")
    model, inputs, cfg = load_checkpoint(a.checkpoint, device)
    for t in texts:
        show(analyse(t, model, inputs, cfg, device))


if __name__ == "__main__":
    main()
