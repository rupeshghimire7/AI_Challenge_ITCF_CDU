"""Train one model for 5 epochs and save every epoch as a self-contained checkpoint.

  python models/train.py --model bge --variant base
  python models/train.py --model bge --variant tweaked
  python models/train.py --model lfm --variant base
  python models/train.py --model lfm --variant tweaked
  (--variant detr: the tweaked encoder with the DETR-style set-prediction head of models/set_detector.py)

Data: DATA/final/train_final.csv rows with split "train" (training) and "validation" (detection threshold, best epoch),
and DATA/final/test_final.csv, the official SemEval-2016 gold test set: scored every epoch, never used to choose anything.

Variants (VARIANTS):
  base      the pretrained encoder + a linear head (plan M2): BCE detection, star cross-entropy, the sentence alone
  tweaked   the recommended combination in one run (no time for an ablation):
              catattn    L1  one query per category, initialised from the category's name: a good start for rare
                             categories in only 5 epochs
              factorised L2  logit(E#A) = s_E + s_A + residual: rare pairs borrow strength from their entity / attribute
              gate       L3  "has an opinion" gate: 18.5% of train and 29% of test sentences have none
              context    L4  previous and next sentence of the review as context (the brief asks for review context),
                             with context dropout 0.3 so a sentence without its review also works
              asl        L5  asymmetric detection loss + focal, class-balanced star loss: neutral is 6% of opinions
              neg_scope  L10 negation-scope embedding ("not bad", "never had a problem")
              long_aug       +25% inputs of 2-6 joined review sentences: long text at inference
              llrd 0.9       layer-wise lr decay: lower layers change less, keeping pretrained knowledge
  detr      the tweaked encoder + set-prediction head instead of the classifier head

Output, per epoch: result/{model}_{variant}/checkpoint_{epoch}/
  model.pt, config.json, encoder_config.json, tokenizer.json   everything models/infer.py needs (no download)
  metrics.csv            train / validation / test metrics of this epoch (detection threshold fitted on validation)
  test_predictions.csv   every test sentence: gold and predicted tuples with star, 90% star interval, confidence
and result/{model}_{variant}/metrics.csv (every epoch), log.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import bge, lfm  # noqa: E402
from models.data import (BRACKET, SENTIMENTS, Example, Inputs, Scorer, load_examples, long_examples,  # noqa: E402
                         star_summary)
from models.heads import ABSAClassifier  # noqa: E402
from models.set_detector import ABSASetDetector  # noqa: E402
from src.config import ROOT  # noqa: E402

BASES = {"bge": bge, "lfm": lfm}
LR = {"bge": 5e-5, "lfm": 3e-5}  # encoder learning rate
VARIANTS = {
    "base": dict(head="linear", factorised=False, gate=False, loss="bce", context=False, neg_scope=False, long_aug=0.0, llrd=1.0),
    "tweaked": dict(head="catattn", factorised=True, gate=True, loss="asl", context=True, neg_scope=True, long_aug=0.25, llrd=0.9),
    "detr": dict(head="set", factorised=False, gate=False, loss="bce", context=True, neg_scope=True, long_aug=0.25, llrd=0.9),
}
FIXED = dict(head_lr=1e-3, weight_decay=0.01, warmup=0.06, max_len=56, ctx_len=32, ctx_dropout=0.3, queries=16)
COLUMNS = [("detect_p", "det P"), ("detect_r", "det R"), ("detect_f1", "det F1"), ("entity_f1", "ent F1"),
           ("attribute_f1", "attr F1"), ("sentiment_acc", "sent acc"), ("sentiment_macro_f1", "sent mF1"),
           ("joint_p", "joint P"), ("joint_r", "joint R"), ("joint_f1", "joint F1"), ("empty_acc", "empty acc"),
           ("star_mae", "star MAE"), ("interval_cover", "cover90"), ("confidence", "conf")]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=BASES, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0, help="encoder learning rate (default: bge 5e-5, lfm 3e-5)")
    p.add_argument("--grad-ckpt", action="store_true", help="less GPU memory, slower (LFM on a small GPU)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--out", default="result")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0, help="only the first N train sentences (smoke test)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------- setup
def pick_device(name: str) -> torch.device:
    """`auto`: a GPU only if a kernel really runs on it (an old GPU can be visible yet unsupported by this torch)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (torch.ones(8, device="cuda") @ torch.ones(8, device="cuda")).item()
            return torch.device("cuda")
        except Exception:
            pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp(device) -> tuple[bool, torch.dtype]:
    if device.type != "cuda":
        return False, torch.float32
    return True, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def make_inputs(model_name: str, v: dict, tokenizer_file: str | None = None) -> Inputs:
    return Inputs(BASES[model_name].SPEC, FIXED["max_len"], FIXED["ctx_len"], v["context"], v["neg_scope"], tokenizer_file)


def class_balanced(examples: list[Example], beta: float = 0.99) -> list[float]:
    """Star class weights (Cui et al. 2019) from the sentiment counts, mean 1 over the sentiments: neutral gets the most.
    beta 0.99 keeps it mild (neutral ~1.3x): 0.999 gave neutral ~5x and the model then over-predicted neutral."""
    n = Counter(int(BRACKET[s]) for ex in examples for _, _, s in ex.gold)
    w = [(1 - beta) / (1 - beta ** max(n[k], 1)) for k in range(3)]
    return [w[b] * 3 / sum(w) for b in BRACKET]


def build(model_name: str, v: dict, categories: list[str], encoder, pol_weights=None, wrap: bool = False,
          grad_ckpt: bool = False) -> torch.nn.Module:
    """Encoder (pretrained for training, architecture only when loading a checkpoint) -> tweaks -> head."""
    enc = encoder
    if wrap:
        kw = dict(context=v["context"], neg_scope=v["neg_scope"], grad_ckpt=grad_ckpt)
        enc = lfm.TweakedLFM(enc, **kw) if model_name == "lfm" else bge.TweakedBGE(enc, **kw)
    if v["head"] == "set":
        return ABSASetDetector(enc, n_queries=FIXED["queries"])
    return ABSAClassifier(enc, categories, head=v["head"], factorised=v["factorised"], gate=v["gate"], loss=v["loss"],
                          pol_class_weights=pol_weights)


def optimiser(model, v: dict, lr: float, steps: int):
    """AdamW: encoder at `lr` (times llrd per layer below the top), new weights at head_lr; no weight decay on biases and
    norms; linear warm-up then linear decay (plan M2.6)."""
    groups, seen, out = model.encoder.layer_groups(), set(), []

    def add(params, rate):
        out.append({"params": [p for p in params if p.ndim >= 2], "lr": rate, "weight_decay": FIXED["weight_decay"]})
        out.append({"params": [p for p in params if p.ndim < 2], "lr": rate, "weight_decay": 0.0})

    for depth, params in enumerate(groups):
        seen.update(id(p) for p in params)
        add(params, lr * v["llrd"] ** (len(groups) - 1 - depth))
    add([p for p in model.parameters() if id(p) not in seen], FIXED["head_lr"])
    opt = torch.optim.AdamW([g for g in out if g["params"]])
    warm = max(1, int(FIXED["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    return opt, sched


class Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.write_text("", encoding="utf-8")

    def __call__(self, *lines: str) -> None:
        for line in lines:
            print(line, flush=True)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


def table(headers, rows) -> list[str]:
    """Plain-text table: numeric columns right-aligned, text left-aligned."""
    cells = [list(map(str, headers)), *([str(c) for c in r] for r in rows)]
    width = [max(map(len, col)) for col in zip(*cells)]
    right = [all(re.fullmatch(r"[-+\d.,% *]+|", c) for c in col[1:]) for col in zip(*cells)]
    line = lambda r: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(r, width, right)).rstrip()  # noqa: E731
    return [line(cells[0]), line(["-" * w for w in width]), *map(line, cells[1:]), ""]


def metric_rows(results: dict[str, dict]) -> list[list]:
    return [[split, m["n"], *(f"{m[k]:.2f}" if k == "star_mae" else f"{100 * m[k]:.1f}" for k, _ in COLUMNS)]
            for split, m in results.items()]


METRIC_HEAD = ["split", "n", *(h for _, h in COLUMNS)]


def short(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


# ----------------------------------------------------------------------------------------------- predict / save
@torch.no_grad()
def predict(model, inputs: Inputs, examples: list[Example], device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Category probabilities [n, C] and star probabilities [n, C, 5], in the order of `examples`. Batches are sorted
    by length (less padding); long examples are split into windows by Inputs and merged by the model."""
    model.eval()
    use_amp, dtype = amp(device)
    c = len(model.categories)
    det, pol = np.zeros((len(examples), c), np.float32), np.zeros((len(examples), c, 5), np.float32)
    order = sorted(range(len(examples)), key=lambda i: inputs.n_tokens(examples[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        batch = inputs.batch([examples[i] for i in idx]).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            d, p = model.probs(model(batch))
        det[idx], pol[idx] = d.float().cpu().numpy(), p.float().cpu().numpy()
    return det, pol


def describe_opinion(category: str, s: dict, i: int, j: int, score: float | None = None) -> dict:
    """One predicted opinion from star_summary() output `s` at example i, category j."""
    out = {"category": category, "sentiment": SENTIMENTS[int(s["sentiment"][i, j])], "star": int(s["star"][i, j]),
           "expected_star": round(float(s["expected"][i, j]), 2), "star_interval_90": [int(s["lo"][i, j]), int(s["hi"][i, j])],
           "confidence": round(float(s["confidence"][i, j]), 3), "certain": bool(s["certain"][i, j])}
    return out | ({"score": round(score, 3)} if score is not None else {})


def save_checkpoint(folder: Path, model, cfg: dict, inputs: Inputs, metrics: dict[str, dict], test: list[Example],
                    det: np.ndarray, pol: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    enc = model.encoder.base if hasattr(model.encoder, "base") else model.encoder
    BASES[cfg["model"]].save_config(enc, folder)
    shutil.copyfile(inputs.tokenizer_file, folder / "tokenizer.json")
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "n", "tau", *(k for k, _ in COLUMNS), "train_loss"])
        for split, m in metrics.items():
            w.writerow([cfg["epoch"], split, m["n"], m["tau"], *(round(m[k], 4) for k, _ in COLUMNS), round(m.get("loss", 0), 4)])
    s = star_summary(pol)
    cats = model.categories
    with (folder / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "text", "gold", "predicted"])
        for i, ex in enumerate(test):
            gold = "|".join(f"{e}#{a}:{SENTIMENTS[int(BRACKET[st])]}(star {st})" for e, a, st in ex.gold)
            pred = []
            for j in np.flatnonzero(det[i] >= cfg["tau"]):
                o = describe_opinion(cats[j], s, i, j)
                lo, hi = o["star_interval_90"]
                pred.append(f"{cats[j]}:{o['sentiment']}(star {o['star']}, 90% {lo}-{hi}, conf {o['confidence']:.2f})")
            w.writerow([ex.sid, ex.text, gold, "|".join(pred)])


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    a = parse()
    v = VARIANTS[a.variant]
    lr = a.lr or LR[a.model]
    device = pick_device(a.device)
    seed_all(a.seed)
    rng = random.Random(a.seed)
    run = ROOT / a.out / f"{a.model}_{a.variant}"
    run.mkdir(parents=True, exist_ok=True)
    log = Log(run / "log.txt")

    examples = load_examples("train")
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "validation"]
    test = load_examples("test")
    if a.limit:
        train, val, test = train[: a.limit], val[: max(a.limit // 4, 8)], test[: max(a.limit // 4, 8)]
    categories = sorted({c for ex in train for c in ex.categories})
    inputs = make_inputs(a.model, v)
    wrap = v["context"] or v["neg_scope"] or a.grad_ckpt
    model = build(a.model, v, categories, BASES[a.model].load(), class_balanced(train), wrap, a.grad_ckpt).to(device)
    if v["head"] == "catattn":
        model.init_queries(inputs)

    n_long = int(v["long_aug"] * len(train))
    per_epoch = math.ceil((len(train) + n_long) / a.batch_size)
    opt, sched = optimiser(model, v, lr, a.epochs * per_epoch)
    use_amp, dtype = amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    scorers = {"train": Scorer(train, model.categories), "validation": Scorer(val, model.categories),
               "test": Scorer(test, model.categories)}
    sets = {"train": train, "validation": val, "test": test}
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    log(f"RUN {a.model}_{a.variant}  device {device.type}{' (' + str(dtype).split('.')[-1] + ')' if use_amp else ''}  "
        f"encoder {BASES[a.model].REPO} {n_enc:,} parameters + head {n_all - n_enc:,}",
        "settings: " + ", ".join(f"{k} {x}" for k, x in v.items()) + f", encoder lr {lr:g}, head lr {FIXED['head_lr']:g}, "
        f"batch {a.batch_size}, epochs {a.epochs}, max_len {FIXED['max_len']}",
        f"data: train {len(train)} sentences (+{n_long} long inputs per epoch), validation {len(val)}, test {len(test)}; "
        f"{len(model.categories)} categories; {per_epoch} steps per epoch", "")

    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        data = train + long_examples(train, n_long, rng, max_opinions=FIXED["queries"]) if n_long else list(train)
        rng.shuffle(data)
        sums, n = Counter(), 0
        every = max(1, per_epoch // 5)
        for s in range(0, len(data), a.batch_size):
            batch = inputs.batch(data[s : s + a.batch_size], rng, FIXED["ctx_dropout"] if v["context"] else 0.0).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                loss, parts = model.loss(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums.update(parts)
            sums["total"] += float(loss.detach())
            n += 1
            if n % every == 0 or n == per_epoch:
                log(f"  epoch {epoch}/{a.epochs}  step {n:>4}/{per_epoch}  loss {sums['total'] / n:.4f} ("
                    + " ".join(f"{k} {x / n:.4f}" for k, x in sums.items() if k != "total")
                    + f")  lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s")
        train_time = time.time() - t0
        probs = {k: predict(model, inputs, ex, device) for k, ex in sets.items()}
        tau = scorers["validation"].best_tau(*probs["validation"])
        metrics = {k: scorers[k].score(*probs[k], tau) for k in sets}
        metrics["train"]["loss"] = sums["total"] / n
        cfg = {"model": a.model, "variant": a.variant, "epoch": epoch, "tau": tau, "categories": model.categories,
               "settings": v, "fixed": FIXED, "wrapped": bool(wrap), "encoder_lr": lr, "repo": BASES[a.model].REPO,
               "metrics": {k: {m: round(x, 4) for m, x in r.items()} for k, r in metrics.items()}}
        folder = run / f"checkpoint_{epoch}"
        save_checkpoint(folder, model, cfg, inputs, metrics, test, *probs["test"])
        history.append(metrics)
        with (run / "metrics.csv").open("w" if epoch == 1 else "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if epoch == 1:
                w.writerow(["epoch", "split", "n", "tau", *(k for k, _ in COLUMNS)])
            for split, m in metrics.items():
                w.writerow([epoch, split, m["n"], tau, *(round(m[k], 4) for k, _ in COLUMNS)])
        log("", f"EPOCH {epoch}/{a.epochs}  train {train_time:.0f}s, evaluation {time.time() - t0 - train_time:.0f}s, "
                f"threshold {tau:.2f} (best validation joint F1)  -> {short(folder)}",
            *table(METRIC_HEAD, metric_rows(metrics)))

    best = max(range(len(history)), key=lambda i: history[i]["validation"]["joint_f1"])
    log("SUMMARY (%; star MAE in stars)", *table(
        ["epoch", "val joint F1", "test det F1", "test sent acc", "test joint F1", "test empty acc", "test star MAE", "test cover90"],
        [[f"{i + 1}{' *' if i == best else ''}", f"{100 * h['validation']['joint_f1']:.1f}", f"{100 * h['test']['detect_f1']:.1f}",
          f"{100 * h['test']['sentiment_acc']:.1f}", f"{100 * h['test']['joint_f1']:.1f}", f"{100 * h['test']['empty_acc']:.1f}",
          f"{h['test']['star_mae']:.2f}", f"{100 * h['test']['interval_cover']:.1f}"] for i, h in enumerate(history)]),
        f"* best validation epoch: {short(run)}/checkpoint_{best + 1}")


if __name__ == "__main__":
    main()
