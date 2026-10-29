"""Train a three-head ABSA model from CSV or JSONL.

Accepted annotation formats:
1) CSV: columns `text` and `gold`; `gold` is JSON like
   [{"category":"FOOD#QUALITY","attribute":"quality","polarity":"positive"}, ...]
2) JSONL: {"text":"...", "gold":[{"category":"...", "attribute":"...", "polarity":"..."}]}
3) One opinion per CSV row: columns `text`, `category`, `attribute`, `polarity`.
   Rows with the same text are grouped into one example.

For category/attribute labels, the model learns multi-label sentence-level presence.
Polarity is learned for each category x attribute pair. This standalone trainer does not
assume the private project-specific `models.data` or `src.config` APIs.

Usage (from the repo root; the script imports custom_model from its own folder):
    python models/custom_train.py --data DATA/final/train_final.csv --output-dir custom_results/run
    python models/custom_train.py --data reviews.jsonl --epochs 15 --batch-size 32
    python models/custom_train.py --data reviews.jsonl --resume custom_results/run/last.pt
    python models/custom_train.py -h                                  every flag and its default
The input is split at random into train, validation (--val-ratio, 0.15) and test (--test-ratio,
0.10); a `split` column is ignored. Training stops early after --patience epochs without a better
validation F1. Written to --output-dir: best.pt (best validation F1), last.pt (for --resume),
train.log, test_metrics.json and labels.json. Run predictions with custom_inference.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from custom_model import CustomABSAModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV or JSONL training data")
    p.add_argument("--text-col", default="text")
    p.add_argument("--gold-col", default="gold", help="JSON annotation column for CSV")
    p.add_argument("--model-name", default="BAAI/bge-small-en-v1.5",
                   help="HF model id/path; e.g. BAAI/bge-small-en-v1.5 or an LFM2.5 encoder path")
    p.add_argument("--output-dir", default="custom_results/run")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", default="", help="checkpoint path to resume from")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_gold(value):
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    out = []
    for item in parsed:
        if isinstance(item, dict):
            c = item.get("category", item.get("entity", item.get("aspect_category")))
            a = item.get("attribute", item.get("aspect", item.get("aspect_term")))
            p = item.get("polarity", item.get("sentiment"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            c, a, p = item[:3]
        else:
            raise ValueError(f"Cannot parse gold annotation item: {item!r}")
        if c is not None and a is not None and p is not None:
            out.append((str(c), str(a), str(p).lower()))
    return out


def load_rows(path, text_col, gold_col):
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        raw = obj if isinstance(obj, list) else obj.get("data", obj.get("examples", []))
    else:
        with path.open(newline="", encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f))
    grouped = {}
    for row in raw:
        text = str(row.get(text_col, "") or "")
        if row.get(gold_col) not in (None, ""):
            gold = parse_gold(row.get(gold_col))
        elif row.get("category") and row.get("attribute") and row.get("polarity"):
            gold = [(str(row["category"]), str(row["attribute"]), str(row["polarity"]).lower())]
        elif isinstance(row.get("gold"), list):
            gold = parse_gold(row["gold"])
        else:
            gold = []
        key = text
        if key not in grouped:
            grouped[key] = {"text": text, "gold": []}
        grouped[key]["gold"].extend(gold)
    examples = list(grouped.values())
    for ex in examples:
        ex["gold"] = sorted(set(ex["gold"]))
    if not examples:
        raise ValueError("No examples loaded. Check --data and --text-col.")
    return examples


class ABSADataset(Dataset):
    def __init__(self, examples, tokenizer, categories, attributes, polarities, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cat_to_id = {v:i for i,v in enumerate(categories)}
        self.attr_to_id = {v:i for i,v in enumerate(attributes)}
        self.pol_to_id = {v:i for i,v in enumerate(polarities)}
        self.max_length = max_length
        self.C, self.A = len(categories), len(attributes)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        tok = self.tokenizer(ex["text"], truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        cat = torch.zeros(self.C, dtype=torch.float32)
        attr = torch.zeros(self.A, dtype=torch.float32)
        pol = torch.full((self.C, self.A), -100, dtype=torch.long)
        for c, a, p in ex["gold"]:
            if c in self.cat_to_id and a in self.attr_to_id and p in self.pol_to_id:
                ci, ai, pi = self.cat_to_id[c], self.attr_to_id[a], self.pol_to_id[p]
                cat[ci] = 1
                attr[ai] = 1
                pol[ci, ai] = pi
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "category_targets": cat,
            "attribute_targets": attr,
            "polarity_targets": pol,
            "gold": ex["gold"],
            "text": ex["text"],
        }


def collate(items):
    out = {}
    for k in ("input_ids", "attention_mask", "category_targets", "attribute_targets", "polarity_targets"):
        out[k] = torch.stack([x[k] for x in items])
    out["gold"] = [x["gold"] for x in items]
    out["text"] = [x["text"] for x in items]
    return out


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def predict_triples(model, batch, categories, attributes, polarities, threshold=0.5):
    out = model(batch["input_ids"], batch["attention_mask"])
    cat_p = out["category_logits"].sigmoid()
    attr_p = out["attribute_logits"].sigmoid()
    pol_p = out["polarity_logits"].softmax(-1)
    preds = []
    for i in range(cat_p.shape[0]):
        cs = (cat_p[i] >= threshold).nonzero().flatten().tolist()
        ats = (attr_p[i] >= threshold).nonzero().flatten().tolist()
        row = set()
        for ci in cs:
            for ai in ats:
                pi = int(pol_p[i, ci, ai].argmax())
                row.add((categories[ci], attributes[ai], polarities[pi]))
        preds.append(row)
    return preds


def metric_report(model, loader, device, categories, attributes, polarities, threshold=0.5):
    model.eval()
    tp = fp = fn = cat_tp = cat_fp = cat_fn = attr_tp = attr_fp = attr_fn = 0
    pol_correct = pol_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["input_ids"], batch["attention_mask"])
            cat_pred = out["category_logits"].sigmoid() >= threshold
            attr_pred = out["attribute_logits"].sigmoid() >= threshold
            cat_gold = batch["category_targets"].bool()
            attr_gold = batch["attribute_targets"].bool()
            cat_tp += int((cat_pred & cat_gold).sum()); cat_fp += int((cat_pred & ~cat_gold).sum()); cat_fn += int((~cat_pred & cat_gold).sum())
            attr_tp += int((attr_pred & attr_gold).sum()); attr_fp += int((attr_pred & ~attr_gold).sum()); attr_fn += int((~attr_pred & attr_gold).sum())
            pol_pred = out["polarity_logits"].argmax(-1)
            pol_gold = batch["polarity_targets"]
            valid = pol_gold.ne(-100)
            pol_correct += int(((pol_pred == pol_gold) & valid).sum())
            pol_total += int(valid.sum())
            predicted = predict_triples(model, batch, categories, attributes, polarities, threshold)
            for pred, gold in zip(predicted, batch["gold"]):
                gold = set(gold)
                tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2*p*r / max(p+r, 1e-12)
        return {"precision": p, "recall": r, "f1": f}
    return {"triple": prf(tp,fp,fn), "category": prf(cat_tp,cat_fp,cat_fn),
            "attribute": prf(attr_tp,attr_fp,attr_fn),
            "polarity_accuracy": pol_correct / max(pol_total,1), "polarity_pairs": pol_total}


def save_checkpoint(path, model, optimizer, epoch, best_metric, args, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_val_f1": best_metric, "args": vars(args),
        "model_config": {"model_name": args.model_name, "num_categories": len(labels["categories"]),
                         "num_attributes": len(labels["attributes"]), "num_polarities": len(labels["polarities"]),
                         "dropout": args.dropout, "pooling": args.pooling,
                         "trust_remote_code": args.trust_remote_code},
        "labels": labels,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(outdir / "train.log", encoding="utf-8"), logging.StreamHandler()])
    log = logging.getLogger("custom_train")

    examples = load_rows(args.data, args.text_col, args.gold_col)
    categories = sorted({c for e in examples for c,_,_ in e["gold"]})
    attributes = sorted({a for e in examples for _,a,_ in e["gold"]})
    polarities = sorted({p for e in examples for _,_,p in e["gold"]})
    if not categories or not attributes or not polarities:
        raise ValueError("No labels found. Each training example needs category, attribute and polarity annotations.")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio and test-ratio must be nonnegative and sum to less than 1.")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n = len(examples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test_examples = examples[:n_test]
    val_examples = examples[n_test:n_test+n_val]
    train_examples = examples[n_test+n_val:]
    log.info("Loaded %d examples; train=%d val=%d test=%d", n, len(train_examples), len(val_examples), len(test_examples))
    labels = {"categories": categories, "attributes": attributes, "polarities": polarities}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    train_ds = ABSADataset(train_examples, tokenizer, categories, attributes, polarities, args.max_length)
    val_ds = ABSADataset(val_examples, tokenizer, categories, attributes, polarities, args.max_length)
    test_ds = ABSADataset(test_examples, tokenizer, categories, attributes, polarities, args.max_length)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    model = CustomABSAModel(args.model_name, len(categories), len(attributes), len(polarities),
                            args.dropout, args.pooling, args.trust_remote_code).to(device)
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    start_epoch, best_f1, stale = 1, -1.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_val_f1", -1.0))
        log.info("Resumed from %s at epoch %d", args.resume, start_epoch)
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sums = defaultdict(float)
        steps = 0
        t0 = time.time()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss, parts = model.compute_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k,v in parts.items(): sums[k] += float(v)
            steps += 1
        train_loss = {k: v/max(steps,1) for k,v in sums.items()}
        val_metrics = metric_report(model, val_loader, device, categories, attributes, polarities) if val_examples else {"triple":{"f1":0.0}}
        val_f1 = val_metrics["triple"]["f1"]
        log.info("epoch=%d time=%.1fs train_loss=%s val_metrics=%s", epoch, time.time()-t0,
                 json.dumps(train_loss), json.dumps(val_metrics))
        if val_examples and val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            save_checkpoint(best_path, model, optimizer, epoch, best_f1, args, labels)
        else:
            stale += 1
        save_checkpoint(last_path, model, optimizer, epoch, best_f1, args, labels)  # after the update: --resume restores best_f1
        if val_examples and stale >= args.patience:
            log.info("Early stopping after %d stale epochs", stale)
            break
    selected = best_path if best_path.exists() else last_path
    ckpt = torch.load(selected, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if test_examples:
        test_metrics = metric_report(model, test_loader, device, categories, attributes, polarities)
        (outdir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        log.info("test metrics=%s", json.dumps(test_metrics))
    (outdir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    log.info("Best checkpoint: %s", selected)


if __name__ == "__main__":
    main()
