"""Load a custom three-head ABSA checkpoint and run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from custom_model import CustomABSAModel


def load_model(checkpoint_path, device="auto"):
    device = _device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    labels = ckpt["labels"]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=cfg.get("trust_remote_code", True)
    )
    model = CustomABSAModel(
        model_name=cfg["model_name"],
        num_categories=cfg["num_categories"],
        num_attributes=cfg["num_attributes"],
        num_polarities=cfg["num_polarities"],
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "mean"),
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, tokenizer, labels, device


def _device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(text, model, tokenizer, labels, device, max_length=128, threshold=0.5):
    batch = tokenizer(text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    batch = {k:v.to(device) for k,v in batch.items() if k in {"input_ids", "attention_mask"}}
    out = model(batch["input_ids"], batch["attention_mask"])
    cat_p = out["category_logits"].sigmoid()[0]
    attr_p = out["attribute_logits"].sigmoid()[0]
    pol_p = out["polarity_logits"].softmax(-1)[0]
    categories = labels["categories"]
    attributes = labels["attributes"]
    polarities = labels["polarities"]
    cat_ids = (cat_p >= threshold).nonzero().flatten().tolist()
    attr_ids = (attr_p >= threshold).nonzero().flatten().tolist()
    opinions = []
    for ci in cat_ids:
        for ai in attr_ids:
            pi = int(pol_p[ci, ai].argmax())
            opinions.append({
                "category": categories[ci],
                "attribute": attributes[ai],
                "polarity": polarities[pi],
                "category_score": float(cat_p[ci]),
                "attribute_score": float(attr_p[ai]),
                "polarity_score": float(pol_p[ci, ai, pi]),
            })
    return {"text": text, "opinions": opinions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", default="")
    p.add_argument("--input-file", default="", help="optional text file, one sentence per line")
    p.add_argument("--output-file", default="")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()
    model, tokenizer, labels, device = load_model(args.checkpoint, args.device)
    texts = [args.text] if args.text else []
    if args.input_file:
        texts += [x.strip() for x in Path(args.input_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not texts:
        p.error("Provide --text or --input-file.")
    results = [predict(t, model, tokenizer, labels, device, args.max_length, args.threshold) for t in texts]
    rendered = "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    if args.output_file:
        Path(args.output_file).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
