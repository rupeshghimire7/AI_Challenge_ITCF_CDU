"""Domain gates: is a sentence about a laptop, and does it fall inside one of the 22 laptop categories? Only a sentence
that passes goes to the model; any other gets {NULL#NULL} in the post-processing.

The ABSA model was fine-tuned on laptop reviews only and reads any opinion as a laptop opinion ("restaurant is good and
has good food" -> LAPTOP#GENERAL, positive). So the system runs a cascade of gates, cheapest first:

  hard gates (any failure rejects the sentence)
    1 tokens        the bge tokenizer finds at least one word piece that is a word ("!!!", "???", "123" -> rejected)
    2 filler        greetings / thanks only ("Hello.", "Thanks everyone!")
    3 off-topic     a non-laptop object is what the sentence is about ("The pizza was cold", "I love my new phone")
  strong pass
    4 laptop named  the laptop itself is named (laptop, computer, a brand, ...): about the laptop, no vote needed
  evidence (domain must pass, and aspect terms or category)
    5 aspect terms  the sentence names one of the 22 entities (screen, battery, keyboard, ...; src.aspect_terms)
    6 domain        similarity margin = best cosine to a laptop label - best cosine to an other label >= MARGIN
    7 category      best cosine to a laptop label >= CATEGORY_FLOOR: it falls inside one of our categories

Similarity: the sentence is tokenised by bge-small-en-v1.5 (pretrained, [CLS], normalised) and compared with the 22
laptop entity labels (LAPTOP_LABELS) and with general other things (OTHER_LABELS). The pretrained model separates
SemEval-2016 laptop from restaurant sentences better than the fine-tuned encoder. MARGIN and CATEGORY_FLOOR are fitted on
the laptop *train* sentences with an opinion that reach gate 5 (1st percentile, so each drops about 1% of them). On
the SemEval-2016 test sets the cascade keeps 98.6% of laptop opinion sentences (565/573) and rejects 59.5% of restaurant
sentences; the rule "domain and (terms or category)" rejects more restaurant sentences than a 2-of-3 vote (55.9%) or the
margin alone (54.4%) at about the same laptop loss.

    python -m src.domain_gate        fit, then check on laptop and restaurant train / test (SemEval-2016, DATA/HF;
                                     the restaurant XML is optional)

DomainGate(device)       loads bge-small (local Hugging Face cache; downloaded once if missing) and embeds the labels
DomainGate.run(records)  preprocess() records -> per record: {"about_laptop", "reason", "category", "margin", "gates"};
                         each gate {"gate", "passed", "detail"}
"""

from __future__ import annotations

import re

import numpy as np
import torch

from src.aspect_terms import is_filler, named_entities, off_topic

REPO = "BAAI/bge-small-en-v1.5"
MARGIN = -0.11  # fitted by python -m src.domain_gate
CATEGORY_FLOOR = 0.43  # fitted by python -m src.domain_gate

LAPTOP_LABELS = {
    "LAPTOP": "laptop computer", "DISPLAY": "laptop screen and display", "KEYBOARD": "keyboard and typing",
    "MOUSE": "touchpad, trackpad and mouse", "MOTHERBOARD": "motherboard", "CPU": "processor and cpu speed",
    "FANS_COOLING": "cooling fan and heat", "PORTS": "usb and hdmi ports", "MEMORY": "ram memory",
    "POWER_SUPPLY": "power adapter and charger", "OPTICAL_DRIVES": "dvd and cd drive", "BATTERY": "battery life",
    "GRAPHICS": "graphics card", "HARD_DISC": "hard drive and storage", "MULTIMEDIA_DEVICES": "speakers, webcam and sound",
    "HARDWARE": "hardware and build", "SOFTWARE": "software and programs", "OS": "operating system, windows or mac os",
    "WARRANTY": "warranty", "SHIPPING": "shipping and delivery", "SUPPORT": "customer service and tech support",
    "COMPANY": "brand and manufacturer",
}
OTHER_LABELS = ["restaurant", "food and drinks", "a meal and its taste", "waiter, waitress and restaurant staff",
                "hotel and travel", "flights and airports", "car and driving", "mobile phone", "television",
                "movie, film or show", "music and songs", "book or novel", "clothes and shoes", "weather", "sports",
                "health and medicine", "pets and animals", "family and friends", "home, furniture and kitchen",
                "money and banking", "school and work", "politics and news"]
GATES = ("tokens", "filler", "off-topic", "laptop named", "aspect terms", "domain", "category")
_WORDPIECE = re.compile(r"[^\W\d_]")


def _gate(name: str, passed: bool | None, detail: str) -> dict:
    """passed: True / False, or None when an earlier gate already decided."""
    return {"gate": name, "passed": passed, "detail": detail}


class DomainGate:
    def __init__(self, device: str | torch.device = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()
        try:
            self.tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
            self.bert = AutoModel.from_pretrained(REPO, local_files_only=True, add_pooling_layer=False)
        except OSError:  # not in the local cache yet: download once (~130 MB)
            self.tok = AutoTokenizer.from_pretrained(REPO)
            self.bert = AutoModel.from_pretrained(REPO, add_pooling_layer=False)
        self.device = torch.device(device)
        self.bert.to(self.device).eval()
        self.laptop_names, self.laptop = list(LAPTOP_LABELS), self.embed(list(LAPTOP_LABELS.values()))
        self.other = self.embed(OTHER_LABELS)

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            states = self.bert(**b.to(self.device)).last_hidden_state
            out.append(torch.nn.functional.normalize(states[:, 0], dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.bert.config.hidden_size), np.float32)

    def similarity(self, texts: list[str]) -> list[dict]:
        """Per text: margin (laptop - other), the closest laptop category and its cosine, the closest other label."""
        e = self.embed(texts)
        lap, oth = e @ self.laptop.T, e @ self.other.T
        return [{"margin": round(float(lap[k].max() - oth[k].max()), 3), "category": self.laptop_names[int(lap[k].argmax())],
                 "category_sim": round(float(lap[k].max()), 3), "other_label": OTHER_LABELS[int(oth[k].argmax())]}
                for k in range(len(texts))]

    def word_pieces(self, text: str) -> list[str]:
        return [p for p in self.tok.tokenize(text) if _WORDPIECE.search(p)]

    def run(self, records: list[dict]) -> list[dict]:
        """The cascade (top of this file) on preprocess() records. Similarity is computed only for the records that
        reach the votes, in one batch."""
        out, ask = [], []
        for r in records:
            pieces = self.word_pieces(r["text"])
            g = [_gate("tokens", bool(pieces), f"{len(pieces)} word pieces: {' '.join(pieces[:12])}"
                       + (" ..." if len(pieces) > 12 else ""))]
            if pieces:
                g.append(_gate("filler", not r["filler"], "greeting / filler only" if r["filler"] else "has content"))
            if pieces and not r["filler"]:
                g.append(_gate("off-topic", not r["off_topic"],
                               f"off-topic: about {r['off_topic']}" if r["off_topic"] else "no other object is the subject"))
            out.append({"about_laptop": False, "reason": next((x["detail"] for x in g if not x["passed"]), ""),
                        "category": "", "margin": None, "gates": g})
            if all(x["passed"] for x in g):
                ask.append(len(out) - 1)
        sims = self.similarity([records[i]["text"] for i in ask])
        for i, s in zip(ask, sims):
            r, v = records[i], out[i]
            named = r["entities"] if "entities" in r else named_entities(r["text"])
            laptop = "LAPTOP" in named
            votes = [_gate("aspect terms", bool(named), ", ".join(named) or "no laptop part named"),
                     _gate("domain", s["margin"] >= MARGIN, f"margin {s['margin']:+.3f} (min {MARGIN:+.2f}); closest "
                           f"other: {s['other_label']}"),
                     _gate("category", s["category_sim"] >= CATEGORY_FLOOR, f"{s['category']} at {s['category_sim']:.3f} "
                           f"(min {CATEGORY_FLOOR:.2f})")]
            terms, domain, category = (x["passed"] for x in votes)
            v["gates"] += [_gate("laptop named", laptop, "the laptop is named" if laptop else "not named, evidence decides")]
            v["gates"] += votes
            v |= {"about_laptop": laptop or (domain and (terms or category)), "category": s["category"],
                  "margin": s["margin"]}
            if not v["about_laptop"]:
                v["reason"] = (f"not about a laptop (closest: {s['other_label']})" if not domain else
                               "no laptop category: no part named, no category close enough")
        return out


# ----------------------------------------------------------------------------------------------- fit and check
def _main() -> None:
    import math
    import xml.etree.ElementTree as ET

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from src.config import DATA_DIR, FINAL_DIR
    from src.preprocess import normalise

    REST_FILES = {"train": "ABSA16_Restaurants_Train_SB1_v2.xml", "test": "EN_REST_SB1_TEST.xml.gold"}

    def laptops(split: str) -> pd.DataFrame:
        df = pd.read_csv(FINAL_DIR / f"{split}_final.csv", dtype=str, keep_default_na=False)
        return df.groupby("sentence_id", sort=False).agg(text=("text", "first"), opinion=("category", lambda c: any(c != "")))

    def restaurants(split: str) -> pd.DataFrame:
        f = DATA_DIR / "HF" / REST_FILES[split]
        rows = [(normalise(s.findtext("text") or ""), len(s.findall("Opinions/Opinion")) > 0)
                for s in ET.parse(f).getroot().iter("sentence")]
        return pd.DataFrame(rows, columns=["text", "opinion"])

    gate = DomainGate()
    sets = {f"laptop {s}": laptops(s) for s in ("train", "test")}
    for split, f in REST_FILES.items():  # the out-of-domain check needs the restaurant XML; skipped without it
        if (DATA_DIR / "HF" / f).exists():
            sets[f"restaurant {split}"] = restaurants(split)
        else:
            print(f"skipped restaurant {split}: DATA/HF/{f} not found")
    for d in sets.values():
        rec = [{"text": t, "filler": is_filler(t), "off_topic": off_topic(t), "entities": named_entities(t)} for t in d.text]
        sim = gate.similarity(d.text.tolist())
        d["hard"] = [is_filler(t) or bool(off_topic(t)) or not gate.word_pieces(t) for t in d.text]
        d["named"] = ["LAPTOP" in r["entities"] for r in rec]
        d["terms"] = [bool(r["entities"]) for r in rec]
        d["margin"] = [s["margin"] for s in sim]
        d["cat_sim"] = [s["category_sim"] for s in sim]
    tr = sets["laptop train"]
    keep = tr.opinion & ~tr.hard & ~tr.named  # the sentences that reach gate 5
    margin = math.floor(100 * np.percentile(tr.margin[keep], 1)) / 100
    floor = math.floor(100 * np.percentile(tr.cat_sim[keep], 1)) / 100
    print(f"fitted on laptop train opinion sentences reaching the votes: MARGIN {margin:.2f} (file {MARGIN:.2f}), "
          f"CATEGORY_FLOOR {floor:.2f} (file {CATEGORY_FLOOR:.2f})\n")
    print(f"{'set':<16}{'sentences':>10}{'rejected':>10}{'opinion sentences rejected':>28}")
    for name, d in sets.items():
        off = d.hard | (~d.named & ~((d.margin >= margin) & (d.terms | (d.cat_sim >= floor))))
        print(f"{name:<16}{len(d):>10}{off.mean():>10.1%}{f'{int((off & d.opinion).sum())}/{int(d.opinion.sum())}':>28}")
    if "restaurant test" not in sets:
        return
    lt, rt = sets["laptop test"], sets["restaurant test"]
    y = np.r_[np.ones(len(lt)), np.zeros(len(rt))]
    print(f"\ntest AUC laptop vs restaurant: margin {roc_auc_score(y, np.r_[lt.margin, rt.margin]):.3f}, "
          f"category similarity {roc_auc_score(y, np.r_[lt.cat_sim, rt.cat_sim]):.3f}")


if __name__ == "__main__":
    _main()
