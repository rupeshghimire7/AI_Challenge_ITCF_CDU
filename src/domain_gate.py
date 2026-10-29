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
    5 aspect terms  the sentence mentions one of the 22 entities (see mentions below)
    6 domain        similarity margin = best cosine to a laptop label - best cosine to an other label >= MARGIN
    7 category      best cosine to a laptop label >= CATEGORY_FLOOR: it falls inside one of our categories

Mentions: the entities a sentence names explicitly, from two sources. The word lists of src.aspect_terms ("trackpad" ->
MOUSE), and word similarity: each content word is embedded on its own and compared with the 22 entity labels; a cosine
>= WORD_FLOOR is a mention ("build" -> HARDWARE 0.85; "price" 0.64 and "fast" 0.67 are not). Words the lists already
matched are not compared again ("mac" names LAPTOP, not also OS). The
post-processing uses the mentions to decide whether an opinion's entity is explicit or NULL.

Similarity: the text is tokenised by bge-small-en-v1.5 (pretrained, [CLS], normalised) and compared with the 22 laptop
entity labels (LAPTOP_LABELS) and with general other things (OTHER_LABELS). The pretrained model separates SemEval-2016
laptop from restaurant sentences better than the fine-tuned encoder. Fitted on laptop *train* sentences only:
  MARGIN, CATEGORY_FLOOR  1st percentile over the opinion sentences that reach gate 5 (each drops about 1% of them)
  WORD_FLOOR              99th percentile over the 300 most frequent train words that the word lists do not match
On the SemEval-2016 test sets the cascade keeps 98.6% of laptop opinion sentences (565/573) and rejects 59.5% of
restaurant sentences; "domain and (terms or category)" rejects more restaurant sentences than a 2-of-3 vote (55.9%) or
the margin alone (54.4%) at about the same laptop loss.

    python -m src.domain_gate        fit, then check on laptop and restaurant train / test (SemEval-2016, DATA/HF;
                                     the restaurant XML is optional)

DomainGate(device)           loads bge-small (local Hugging Face cache; downloaded once if missing), embeds the labels
DomainGate.mentions(texts)   per text: {entity: evidence} of its explicit mentions
DomainGate.run(records)      preprocess() records -> per record: {"about_laptop", "reason", "category", "margin",
                             "top_labels", "mentions", "gates"}; each gate {"gate", "passed", "detail"}
hard_rejected(gate)          a hard gate (tokens, filler, off-topic) rejected the sentence
"""

from __future__ import annotations

import re

import numpy as np
import torch

from src.aspect_terms import FILLER, entity_matches, is_filler, listed_words, named_entities, off_topic

REPO = "BAAI/bge-small-en-v1.5"
MARGIN = -0.11  # fitted by python -m src.domain_gate
CATEGORY_FLOOR = 0.43  # fitted by python -m src.domain_gate
WORD_FLOOR = 0.74  # fitted by python -m src.domain_gate

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
HARD_GATES = ("tokens", "filler", "off-topic")
_WORDPIECE = re.compile(r"[^\W\d_]")
_CONTENT = re.compile(r"[a-z][a-z'-]{2,}")


def _gate(name: str, passed: bool, detail: str) -> dict:
    return {"gate": name, "passed": passed, "detail": detail}


def hard_rejected(gate: dict) -> bool:
    return any(g["gate"] in HARD_GATES and not g["passed"] for g in gate["gates"])


def content_words(text: str) -> list[str]:
    return list(dict.fromkeys(w for w in _CONTENT.findall(text.lower()) if w not in FILLER))


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
        self._words: dict[str, tuple[str, float]] = {}  # word -> closest entity label and its cosine

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            states = self.bert(**b.to(self.device)).last_hidden_state
            out.append(torch.nn.functional.normalize(states[:, 0], dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.bert.config.hidden_size), np.float32)

    def similarity(self, texts: list[str]) -> list[dict]:
        """Per text: margin (laptop - other), the closest laptop category and its cosine, the three closest laptop
        categories, the closest other label."""
        e = self.embed(texts)
        lap, oth = e @ self.laptop.T, e @ self.other.T
        return [{"margin": round(float(lap[k].max() - oth[k].max()), 3), "category": self.laptop_names[int(lap[k].argmax())],
                 "category_sim": round(float(lap[k].max()), 3), "other_label": OTHER_LABELS[int(oth[k].argmax())],
                 "top_labels": [self.laptop_names[int(j)] for j in np.argsort(-lap[k])[:3]]} for k in range(len(texts))]

    def word_similarity(self, words: list[str]) -> dict[str, tuple[str, float]]:
        """word -> (closest entity, cosine), cached across calls."""
        new = [w for w in dict.fromkeys(words) if w not in self._words]
        if new:
            s = self.embed(new) @ self.laptop.T
            self._words |= {w: (self.laptop_names[int(s[k].argmax())], float(s[k].max())) for k, w in enumerate(new)}
        return {w: self._words[w] for w in words}

    def mentions(self, texts: list[str]) -> list[dict[str, str]]:
        words = [content_words(t) for t in texts]
        sim = self.word_similarity([w for ws in words for w in ws])
        out = []
        for t, ws in zip(texts, words):
            m = {e: f'"{w}" is in its word list' for e, w in entity_matches(t).items()}
            listed = listed_words(t)  # a word the lists already matched does not vote again by similarity
            for w in (w for w in ws if w not in listed):
                e, s = sim[w]
                if s >= WORD_FLOOR and e not in m:
                    m[e] = f'"{w}" is close to its label ({s:.2f})'
            out.append(m)
        return out

    def word_pieces(self, text: str) -> list[str]:
        return [p for p in self.tok.tokenize(text) if _WORDPIECE.search(p)]

    def run(self, records: list[dict]) -> list[dict]:
        """The cascade (top of this file) on preprocess() records. Mentions are found for every record (neighbours use
        them as review context); similarity only for the records that reach the evidence gates, in one batch."""
        mentions = self.mentions([r["text"] for r in records])
        out, ask = [], []
        for r, m in zip(records, mentions):
            pieces = self.word_pieces(r["text"])
            g = [_gate("tokens", bool(pieces), f"{len(pieces)} word pieces: {' '.join(pieces[:12])}"
                       + (" ..." if len(pieces) > 12 else ""))]
            if pieces:
                g.append(_gate("filler", not r["filler"], "greeting or filler only" if r["filler"] else "has content"))
            if pieces and not r["filler"]:
                g.append(_gate("off-topic", not r["off_topic"],
                               f"about {r['off_topic']}, not a laptop" if r["off_topic"] else "no other object is the subject"))
            out.append({"about_laptop": False, "reason": next((x["detail"] for x in g if not x["passed"]), ""),
                        "category": "", "margin": None, "top_labels": [], "mentions": m, "gates": g})
            if all(x["passed"] for x in g):
                ask.append(len(out) - 1)
        sims = self.similarity([records[i]["text"] for i in ask])
        for i, s in zip(ask, sims):
            r, v = records[i], out[i]
            laptop = "LAPTOP" in (r["entities"] if "entities" in r else named_entities(r["text"]))
            m = v["mentions"]
            evidence = [_gate("aspect terms", bool(m), ", ".join(m) or "no laptop part mentioned"),
                        _gate("domain", s["margin"] >= MARGIN, f"margin {s['margin']:+.3f}, needs {MARGIN:+.2f} or more "
                              f"(closest other topic: {s['other_label']})"),
                        _gate("category", s["category_sim"] >= CATEGORY_FLOOR, f"{s['category']} at {s['category_sim']:.3f}, "
                              f"needs {CATEGORY_FLOOR:.2f} or more")]
            terms, domain, category = (x["passed"] for x in evidence)
            v["gates"] += [_gate("laptop named", laptop, "the laptop is named" if laptop else "not named, evidence decides")]
            v["gates"] += evidence
            v |= {"about_laptop": laptop or (domain and (terms or category)), "category": s["category"],
                  "margin": s["margin"], "top_labels": s["top_labels"]}
            if not v["about_laptop"]:
                v["reason"] = (f"not about a laptop (closest topic: {s['other_label']})" if not domain else
                               "no laptop category: no part mentioned, no category close enough")
        return out


# ----------------------------------------------------------------------------------------------- fit and check
def _main() -> None:
    import collections
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

    counts = collections.Counter(w for t in sets["laptop train"].text for w in content_words(t))
    unlisted = [w for w, _ in counts.most_common() if not named_entities(w)][:300]
    word_sim = [s for _, s in gate.word_similarity(unlisted).values()]
    word_floor = math.ceil(100 * np.percentile(word_sim, 99)) / 100
    for d in sets.values():
        sim = gate.similarity(d.text.tolist())
        d["hard"] = [is_filler(t) or bool(off_topic(t)) or not gate.word_pieces(t) for t in d.text]
        d["named"] = ["LAPTOP" in named_entities(t) for t in d.text]
        d["terms"] = [bool(m) for m in gate.mentions(d.text.tolist())]
        d["margin"] = [s["margin"] for s in sim]
        d["cat_sim"] = [s["category_sim"] for s in sim]
    tr = sets["laptop train"]
    keep = tr.opinion & ~tr.hard & ~tr.named  # the sentences that reach gate 5
    margin = math.floor(100 * np.percentile(tr.margin[keep], 1)) / 100
    floor = math.floor(100 * np.percentile(tr.cat_sim[keep], 1)) / 100
    print(f"fitted on laptop train: MARGIN {margin:.2f} (file {MARGIN:.2f}), CATEGORY_FLOOR {floor:.2f} (file "
          f"{CATEGORY_FLOOR:.2f}), WORD_FLOOR {word_floor:.2f} (file {WORD_FLOOR:.2f})\n")
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
