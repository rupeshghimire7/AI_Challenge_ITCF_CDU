"""Exploratory data analysis of the prepared corpora (plan D4, D5.1).

Sets, named as in plan D1: S1 = SemEval-2016 laptops train, T1 = SemEval test, S4 = Laptop-ACOS train, T2 = ACOS test.
Numbers come from the cleaned files (*_merged_cleaned.csv) unless a section says otherwise. An opinion is a
(sentence, category, polarity) tuple; a category count is the number of sentences containing the category.

Sections: 1 overview, 2 label space, 3 polarity, 4 opinions per sentence, 5 text and token length (sets max_len),
6 review structure and context, 7 category co-occurrence, 8 out-of-scope sentences, 9 SemEval vs ACOS annotation,
10 train -> test shift, 11 nuances (D5.1 slices, src.nuance_checker).

Outputs: DATA/final/reports/eda.txt (the printed report), reports/figures/*.png, reports/tables/*.csv
Run:     python scripts/eda.py     (after clean_data.py). Section 5 downloads two tokenizer.json files from the Hugging
         Face Hub on first run; if that fails, the section is skipped.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTRIBUTES, ENTITIES, FINAL_DIR, OUT_OF_SCOPE, POLARITIES, REPORT_DIR, ROOT  # noqa: E402
from src.data_maker import load_long, to_sentences  # noqa: E402
from src.nuance_checker import NUANCES  # noqa: E402
from src.preprocess import match_key  # noqa: E402

FINAL = FINAL_DIR
FIG, TAB = REPORT_DIR / "figures", REPORT_DIR / "tables"
SETS = {"S1": ("train", "semeval16"), "T1": ("test", "semeval16"), "S4": ("train", "acos"), "T2": ("test", "acos")}
NAMES = {"S1": "SemEval train (S1)", "T1": "SemEval test (T1)", "S4": "ACOS train (S4)", "T2": "ACOS test (T2)"}
TOKENIZERS = {"bge-small-en-v1.5 (model B)": "BAAI/bge-small-en-v1.5", "LFM2.5-Encoder-230M (model A)": "LiquidAI/LFM2.5-Encoder-230M"}

# Chart palette: the dataviz reference palette (validated: adjacent CVD dE >= 9.1 in this slot order). A set keeps its
# colour in every figure. Polarity is diverging: red (negative) / grey midpoint (neutral) / blue (positive).
SURFACE, INK, INK2, RULE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c2", "#e9e8e4"
SET_COLOR = {"S1": "#2a78d6", "T1": "#eb6834", "S4": "#1baf7a", "T2": "#eda100"}
POL_COLOR = {"negative": "#e34948", "neutral": "#9c9b96", "positive": "#2a78d6"}
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlecolor": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False})



# ----------------------------------------------------------------------------------------------- helpers
def pct(a: float, b: float, d: int = 1) -> str:
    return f"{100 * a / b:.{d}f}%" if b else "–"


def table(headers, rows) -> list[str]:
    """Plain-text table: numeric columns right-aligned, text left-aligned."""
    cells = [list(map(str, headers)), *([str(c) for c in r] for r in rows)]
    width = [max(map(len, col)) for col in zip(*cells)]
    right = [all(re.fullmatch(r"-?[\d.,]+%?|–", c) for c in col[1:]) for col in zip(*cells)]
    line = lambda r: "  ".join(c.rjust(w) if a else c.ljust(w) for c, w, a in zip(r, width, right)).rstrip()  # noqa: E731
    return [line(cells[0]), line(["-" * w for w in width]), *map(line, cells[1:]), ""]


def opinions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.aspect_category != ""]


def cat_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Unique (sentence, category): the unit of the official Slot1 evaluation."""
    return opinions(df).drop_duplicates(["sentence_id", "aspect_category"])


def quantiles(x, qs=(50, 95, 99)) -> list:
    return [int(np.percentile(x, q)) for q in qs] + [int(np.max(x))]


def style(ax, grid: str | None = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figure: reports/figures/{name}"


def stacked_share(ax, labels: list[str], shares: dict[str, list[float]], colors: dict[str, str]) -> None:
    """Horizontal 100% stacked bars, a surface-coloured gap between segments, % labels where the segment fits."""
    left = np.zeros(len(labels))
    y = np.arange(len(labels))[::-1]
    for key, vals in shares.items():
        vals = np.asarray(vals)
        ax.barh(y, vals, left=left, color=colors[key], height=0.62, edgecolor=SURFACE, linewidth=1.5, label=key)
        for yi, l, v in zip(y, left, vals):
            if v >= 7:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if key != "neutral" else INK)
        left += vals
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style(ax)


# ----------------------------------------------------------------------------------------------- sections
def overview(long, sents, merged_sents) -> list[str]:
    rows, impact = [], []
    for k in SETS:
        d, s = long[k], sents[k]
        n, has = len(s), s.n_opinions > 0
        op = opinions(d)
        rows.append([NAMES[k], s.review_id[s.review_id != ""].nunique() or "–", n, int(has.sum()), pct((~has).sum(), n),
                     pct(s.out_of_scope.sum(), n), len(op), len(cat_rows(d)), f"{len(op) / max(has.sum(), 1):.2f}",
                     pct((s.n_opinions >= 2).sum(), n), op.aspect_category.nunique(), op.entity.nunique(), op.attribute.nunique()])
        m = merged_sents[k]
        impact.append([NAMES[k], len(m), n, len(m) - n, int(m.opinions.map(len).sum()), len(op)])
    out = ["1. OVERVIEW"]
    out += table(["set", "reviews", "sentences", "with opinion", "without opinion", "out of scope", "opinions", "unique sentence x category",
               "opinions per opinionated sentence", "multi-opinion sentences", "categories", "entities", "attributes"], rows)
    out += ["cleaning impact (*_merged.csv -> *_merged_cleaned.csv)"]
    out += table(["set", "sentences merged", "sentences cleaned", "removed", "opinions merged", "opinions cleaned"], impact)
    return out


def label_space(long) -> tuple[list[str], dict[str, Counter]]:
    counts = {k: Counter(cat_rows(long[k]).aspect_category) for k in SETS}
    cats = sorted(set().union(*counts.values()), key=lambda c: (-counts["S1"][c], -counts["S4"][c], c))
    pd.DataFrame({"category": cats, **{k: [counts[k][c] for c in cats] for k in SETS}}).to_csv(TAB / "category_counts.csv", index=False)
    s1 = counts["S1"]
    total = sum(s1.values())
    top = [c for c, _ in s1.most_common()]
    out = ["2. LABEL SPACE", f"inventory: {len(ENTITIES)} entities x {len(ATTRIBUTES)} attributes = {len(ENTITIES) * len(ATTRIBUTES)} possible categories"]
    rows = []
    for k in SETS:
        c = counts[k]
        rows.append([NAMES[k], len(c), sum(v < 5 for v in c.values()), sum(v < 10 for v in c.values()),
                     pct(sum(v for _, v in c.most_common(5)), sum(c.values())), pct(sum(v for _, v in c.most_common(10)), sum(c.values())),
                     c.most_common(1)[0][0] + f" ({pct(c.most_common(1)[0][1], sum(c.values()))})"])
    out += table(["set", "categories", "with < 5 sentences", "with < 10 sentences", "top-5 share", "top-10 share", "most frequent"], rows)
    merged_train = counts["S1"] + counts["S4"]
    out += [f"S1 + S4 categories: {len(merged_train)} ({len(set(counts['S4']) - set(s1))} only in S4)",
            f"S1 LAPTOP#GENERAL share: {pct(s1['LAPTOP#GENERAL'], total)}; S1 top-10 share: {pct(sum(s1[c] for c in top[:10]), total)}; "
            f"S1 categories with < 5 sentences: {sum(v < 5 for v in s1.values())} of {len(s1)}", "",
            "top 20 categories (all: reports/tables/category_counts.csv)"]
    out += table(["category", *SETS], [[c, *(counts[k][c] for k in SETS)] for c in cats[:20]])

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k in ("S1", "S4"):
        v = sorted(counts[k].values(), reverse=True)
        ax.plot(range(1, len(v) + 1), v, color=SET_COLOR[k], linewidth=2, label=f"{NAMES[k]}: {len(v)} categories")
    ax.axhline(5, color=INK2, linewidth=0.8)
    ax.text(len(merged_train) * 0.97, 5.6, "5 sentences", ha="right", va="bottom", color=INK2, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("category rank")
    ax.set_ylabel("sentences (log scale)")
    ax.set_title("Category frequency is a long tail")
    ax.legend(loc="upper right", fontsize=8)
    style(ax, "y")
    out += [save(fig, "category_long_tail.png"), ""]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    show = top[:25]
    y = np.arange(len(show))[::-1]
    for i, k in enumerate(("S1", "S4")):
        share = [100 * counts[k][c] / sum(counts[k].values()) for c in show]
        ax.barh(y + (0.2 if i == 0 else -0.2), share, height=0.38, color=SET_COLOR[k], label=NAMES[k], edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, show, fontsize=7.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("share of the set's sentence x category pairs")
    ax.set_title("Top 25 S1 categories: how often each corpus uses them")
    ax.legend(loc="lower right", fontsize=8)
    style(ax)
    out += [save(fig, "top_categories.png"), ""]

    ent_rows = [[e, *(f"{n} ({pct(n, sum(counts[k].values()))})" for k in SETS for n in [sum(v for c, v in counts[k].items() if c.startswith(e + "#"))])]
                for e in sorted(ENTITIES, key=lambda e: -sum(v for c, v in s1.items() if c.startswith(e + "#")))]
    att_rows = [[a, *(f"{n} ({pct(n, sum(counts[k].values()))})" for k in SETS for n in [sum(v for c, v in counts[k].items() if c.endswith("#" + a))])]
                for a in sorted(ATTRIBUTES, key=lambda a: -sum(v for c, v in s1.items() if c.endswith("#" + a)))]
    out += ["entities (sentences, share of the set)"] + table(["entity", *SETS], ent_rows)
    out += ["attributes"] + table(["attribute", *SETS], att_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True)
    ents = [r[0] for r in ent_rows]
    vmax = max(100 * v / sum(counts[k].values()) for k in ("S1", "S4") for v in counts[k].values())
    for ax, k in zip(axes, ("S1", "S4")):
        tot = sum(counts[k].values())
        grid = np.array([[counts[k][f"{e}#{a}"] for a in ATTRIBUTES] for e in ents], dtype=float)
        share = np.ma.masked_equal(100 * grid / tot, 0)
        im = ax.imshow(share, cmap=RAMP, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), n in np.ndenumerate(grid):
            if n:
                ax.text(j, i, int(n), ha="center", va="center", fontsize=6.5, color="white" if share[i, j] > vmax * 0.45 else INK)
        ax.set_xticks(range(len(ATTRIBUTES)), [a.replace("_", " ").lower() for a in ATTRIBUTES], rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(ents)), ents, fontsize=7.5)
        ax.set_xticks(np.arange(-0.5, len(ATTRIBUTES)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ents)), minor=True)
        ax.grid(which="minor", color=GRID, linewidth=0.6)
        ax.tick_params(which="both", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{NAMES[k]}: {int(grid.astype(bool).sum())} of 198 cells used", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="share of the set's sentence x category pairs (%)")
    fig.suptitle("Entity x attribute: counts in cells, colour = share", x=0.06, ha="left", fontweight="bold", color=INK)
    out += [save(fig, "entity_attribute_heatmap.png"), ""]
    return out, counts


def polarity(long, sents) -> list[str]:
    out = ["3. POLARITY"]
    rows, shares = [], {p: [] for p in POLARITIES}
    for k in SETS:
        c = Counter(opinions(long[k]).polarity)
        n = sum(c.values())
        rows.append([NAMES[k], n, *(f"{c[p]} ({pct(c[p], n)})" for p in POLARITIES)])
        for p in POLARITIES:
            shares[p].append(100 * c[p] / n)
    out += table(["set", "opinions", *POLARITIES], rows)
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    stacked_share(ax, [NAMES[k] for k in SETS], shares, POL_COLOR)
    ax.set_title("Polarity of opinions")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    out += [save(fig, "polarity_by_set.png"), ""]

    s1 = opinions(long["S1"])
    by_ent = s1.groupby("entity").polarity.value_counts().unstack(fill_value=0).reindex(columns=list(POLARITIES), fill_value=0)
    by_ent["n"] = by_ent.sum(axis=1)
    by_ent = by_ent.sort_values("n", ascending=False)
    out += ["S1 polarity by entity"]
    out += table(["entity", "n", *POLARITIES], [[e, int(r.n), *(pct(r[p], r.n) for p in POLARITIES)] for e, r in by_ent.iterrows()])
    big = by_ent[by_ent.n >= 15].assign(neg=lambda d: d.negative / d.n).sort_values("neg", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(big) + 1.2))
    stacked_share(ax, [f"{e} (n={int(r.n)})" for e, r in big.iterrows()], {p: list(100 * big[p] / big.n) for p in POLARITIES}, POL_COLOR)
    ax.set_title("S1 polarity by entity (entities with n >= 15), most negative first")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8)
    out += [save(fig, "polarity_by_entity.png"), ""]

    by_att = s1.groupby("attribute").polarity.value_counts().unstack(fill_value=0).reindex(columns=list(POLARITIES), fill_value=0)
    by_att["n"] = by_att.sum(axis=1)
    out += ["S1 polarity by attribute"]
    out += table(["attribute", "n", *POLARITIES], [[a, int(r.n), *(pct(r[p], r.n) for p in POLARITIES)] for a, r in by_att.sort_values("n", ascending=False).iterrows()])

    rows = []
    for k in SETS:
        s = sents[k]
        multi = s[s.n_opinions >= 2]
        mixed = multi.opinions.map(lambda o: len({p for _, p in o}) > 1)
        conflict = long[k].drop_duplicates(["sentence_id", "aspect_category", "polarity"])
        rows.append([NAMES[k], len(multi), f"{int(mixed.sum())} ({pct(mixed.sum(), len(multi))})", int(conflict[conflict.polarity_conflict].sentence_id.nunique())])
    out += ["mixed polarity inside a sentence"]
    out += table(["set", "multi-opinion sentences", "with >= 2 polarities", "sentences with polarity_conflict (one category, two polarities)"], rows)
    return out


def per_sentence(sents) -> list[str]:
    bins = ["0", "1", "2", "3", "4", "5+"]
    dist = {k: Counter(min(n, 5) for n in sents[k].n_opinions) for k in SETS}
    out = ["4. OPINIONS PER SENTENCE"]
    out += table(["set", *bins, "mean"], [[NAMES[k], *(pct(dist[k][i], len(sents[k])) for i in range(6)), f"{sents[k].n_opinions.mean():.2f}"] for k in SETS])
    fig, ax = plt.subplots(figsize=(7.5, 3))
    x = np.arange(len(bins))
    for i, k in enumerate(SETS):
        ax.bar(x + (i - 1.5) * 0.2, [100 * dist[k][j] / len(sents[k]) for j in range(6)], width=0.19, color=SET_COLOR[k], label=NAMES[k],
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, bins)
    ax.set_xlabel("opinions in the sentence")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("Opinions per sentence: ACOS has almost no opinion-less sentences")
    ax.legend(fontsize=8)
    style(ax, "y")
    out += [save(fig, "opinions_per_sentence.png"), ""]
    return out


def lengths(sents) -> list[str]:
    out = ["5. TEXT AND TOKEN LENGTH"]
    out += table(["set", "words p50", "p95", "p99", "max", "characters p50", "p95", "p99", "max"],
              [[NAMES[k], *quantiles(sents[k].text.str.split().map(len)), *quantiles(sents[k].text.str.len())] for k in SETS])
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from tokenizers import Tokenizer

        def tokenizer_file(repo: str) -> str:  # local cache first; the Hub only when the file was never downloaded
            try:
                return hf_hub_download(repo, "tokenizer.json", local_files_only=True)
            except LocalEntryNotFoundError:
                return hf_hub_download(repo, "tokenizer.json")

        toks = {name: Tokenizer.from_file(tokenizer_file(repo)) for name, repo in TOKENIZERS.items()}
    except Exception as e:  # offline, hub unavailable: the rest of the report still runs
        return out + [f"tokenizer lengths skipped ({type(e).__name__}: {e})", ""]
    rows, ctx_rows, hist = [], [], {}
    for name, tok in toks.items():
        tok.no_truncation()
        tok.no_padding()
        n_special = len(tok.encode("x").ids) - len(tok.encode("x", add_special_tokens=False).ids)

        def n(texts, special=True):
            return np.array([len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=special)])

        for k in SETS:
            L = n(sents[k].text)
            hist[name, k] = L
            rows.append([name, NAMES[k], *quantiles(L), pct((L > 64).sum(), len(L), 2), pct((L > 128).sum(), len(L), 2)])
        train = np.concatenate([hist[name, "S1"], hist[name, "S4"]])
        rows.append([name, "S1 + S4", *quantiles(train), pct((train > 64).sum(), len(train), 2), pct((train > 128).sum(), len(train), 2)])
        for k in ("S1", "T1"):  # plan L4 input: [prev] [SEP] target [SEP] [next]
            s = sents[k]
            L = n(s.prev_text, False) + n(s.text, False) + n(s.next_text, False) + n_special + 2
            ctx_rows.append([name, NAMES[k], *quantiles(L), pct((L > 128).sum(), len(L), 2), pct((L > 256).sum(), len(L), 2)])
    out += ["tokens per sentence, special tokens included"]
    out += table(["tokenizer", "set", "p50", "p95", "p99", "max", "> 64", "> 128"], rows)
    out += ["tokens with review context: [prev] [SEP] target [SEP] [next] (plan L4; SemEval only)"]
    out += table(["tokenizer", "set", "p50", "p95", "p99", "max", "> 128", "> 256"], ctx_rows)
    rec = {name: int(math.ceil(np.percentile(hist[name, "S1"], 99) / 8) * 8) for name in toks}
    out += ["max_len (S1 p99 rounded up to a multiple of 8): " + "; ".join(f"{k}: {v}" for k, v in rec.items()), ""]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, name in zip(axes, toks):
        bins = np.arange(0, 100, 2)
        for k in ("S1", "S4"):
            ax.hist(np.minimum(hist[name, k], 98), bins=bins, histtype="step", linewidth=2, color=SET_COLOR[k], label=NAMES[k],
                    weights=np.full(len(hist[name, k]), 100 / len(hist[name, k])))
        p99 = np.percentile(hist[name, "S1"], 99)
        ax.axvline(p99, color=INK2, linewidth=1)
        ax.text(p99 + 1, ax.get_ylim()[1] * 0.92, f"S1 p99 = {p99:.0f}", color=INK2, fontsize=7.5, va="top")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("tokens per sentence (values >= 98 in the last bin)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        style(ax, "y")
    axes[0].legend(fontsize=8)
    out += [save(fig, "token_lengths.png"), ""]
    return out


def structure(sents) -> list[str]:
    out = ["6. REVIEW STRUCTURE AND CONTEXT (SemEval)"]
    rows = []
    for k in ("S1", "T1"):
        s = sents[k]
        per = s.drop_duplicates("review_id").review_n_sentences
        rows.append([NAMES[k], s.review_id.nunique(), f"{per.mean():.1f}", int(per.median()), int(per.max())])
    out += table(["set", "reviews", "sentences per review: mean", "median", "max"], rows)

    s = sents["S1"].copy()
    s["position"] = np.select([s.review_n_sentences == 1, s.sentence_index == 0, s.sentence_index == s.review_n_sentences - 1],
                              ["only sentence", "first", "last"], "middle")
    s["general"] = s.opinions.map(lambda o: any(c == "LAPTOP#GENERAL" for c, _ in o))
    order = ["first", "middle", "last", "only sentence"]
    g = s.groupby("position").agg(n=("text", "size"), has=("n_opinions", lambda x: (x > 0).mean()), general=("general", "mean"),
                                  oos=("out_of_scope", "mean")).reindex(order)
    edge = s.position.isin(["first", "last"])
    out += [f"S1 first + last sentences: {pct(edge.mean(), 1)} of sentences, {pct(s.general[edge].sum(), s.general.sum())} of LAPTOP#GENERAL opinions",
            "S1 by position in the review"]
    out += table(["position", "sentences", "has an opinion", "has LAPTOP#GENERAL", "out of scope"],
              [[p, int(r.n), pct(r.has, 1), pct(r.general, 1), pct(r.oos, 1)] for p, r in g.iterrows()])
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x = np.arange(len(order))
    for i, (col, label, color) in enumerate([("has", "has an opinion", SET_COLOR["S1"]), ("general", "has LAPTOP#GENERAL", SET_COLOR["T1"])]):
        ax.bar(x + (i - 0.5) * 0.36, 100 * g[col], width=0.34, color=color, label=label, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [f"{p}\n(n={int(g.n[p])})" for p in order])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("S1: share of sentences by position in the review")
    ax.legend(fontsize=8)
    style(ax, "y")
    out += [save(fig, "position_effect.png"), ""]

    # Entity carry-over: does an opinionated sentence share an entity with the previous opinionated sentence of its review?
    pairs = []
    for _, r in s[s.n_opinions > 0].sort_values(["review_id", "sentence_index"]).groupby("review_id", sort=False):
        ents = [{c.split("#")[0] for c, _ in o} for o in r.opinions]
        pairs += list(zip(ents, ents[1:]))
    rng = np.random.default_rng(2026)
    shuffled = [pairs[i][0] for i in rng.permutation(len(pairs))]
    rate = lambda ps, drop=frozenset(): np.mean([bool((a - drop) & (b - drop)) for a, b in ps])  # noqa: E731
    base = list(zip(shuffled, [b for _, b in pairs]))
    no_lap = frozenset({"LAPTOP"})
    out += ["S1 entity carry-over: opinionated sentence shares an entity with the previous opinionated sentence of its review"]
    out += table(["entities", "same review", "random previous sentence"],
                 [["all", pct(rate(pairs), 1), pct(rate(base), 1)], ["excluding LAPTOP", pct(rate(pairs, no_lap), 1), pct(rate(base, no_lap), 1)]])
    return out


def cooccurrence(sents) -> list[str]:
    s = sents["S1"]
    sets_ = [sorted({c for c, _ in o}) for o in s.opinions if o]
    n = len(sets_)
    single = Counter(c for cs in sets_ for c in cs)
    pair = Counter(p for cs in sets_ for p in itertools.combinations(cs, 2))
    rows = [(a, b, v, math.log2(v * n / (single[a] * single[b]))) for (a, b), v in pair.items()]
    pmi = pd.DataFrame(rows, columns=["category_a", "category_b", "sentences", "pmi"]).sort_values("sentences", ascending=False)
    pmi.to_csv(TAB / "cooccurrence_pmi_s1.csv", index=False)
    out = ["7. CATEGORY CO-OCCURRENCE (S1)", f"opinionated sentences: {n}; with >= 2 categories: {sum(len(c) >= 2 for c in sets_)}",
           "most frequent pairs (all: reports/tables/cooccurrence_pmi_s1.csv)"]
    out += table(["category a", "category b", "sentences", "PMI"], [[a, b, v, f"{p:.2f}"] for a, b, v, p in pmi.head(12).itertuples(index=False)])
    out += ["highest PMI, pairs in >= 5 sentences"]
    strong = pmi[pmi.sentences >= 5].sort_values("pmi", ascending=False).head(10)
    out += table(["category a", "category b", "sentences", "PMI"], [[a, b, v, f"{p:.2f}"] for a, b, v, p in strong.itertuples(index=False)])
    return out


def out_of_scope(sents) -> list[str]:
    comp = NUANCES["comparison"]
    out = ["8. OUT-OF-SCOPE SENTENCES"]
    rows = []
    for k in SETS:
        s = sents[k]
        groups = {"out of scope": s[s.out_of_scope], "in scope, no opinion": s[~s.out_of_scope & (s.n_opinions == 0)],
                  "in scope, opinionated": s[s.n_opinions > 0]}
        rows.append([NAMES[k], f"{int(s.out_of_scope.sum())} ({pct(s.out_of_scope.sum(), len(s))})",
                     *(pct(g.text.map(lambda t: bool(comp.search(t))).sum(), len(g)) for g in groups.values())])
    out += table(["set", "out-of-scope sentences", "comparison nuance: out of scope", "in scope, no opinion", "in scope, opinionated"], rows)
    s1 = sents["S1"]
    oos = s1[s1.out_of_scope].sample(8, random_state=2026)
    out += [f"out-of-scope share of opinion-less sentences: S1 {pct(s1.out_of_scope.sum(), (s1.n_opinions == 0).sum())}, "
            f"T1 {pct(sents['T1'].out_of_scope.sum(), (sents['T1'].n_opinions == 0).sum())}", "random S1 out-of-scope sentences:"]
    out += [f"  {t}" for t in oos.text] + [""]
    return out


def annotation_differences(long, counts) -> list[str]:
    from prepare_data import hyphen_evidence, read_acos, read_sb1  # raw, pre-deduplication units

    sb1 = read_sb1("train") + read_sb1("test")
    compounds, pairs = hyphen_evidence(sb1)
    acos = read_acos("train", compounds, pairs) + read_acos("test", compounds, pairs)
    labels = lambda u: frozenset((c, p) for c, p, _, _ in u["opinions"] if not c.startswith(OUT_OF_SCOPE))  # noqa: E731
    by_key: dict[str, tuple[list, list]] = {}
    for corpus, units in ((0, sb1), (1, acos)):
        for u in units:
            by_key.setdefault(match_key(u["text"]), ([], []))[corpus].append(u)
    shared, examples = Counter(), []
    for sem, aco in by_key.values():  # one verdict per distinct text found in both corpora
        if not (sem and aco):
            continue
        a, b = {labels(u) for u in aco}, {labels(v) for v in sem}
        cats = lambda ls: {frozenset(c for c, _ in x) for x in ls}  # noqa: E731
        kind = "identical labels" if a == b else "same categories, different polarity" if cats(a) == cats(b) else "different categories"
        shared[kind] += 1
        if kind != "identical labels":
            examples.append([sem[0]["text"], "; ".join(", ".join(sorted(f"{c} {p}" for c, p in x)) or "(none)" for x in b),
                             "; ".join(", ".join(sorted(f"{c} {p}" for c, p in x)) or "(none)" for x in a)])
    out = ["9. SEMEVAL VS ACOS ANNOTATION (plan D7.4)",
           f"same sentence in both corpora (raw data, all splits): {sum(shared.values())} distinct texts; "
           + ", ".join(f"{k}: {v}" for k, v in shared.most_common()), ""]
    if examples:
        out += table(["sentence (SemEval text)", "SemEval labels", "ACOS labels"], examples)
    rows = []
    for e in ("LAPTOP", "BATTERY", "DISPLAY", "KEYBOARD", "CPU", "SUPPORT", "OS", "SOFTWARE", "MULTIMEDIA_DEVICES"):
        dist = {k: np.array([counts[k][f"{e}#{a}"] for a in ATTRIBUTES], dtype=float) for k in ("S1", "S4")}
        if min(d.sum() for d in dist.values()) < 20:
            continue
        top = {k: ATTRIBUTES[int(np.argmax(d))] for k, d in dist.items()}
        js = jensenshannon(dist["S1"], dist["S4"], base=2) ** 2
        rows.append([e, int(dist["S1"].sum()), int(dist["S4"].sum()), f"{js:.3f}",
                     *(f"{top[k]} ({pct(d.max(), d.sum())})" for k, d in dist.items())])
    out += ["attribute distribution per entity, S1 vs S4 (Jensen-Shannon divergence: 0 = same, 1 = disjoint)"]
    out += table(["entity", "S1 n", "S4 n", "JS divergence", "S1 most used attribute", "S4 most used attribute"], rows)
    bat = {k: ", ".join(f"{a} {counts[k][f'BATTERY#{a}']}" for a in sorted(ATTRIBUTES, key=lambda a: -counts[k][f"BATTERY#{a}"])
                        if counts[k][f"BATTERY#{a}"]) for k in ("S1", "S4")}
    pad = re.compile(r"\b(?:touch ?pad|track ?pad|mouse ?pad)", re.I)
    pad_ent = {k: Counter(cat_rows(long[k])[lambda d: d.text.map(lambda t: bool(pad.search(t)))].entity) for k in ("S1", "S4")}
    pad_txt = "; ".join(f"{NAMES[k]}: " + ", ".join(f"{e} {v}" for e, v in c.most_common(4)) for k, c in pad_ent.items())
    acos_ops = opinions(pd.concat([long["S4"], long["T2"]]))
    asp = [t for terms in acos_ops.aspect_terms for t in terms.split(" | ")]
    opn = [t for terms in acos_ops.opinion_terms for t in terms.split(" | ")]
    out += [f"BATTERY attributes: S1 {bat['S1']}; S4 {bat['S4']}",
            f"entities of opinions in sentences mentioning a touchpad / trackpad: {pad_txt}",
            f"opinion-less sentences: S4 {pct((long['S4'].drop_duplicates('sentence_id').n_opinions == 0).sum(), long['S4'].sentence_id.nunique())}, "
            f"S1 {pct((long['S1'].drop_duplicates('sentence_id').n_opinions == 0).sum(), long['S1'].sentence_id.nunique())}",
            f"ACOS implicit (NULL) terms: aspect {pct(asp.count('NULL'), len(asp))}, opinion {pct(opn.count('NULL'), len(opn))}",
            "sentences with an uppercase letter: " + ", ".join(
                f"{k} {pct(t.str.contains('[A-Z]').sum(), len(t))}" for k in SETS for t in [long[k].drop_duplicates("sentence_id").text]), ""]
    return out


def shift(long, sents, counts) -> list[str]:
    def dist(cnt, cats):
        v = np.array([cnt[c] for c in cats], dtype=float)
        return v / v.sum()

    space = json.loads((FINAL / "label_space.json").read_text())
    s14 = counts["S1"] + counts["S4"]
    vocab = {k: Counter(w for t in sents[k].text for w in re.findall(r"[a-z0-9']+", t.lower())) for k in SETS}
    vocab["S1+S4"] = vocab["S1"] + vocab["S4"]
    rows = []
    for tr, te in (("S1", "T1"), ("S1", "T2"), ("S4", "T2"), ("S1+S4", "T1"), ("S1+S4", "T2")):
        cnt = s14 if tr == "S1+S4" else counts[tr]
        cats = sorted(set(cnt) | set(counts[te]))
        js = jensenshannon(dist(cnt, cats), dist(counts[te], cats), base=2) ** 2
        unseen = [c for c in counts[te] if not cnt[c]]
        oov_tok = sum(v for w, v in vocab[te].items() if w not in vocab[tr])
        rows.append([tr, NAMES[te], f"{js:.3f}", f"{len(unseen)} cats / {sum(counts[te][c] for c in unseen)} sentences",
                     pct(oov_tok, sum(vocab[te].values())), pct(sum(w not in vocab[tr] for w in vocab[te]), len(vocab[te]))])
    out = ["10. TRAIN -> TEST SHIFT"]
    out += table(["train", "test", "category JS divergence", "test categories unseen in train", "OOV tokens", "OOV word types"], rows)
    empty = {k: (sents[k].n_opinions == 0).mean() for k in SETS}
    out += [f"opinion-less sentences: S1 {pct(empty['S1'], 1)} -> T1 {pct(empty['T1'], 1)}; S4 {pct(empty['S4'], 1)} -> T2 {pct(empty['T2'], 1)}",
            f"T1 categories unseen in S1: {space['unseen_in_semeval16_train']['semeval16_test']}",
            f"T1 categories unseen in S1 + S4: {space['unseen_in_merged_train']['semeval16_test']}", ""]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    ents = sorted(ENTITIES, key=lambda e: -sum(v for c, v in counts["S1"].items() if c.startswith(e + "#")))
    y = np.arange(len(ents))[::-1]
    for ax, (a, b) in zip(axes, (("S1", "T1"), ("S4", "T2"))):
        for i, k in enumerate((a, b)):
            tot = sum(counts[k].values())
            share = [100 * sum(v for c, v in counts[k].items() if c.startswith(e + "#")) / tot for e in ents]
            ax.barh(y + (0.2 if i == 0 else -0.2), share, height=0.38, color=SET_COLOR[k], label=NAMES[k], edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, ents, fontsize=7.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{'SemEval' if a == 'S1' else 'ACOS'}: entity share, train vs test", fontsize=10)
        style(ax)
    out += [save(fig, "entity_shift.png"), ""]
    return out


def nuance_slices(long, sents) -> list[str]:
    """D5.1 text slices from the `nuances` column that clean_data.py writes with src.nuance_checker."""
    tags = {k: sents[k].nuances.map(lambda v: set(v.split("|")) - {""}) for k in SETS}
    out = ["11. NUANCES (D5.1 text slices, src.nuance_checker; unvalidated until D5.3)", "share of sentences"]
    out += table(["set", *NUANCES, "any", "elongated"],
                 [[k, *(pct(tags[k].map(lambda t, n=n: n in t).sum(), len(tags[k])) for n in NUANCES),
                   pct((tags[k].map(len) > 0).sum(), len(tags[k])), pct(sents[k].elongated.sum(), len(tags[k]))] for k in SETS])
    s1 = opinions(long["S1"])
    rows = []
    for name in NUANCES:
        hit = s1.nuances.map(lambda v, n=name: n in v.split("|"))
        for flag, sub in (("with", s1[hit]), ("without", s1[~hit])):
            c = Counter(sub.polarity)
            rows.append([name, flag, len(sub), *(pct(c[p], len(sub)) for p in POLARITIES)])
    out += ["S1 opinion polarity with and without each nuance"]
    out += table(["nuance", "", "opinions", *POLARITIES], rows)
    return out


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    cleaned = {s: load_long(FINAL / f"{s}_merged_cleaned.csv") for s in ("train", "test")}
    merged = {s: load_long(FINAL / f"{s}_merged.csv") for s in ("train", "test")}
    long = {k: cleaned[sp][cleaned[sp].source == src].reset_index(drop=True) for k, (sp, src) in SETS.items()}
    sents = {k: to_sentences(v) for k, v in long.items()}
    merged_sents = {k: to_sentences(merged[sp][merged[sp].source == src]) for k, (sp, src) in SETS.items()}

    space_txt, counts = label_space(long)
    s1_ops = opinions(long["S1"])
    report = [
        "EDA: SemEval-2016 laptops + Laptop-ACOS (scripts/eda.py, from DATA/final/*_merged_cleaned.csv)",
        "sets: " + ", ".join(NAMES.values()), "",
        f"train sentences: {sum(len(sents[k]) for k in ('S1', 'S4'))} ({len(sents['S1'])} S1 + {len(sents['S4'])} S4)",
        f"test sentences: {sum(len(sents[k]) for k in ('T1', 'T2'))} ({len(sents['T1'])} T1 + {len(sents['T2'])} T2)",
        f"categories used: S1 {len(counts['S1'])}, S1 + S4 {len(counts['S1'] + counts['S4'])}, possible {len(ENTITIES) * len(ATTRIBUTES)}",
        "S1 polarity: " + ", ".join(f"{p} {pct(v, len(s1_ops))}" for p, v in sorted(Counter(s1_ops.polarity).items(), key=lambda x: -x[1])),
        "opinion-less sentences: " + ", ".join(f"{k} {pct((sents[k].n_opinions == 0).mean(), 1)}" for k in SETS), "",
    ]
    report += overview(long, sents, merged_sents)
    report += space_txt
    report += polarity(long, sents)
    report += per_sentence(sents)
    report += lengths(sents)
    report += structure(sents)
    report += cooccurrence(sents)
    report += out_of_scope(sents)
    report += annotation_differences(long, counts)
    report += shift(long, sents, counts)
    report += nuance_slices(long, sents)
    (REPORT_DIR / "eda.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"saved {(REPORT_DIR / 'eda.txt').relative_to(ROOT)}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
