"""Streamlit app: laptop review -> preprocessing -> domain gates -> bge_detr -> post-processing, every step visible.

Run from the repo root (so .streamlit/config.toml sets the theme colour):
    streamlit run src/ui_streamlit.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run` puts src/ on the path, not the repo root
from src import domain_gate  # noqa: E402
from src.main import Pipeline  # noqa: E402
from src.post_process import aspect_table, brief, confidence_label, face, opinion_rows, rating, stars  # noqa: E402

EXAMPLES = {
    "Mixed review": "I bought this laptop last month. The battery easily lasts 9 hours and the screen is gorgeous. "
                    "But the keyboard feels cheap and the fan gets loud when gaming. Customer support was useless.",
    "Implicit opinion": "The So called laptop Runs to Slow and I hate it! Do not buy it!",
    "Off-domain": "The restaurant is good and has good food. I love my new phone.",
    "Greeting + spec": "Hello everyone! Apple mac mini. The trackpad is smooth and the 16GB of RAM handles everything.",
    "Sarcasm / negation": "Yeah right, \"premium\" build. The hinge broke after two weeks. Not bad for the price though.",
}

CSS = """
<style>
:root { --accent:#6366f1; --accent2:#06b6d4; --pos:#16a34a; --neu:#d97706; --neg:#dc2626; --muted:rgba(127,127,127,.75);
        --card:rgba(127,127,127,.07); --line:rgba(127,127,127,.22); }
.block-container { padding-top: 3.6rem; max-width: 1180px; }
[data-testid="stBaseButton-primary"] { background: linear-gradient(120deg, #4f46e5, #06b6d4); border: 0; font-weight: 700; }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); border: 0; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent); color: var(--accent); }
.hero { padding: 1.6rem 1.8rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
        background: radial-gradient(1200px 300px at 0% 0%, #818cf8 0%, transparent 60%),
                    linear-gradient(120deg, #4338ca 0%, #6366f1 45%, #06b6d4 100%);
        box-shadow: 0 10px 30px rgba(67,56,202,.25); }
.hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.hero p { margin: .35rem 0 0; opacity: .92; font-size: 1.02rem; }
.flow { display:flex; flex-wrap:wrap; gap:.4rem; margin-top: .9rem; }
.flow span { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); padding: .25rem .7rem;
             border-radius: 999px; font-size: .83rem; font-weight: 600; }
.flow b { opacity: .7; font-weight: 400; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .9rem 1rem; height: 100%; }
.kpi .v { font-size: 1.7rem; font-weight: 800; line-height: 1.1; }
.kpi .l { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.card { background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--accent); border-radius: 14px;
        padding: 1rem 1.1rem; margin: .7rem 0; }
.card.rejected { border-left-color: #9ca3af; opacity: .85; }
.card .sent { font-size: 1.05rem; font-weight: 600; margin-bottom: .45rem; }
.card .idx { color: var(--muted); font-weight: 500; margin-right: .35rem; }
.chip { display:inline-block; padding: .14rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 600;
        margin: .12rem .18rem .12rem 0; border: 1px solid var(--line); }
.chip.ok { background: rgba(22,163,74,.12); color: var(--pos); border-color: rgba(22,163,74,.35); }
.chip.no { background: rgba(220,38,38,.12); color: var(--neg); border-color: rgba(220,38,38,.35); }
.chip.skip { color: var(--muted); }
.chip.nu { background: rgba(99,102,241,.12); color: var(--accent); border-color: rgba(99,102,241,.35); }
.op { display:grid; grid-template-columns: 2.2rem minmax(0,1.5fr) 7.5rem minmax(0,1fr); gap:.7rem; align-items:center;
      padding: .55rem .2rem; border-top: 1px dashed var(--line); }
.op .f { font-size: 1.7rem; line-height: 1; }
.op .a { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86rem; font-weight: 700; overflow-wrap:anywhere; }
.op .a small { display:block; font-family: inherit; color: var(--muted); font-weight: 500; }
.op .s { color: #f59e0b; font-size: 1.15rem; letter-spacing: .06em; white-space: nowrap; }
.op .s small { display:block; color: var(--muted); font-size: .74rem; letter-spacing: 0; }
.bar { height: 7px; border-radius: 99px; background: var(--line); overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; }
.conf { font-size: .76rem; color: var(--muted); margin-top: .2rem; }
.null { color: var(--muted); font-style: italic; padding-top: .35rem; border-top: 1px dashed var(--line); }
.tuple { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; background: rgba(99,102,241,.08);
         border: 1px solid rgba(99,102,241,.25); border-radius: 8px; padding: .35rem .6rem; margin-top: .55rem;
         overflow-wrap: anywhere; }
.verdict { border-radius: 16px; padding: 1.1rem 1.3rem; display:flex; gap: 1rem; align-items:center;
           background: var(--card); border: 1px solid var(--line); }
.verdict .big { font-size: 3rem; line-height: 1; }
.verdict .t { font-size: 1.25rem; font-weight: 800; }
.verdict .st { color: #f59e0b; font-size: 1.4rem; letter-spacing: .08em; }
@media (max-width: 640px) { .op { grid-template-columns: 2rem minmax(0,1fr); } .hero h1 { font-size: 1.5rem; } }
</style>
"""

SENT_COLOR = {"positive": "#16a34a", "neutral": "#d97706", "negative": "#dc2626"}


@st.cache_resource(show_spinner="Loading bge_detr and the domain gate (first run only)...")
def load_pipeline() -> Pipeline:
    return Pipeline()


def esc(x) -> str:
    return html.escape(str(x))


def gate_chips(g: dict) -> str:
    out = []
    for x in g["gates"]:
        cls, mark = ("ok", "✓") if x["passed"] else ("skip", "–") if x["gate"] == "laptop named" else ("no", "✗")
        out.append(f'<span class="chip {cls}" title="{esc(x["detail"])}">{mark} {esc(x["gate"])}</span>')
    ran = {x["gate"] for x in g["gates"]}
    out += [f'<span class="chip skip">· {esc(n)}</span>' for n in domain_gate.GATES if n not in ran]
    return "".join(out)


def opinion_html(t: dict) -> str:
    if not t["sentiment"]:
        return f'<div class="null">{esc(t["aspect"])} · {esc(t["note"])}</div>'
    c, color = t["confidence"], SENT_COLOR[t["sentiment"]]
    interval = f'{t["star_low"] + 1}-{t["star_high"] + 1}' if t["star_low"] != t["star_high"] else str(t["star_low"] + 1)
    model_cat = "" if t["aspect"] == t["category"] else f'<small>model: {esc(t["category"])}</small>'
    return (f'<div class="op"><div class="f">{face(t)}</div>'
            f'<div class="a">{esc(t["aspect"])}{model_cat}<small style="color:{color}">{esc(t["sentiment"])}</small></div>'
            f'<div class="s">{stars(t)}<small>{rating(t)}/5 · 90% in {interval}★</small></div>'
            f'<div><div class="bar"><div style="width:{c * 100:.0f}%;background:{color}"></div></div>'
            f'<div class="conf">confidence {c:.0%} · {confidence_label(t)} · detection {t["score"]:.2f}</div></div></div>')


def sentence_card(r: dict) -> str:
    g = r["gate"]
    nu = "".join(f'<span class="chip nu">{esc(n)}</span>' for n in r["nuances"])
    body = "".join(opinion_html(t) for t in r["tuples"])
    return (f'<div class="card{"" if g["about_laptop"] else " rejected"}">'
            f'<div class="sent"><span class="idx">#{r["sentence_index"] + 1}</span>{esc(r["sentence"])}</div>'
            f'<div>{gate_chips(g)}{nu}</div>{body}<div class="tuple">{esc(brief(r["tuples"]))}</div></div>')


def kpi(value, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{esc(label)}</div></div>'


def verdict(rows: pd.DataFrame) -> str:
    op = rows[rows.sentiment.fillna("") != ""]
    if op.empty:
        return ('<div class="verdict"><div class="big">🤷</div><div><div class="t">No laptop opinion found</div>'
                '<div class="conf">Every sentence was rejected by a gate or carries no opinion.</div></div></div>')
    r = 1 + op.expected_star.mean()
    mood = "positive" if r >= 3.5 else "negative" if r < 2.5 else "neutral"
    st_ = "★" * int(r + 0.5) + "☆" * (5 - int(r + 0.5))
    counts = op.sentiment.value_counts()
    split = " · ".join(f"{face({'sentiment': s})} {counts.get(s, 0)} {s}" for s in ("positive", "neutral", "negative"))
    return (f'<div class="verdict"><div class="big">{face({"sentiment": mood})}</div><div>'
            f'<div class="t">Overall {mood} · {r:.1f} / 5</div><div class="st">{st_}</div>'
            f'<div class="conf">{split} · mean sentiment confidence {op.confidence.mean():.0%}</div></div></div>')


def show_results(records: list[dict], key: str) -> None:
    rows = opinion_rows(records)
    passed = sum(r["gate"]["about_laptop"] for r in records)
    n_op = int((rows.sentiment.fillna("") != "").sum())
    cols = st.columns(4)
    for c, (v, label) in zip(cols, [(len(records), "sentences"), (f"{passed}/{len(records)}", "passed the gates"),
                                    (n_op, "opinions"), (len({r['review_id'] for r in records}), "reviews")]):
        c.markdown(kpi(v, label), unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.markdown(verdict(rows), unsafe_allow_html=True)

    t_sent, t_out, t_table = st.tabs(["Sentence by sentence", "Output (brief format)", "Aspect-sentiment table"])
    with t_sent:
        for review_id in dict.fromkeys(r["review_id"] for r in records):
            rs = [r for r in records if r["review_id"] == review_id]
            if len({r["review_id"] for r in records}) > 1:
                product = rs[0]["product"]
                st.markdown(f"**Review {review_id + 1 if isinstance(review_id, int) else review_id}**"
                            + (f" · {esc(product)}" if product != "reviews" else ""))
            st.markdown("".join(sentence_card(r) for r in rs), unsafe_allow_html=True)
    with t_out:
        st.code("\n".join(f"{r['sentence']}\n  -> {brief(r['tuples'])}" for r in records), language="text")
        out = [{"review_id": r["review_id"], "sentence": r["sentence"], "about_laptop": r["gate"]["about_laptop"],
                "gate_reason": r["gate"]["reason"],
                "tuples": [{k: v for k, v in t.items() if k in ("aspect", "category", "sentiment", "confidence")}
                           | ({"rating": rating(t), "stars": stars(t), "face": face(t)} if t["sentiment"] else {})
                           for t in r["tuples"]]} for r in records]
        c1, c2 = st.columns(2)
        c1.download_button("Download opinions.csv", rows.to_csv(index=False), "opinions.csv", "text/csv",
                           key=f"csv_{key}", width="stretch")
        c2.download_button("Download output.json", json.dumps(out, ensure_ascii=False, indent=2, default=str),
                           "output.json", "application/json", key=f"json_{key}", width="stretch")
        with st.expander("JSON"):
            st.json(out, expanded=False)
    with t_table:
        table = aspect_table(rows)
        if table.empty:
            st.info("No opinions, so no aspect table.")
        else:
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"rating": st.column_config.ProgressColumn("rating", min_value=1, max_value=5,
                                                                                   format="%.1f")})


def sidebar(pipe: Pipeline) -> None:
    m = pipe.model
    with st.sidebar:
        st.markdown("### Model")
        st.markdown(f"**bge_detr** · epoch {m.epoch}  \nthreshold τ = {m.tau:.2f} · device `{m.device.type}`  \n"
                    f"{len(m.categories)} E#A categories (22 entities × 9 attributes)")
        st.markdown("### Gates")
        st.markdown("Hard: **tokens**, **filler**, **off-topic**  \nStrong pass: **laptop named**  \n"
                    f"Evidence: **domain** (margin ≥ {domain_gate.MARGIN:+.2f}) and **aspect terms** or **category** "
                    f"(cosine ≥ {domain_gate.CATEGORY_FLOOR:.2f})")
        st.caption("Hover a gate chip to see its score.")
        st.markdown("### Legend")
        st.markdown("🙂 positive · 😐 neutral · 🙁 negative  \n★ rating 1-5, shown only for opinions  \n"
                    "`NULL#…` entity not named in the sentence  \n`{NULL#NULL}` rejected by a gate or no opinion")


def main() -> None:
    st.set_page_config(page_title="Laptop ABSA", page_icon="💻", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>💻 Laptop Review Analyser</h1>'
                '<p>Aspect-based sentiment analysis: which part of the laptop each sentence talks about, and how the '
                'reviewer feels about it.</p><div class="flow"><span>1 Preprocess</span><b>→</b><span>2 Domain gates</span>'
                '<b>→</b><span>3 bge_detr</span><b>→</b><span>4 Post-process</span></div></div>', unsafe_allow_html=True)
    pipe = load_pipeline()
    sidebar(pipe)

    t_one, t_batch = st.tabs(["✍️  Analyse a review", "📄  Batch (CSV)"])
    with t_one:
        if "text" not in st.session_state:
            st.session_state.text = EXAMPLES["Mixed review"]
        st.caption("Try an example")
        cols = st.columns(len(EXAMPLES))
        for c, (name, text) in zip(cols, EXAMPLES.items()):
            if c.button(name, width="stretch"):
                st.session_state.text = text
        text = st.text_area("Review", key="text", height=130, placeholder="Type or paste a laptop review...")
        if st.button("Analyse", type="primary", width="stretch") or "last" not in st.session_state:
            st.session_state.last = text
        review = st.session_state.last
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            with st.spinner("Analysing..."):
                records = pipe.run(pd.DataFrame({"product": ["review"], "review_id": [0], "text": [review]}))
            if not records:
                st.warning("Nothing to analyse: the text has no sentence after preprocessing.")
            else:
                show_results(records, "one")

    with t_batch:
        st.markdown("Upload a CSV with one review per row. Rows of one product share an aspect table.")
        f = st.file_uploader("CSV file", type=["csv"])
        if f is not None:
            d = pd.read_csv(f, dtype=str, keep_default_na=False)
            if d.empty:
                st.warning("The file has no rows.")
                return
            c1, c2, c3 = st.columns(3)
            text_col = c1.selectbox("Text column", list(d.columns),
                                    index=next((i for i, c in enumerate(d.columns) if "text" in c.lower() or "review" in c.lower()), 0))
            prod_col = c2.selectbox("Product column (optional)", ["(none)", *d.columns])
            limit = c3.number_input("Max rows", 1, max(1, len(d)), min(200, len(d)))
            if st.button("Analyse file", type="primary", width="stretch"):
                d = d.head(int(limit))
                reviews = pd.DataFrame({"product": d[prod_col] if prod_col != "(none)" else "reviews",
                                        "review_id": d.index, "text": d[text_col]})
                with st.spinner(f"Analysing {len(reviews)} reviews..."):
                    records = pipe.run(reviews)
                if records:
                    show_results(records, "batch")
                else:
                    st.warning("No sentences found in that column.")


main()
