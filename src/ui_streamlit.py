"""Streamlit app: laptop reviews -> aspect-based sentiment, with every step of the system open for inspection.

Pages (top navigation): Analyse review (default), Batch CSV, and About system with How it works and Models and
experiments. The Verbose switch on the two analysis pages opens a trace panel on the right: preprocessing, every gate and
why it passed or failed, the raw model output, and the post-processing checks that produced the final tuples.

Run from the repo root (so .streamlit/config.toml sets the theme):
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
from src import domain_gate, post_process  # noqa: E402
from src.main import FLOOR, Pipeline  # noqa: E402
from src.post_process import aspect_table, brief, confidence_label, opinion_rows, rating, stars  # noqa: E402

EXAMPLES = {
    "Mixed review": "I bought this laptop last month. The battery easily lasts 9 hours and the screen is gorgeous. "
                    "But the keyboard feels cheap and the fan gets loud when gaming. Customer support was useless.",
    "Sarcasm and parts": 'Yeah right, "premium" build. The hinge broke after two weeks. Not bad for the price though.',
    "Greeting and specs": "Hello everyone! Apple mac mini. The trackpad is smooth and the 16GB of RAM handles everything.",
    "Implicit opinion": "The So called laptop Runs to Slow and I hate it! Do not buy it!",
    "Support story": "Called tech support about the screen. Gave me the runaround for an entire day and treated me like "
                     "an imbecile.",
    "Off-domain": "The restaurant is good and has good food. I love my new phone.",
}
SAMPLE_CSV = pd.DataFrame({
    "product": ["Apple Mac mini", "Apple Mac mini", "Apple Mac mini", "Dell XPS 13", "Dell XPS 13", "Dell XPS 13"],
    "review": [EXAMPLES["Greeting and specs"], "Setup took five minutes. It is silent and fast.",
               "The keyboard I bought with it died in a week.", EXAMPLES["Mixed review"],
               EXAMPLES["Sarcasm and parts"], EXAMPLES["Support story"]]})
FACE = {"positive": "sentiment_satisfied", "neutral": "sentiment_neutral", "negative": "sentiment_dissatisfied"}
MOOD = {"positive": "pos", "neutral": "neu", "negative": "neg"}

CSS = """
<style>
:root { --muted:#5b5b66; --line:rgba(24,24,27,.12); --soft:rgba(24,24,27,.035); --accent:#2f54c6;
        --pos:#1c7a50; --neu:#8a5f0c; --neg:#b2312d; --star:#2f54c6; }
@media (prefers-color-scheme: dark) {
  :root { --muted:#a4a4ae; --line:rgba(236,236,239,.14); --soft:rgba(236,236,239,.05); --accent:#93a8f2;
          --pos:#55c18f; --neu:#dcaa48; --neg:#f07d74; --star:#93a8f2; } }
.block-container { max-width: 1240px; padding-top: 4.6rem; padding-bottom: 4rem; }
h1 { letter-spacing: -.015em; }
.ms { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
      white-space: nowrap; letter-spacing: normal; text-transform: none; font-feature-settings: 'liga'; vertical-align: middle; }
.lede { color: var(--muted); max-width: 68ch; margin: -.5rem 0 1.4rem; font-size: 1.02rem; }
.pos { color: var(--pos); } .neu { color: var(--neu); } .neg { color: var(--neg); } .muted { color: var(--muted); }
.mono { font-family: "Source Code Pro", ui-monospace, monospace; }

.summary { display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 1.25rem; align-items: center;
           padding: 1.1rem 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: .8rem 0 1.1rem; }
.summary .ms { font-size: 3.1rem; }
.summary .t { font-size: 1.32rem; font-weight: 700; }
.summary .s { color: var(--star); font-size: 1.2rem; letter-spacing: .1em; }
.summary .c { color: var(--muted); font-size: .9rem; line-height: 1.55; text-align: right; font-variant-numeric: tabular-nums; }

.sent { padding: .95rem 0 .85rem; border-top: 1px solid var(--line); }
.sent:first-of-type { border-top: 0; padding-top: .2rem; }
.sent .x { font-weight: 600; font-size: 1.02rem; margin-bottom: .3rem; }
.sent .n { color: var(--muted); font-weight: 500; margin-right: .5rem; font-variant-numeric: tabular-nums; }
.sent.off .x { color: var(--muted); font-weight: 500; }
.op { display: grid; grid-template-columns: 2.1rem minmax(0,1.5fr) minmax(0,.9fr) minmax(0,1.1fr); gap: .8rem;
      align-items: center; padding: .32rem 0; }
.op > .ms { font-size: 1.75rem; }
.op .a { overflow-wrap: anywhere; }
.op .asp { font-family: "Source Code Pro", ui-monospace, monospace; font-size: .86rem; font-weight: 600; }
.op .sub { display: block; font-size: .8rem; color: var(--muted); margin-top: .1rem; }
.op .r { white-space: nowrap; }
.op .r .st { color: var(--star); letter-spacing: .08em; font-size: 1.05rem; }
.op .r .sub { letter-spacing: 0; }
.op .k { font-size: .86rem; }
.tag { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0 .5rem; font-size: .74rem;
       color: var(--muted); margin-left: .35rem; line-height: 1.5; font-weight: 500; vertical-align: 1px; }
.tag.on { border-color: var(--accent); color: var(--accent); }
.none { color: var(--muted); padding: .15rem 0 .1rem 2.9rem; font-size: .92rem; }
.brief { font-family: "Source Code Pro", ui-monospace, monospace; font-size: .78rem; color: var(--muted);
         padding: .35rem 0 0 2.9rem; overflow-wrap: anywhere; }

.st-key-trace { position: sticky; top: 4.4rem; max-height: calc(100dvh - 5.6rem); overflow-y: auto;
                border-left: 1px solid var(--line); padding: 0 .2rem 1rem 1.3rem; }
[data-testid="stColumn"]:has(.st-key-trace) { align-self: stretch; }
.th { font-weight: 700; font-size: .86rem; margin: .8rem 0 .25rem; }
.th:first-child { margin-top: 0; }
.chk { display: grid; grid-template-columns: 1.35rem 7.6rem minmax(0,1fr); gap: .35rem; font-size: .83rem; padding: .12rem 0;
       align-items: start; }
.chk .ms { font-size: 1.05rem; }
.chk .d { color: var(--muted); overflow-wrap: anywhere; }
.kv { display: grid; grid-template-columns: 6.6rem minmax(0,1fr); gap: .35rem; font-size: .83rem; padding: .1rem 0; }
.kv .d { overflow-wrap: anywhere; }
.verdict { font-size: .84rem; margin: .35rem 0 .1rem; font-weight: 600; }
.mrow { display: grid; grid-template-columns: minmax(0,1fr) 3.2rem 4.6rem; gap: .4rem; font-size: .8rem; padding: .08rem 0;
        font-variant-numeric: tabular-nums; }
.mrow .mono { overflow-wrap: normal; }
.small { font-size: .8rem; }

.flow { display: grid; grid-template-columns: 11rem minmax(0,1fr); gap: 2.2rem; padding: 1.3rem 0;
        border-top: 1px solid var(--line); }
.flow h3 { margin: 0; padding: 0; font-size: 1.15rem; }
.flow p { margin: 0 0 .5rem; max-width: 70ch; }
.flow .lead { font-weight: 600; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1.6rem; margin: .6rem 0 2rem; }
.stats .v { font-size: 2.1rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; }
.stats .l { color: var(--muted); font-size: .88rem; }
.group { padding: .9rem 1rem; border-radius: 10px; background: var(--soft); height: 100%; }
.group h4 { margin: 0 0 .5rem; padding: 0; font-size: 1rem; }
.group p { margin: .25rem 0; font-size: .9rem; }
.legend { display: grid; grid-template-columns: 2.2rem minmax(0,1fr); gap: .6rem 1rem; align-items: center; max-width: 64ch; }
.legend .ms { font-size: 1.6rem; }
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-secondary"]:active { transform: translateY(1px); }

@media (max-width: 768px) {
  .op { grid-template-columns: 2.1rem minmax(0,1fr); }
  .op .r, .op .k { grid-column: 2; }
  .summary { grid-template-columns: auto minmax(0,1fr); }
  .summary .c { grid-column: 1 / -1; text-align: left; }
  .flow { grid-template-columns: 1fr; gap: .4rem; }
  .stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .st-key-trace { position: static; max-height: none; border-left: 0; padding-left: 0; }
}
</style>
"""


@st.cache_resource(show_spinner="Loading bge_detr and the domain gate (first run only)")
def load_pipeline() -> Pipeline:
    return Pipeline()


def esc(x) -> str:
    return html.escape(str(x))


def icon(name: str, cls: str = "") -> str:
    return f'<span class="ms {cls}">{name}</span>'


def md(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- results
def summary_html(records: list[dict]) -> str:
    ops = [t for r in records for t in r["tuples"] if t["sentiment"]]
    kept = sum(r["gate"]["about_laptop"] for r in records)
    counts = (f'{len(records)} sentence{"s" * (len(records) != 1)}<br>{kept} passed the gates<br>'
              f'{len(ops)} opinion{"s" * (len(ops) != 1)}')
    if not ops:
        return (f'<div class="summary">{icon("sentiment_neutral", "muted")}<div><div class="t">No laptop opinion found'
                f'</div><div class="muted">Every sentence was stopped by a gate or carries no opinion.</div></div>'
                f'<div class="c">{counts}</div></div>')
    score = 1 + sum(t["expected_star"] for t in ops) / len(ops)
    mood = "positive" if score >= 3.5 else "negative" if score < 2.5 else "neutral"
    full = int(score + 0.5)
    return (f'<div class="summary">{icon(FACE[mood], MOOD[mood])}<div><div class="t">Mostly {mood}</div>'
            f'<div><span class="s">{"★" * full}{"☆" * (5 - full)}</span> <span class="muted">{score:.1f} out of 5</span>'
            f'</div></div><div class="c">{counts}</div></div>')


def opinion_html(t: dict) -> str:
    m = MOOD[t["sentiment"]]
    sub = (f'<span class="sub">model said {esc(t["model_category"])}</span>'
           if t["model_category"] != t["aspect"] else "")
    source = "" if t["source"] == "model" else f'<span class="tag on">{esc(t["source"])}</span>'
    return (f'<div class="op">{icon(FACE[t["sentiment"]], m)}<div class="a"><span class="asp">{esc(t["aspect"])}</span>'
            f'{source}{sub}</div>'
            f'<div class="r"><span class="st">{stars(t)}</span><span class="sub">{rating(t)} out of 5, '
            f'<span class="{m}">{t["sentiment"]}</span></span></div>'
            f'<div class="k">{t["confidence"]:.0%} confident<span class="tag">{confidence_label(t)}</span></div></div>')


def sentence_html(r: dict) -> str:
    body = "".join(opinion_html(t) for t in r["tuples"] if t["sentiment"])
    rest = [t for t in r["tuples"] if not t["sentiment"]]
    if rest:
        note = rest[0]["note"]
        note = "no opinion" if note == "no opinion" else f"stopped: {note}"
        body += f'<div class="none">{esc(", ".join(t["aspect"] for t in rest))}, {esc(note)}</div>'
    off = "" if r["gate"]["about_laptop"] or r["merged"] else " off"
    return (f'<div class="sent{off}"><div class="x"><span class="n">{r["sentence_index"] + 1}</span>{esc(r["sentence"])}</div>'
            f'{body}<div class="brief">{esc(brief(r["tuples"]))}</div></div>')


# ----------------------------------------------------------------------------------------------- verbose trace
def check_row(name: str, passed: bool | None, detail: str) -> str:
    sym, cls = ("check_circle", "pos") if passed else ("remove", "muted") if passed is None else ("cancel", "neg")
    return f'<div class="chk">{icon(sym, cls)}<div>{esc(name)}</div><div class="d">{esc(detail)}</div></div>'


def breakable(category: str) -> str:
    """E#A with line-break opportunities after # and _, so long names wrap between words in the narrow trace panel."""
    return esc(category).replace("#", "#<wbr>").replace("_", "_<wbr>")


def kv(key: str, value: str) -> str:
    return f'<div class="kv"><div class="muted">{esc(key)}</div><div class="d">{value}</div></div>'


def trace_html(r: dict, tau: float) -> str:
    g = r["gate"]
    out = ['<div class="th">Preprocessing</div>', kv("Text", esc(r["text"])),
           kv("Repaired", esc(", ".join(r["artefacts"]) or "nothing")),
           kv("Nuances", esc(", ".join(n.replace("_cand", " (candidate)") for n in r["nuances"]) or "none")),
           kv("Mentions", "<br>".join(f"<b>{esc(e)}</b>: {esc(v)}" for e, v in g["mentions"].items()) or "none"),
           '<div class="th">Gates</div>']
    ran = {x["gate"] for x in g["gates"]}
    for x in g["gates"]:
        out.append(check_row(x["gate"], None if x["gate"] == "laptop named" and not x["passed"] else x["passed"],
                             x["detail"]))
    out += [check_row(n, None, "not reached") for n in domain_gate.GATES if n not in ran]
    out.append(f'<div class="verdict {"pos" if g["about_laptop"] else "neg"}">'
               + ("Passed: sent to the model" if g["about_laptop"] else f"Stopped: {esc(g['reason'])}") + "</div>")
    out.append('<div class="th">Model output</div>')
    if g["about_laptop"]:
        rows = r["opinions"][:6]
        out.append(f'<div class="muted small">Categories scoring {FLOOR} or more. Kept at τ = {tau:.2f}, or at '
                   f'{post_process.TAU_LOW:.2f} when the sentence mentions a part and nothing reaches τ.</div>')
        out += [f'<div class="mrow"><span class="mono">{breakable(o["category"])}</span><span>{o["score"]:.2f}</span>'
                f'<span class="{MOOD[o["sentiment"]]}">{o["sentiment"]}</span></div>' for o in rows]
        out += [] if rows else [f'<div class="muted small">No category scores {FLOOR} or more.</div>']
    else:
        out.append('<div class="muted small">Not run: the gates stopped this sentence.</div>')
    if r["merged"]:
        m = r["merged"]
        out += ['<div class="th">Merge</div>', kv("Joined with", f"the {m['with']} sentence"), kv("Text", esc(m["text"])),
                kv("New", esc(", ".join(o["category"] for o in r["merged_opinions"])))]
    out.append('<div class="th">Post-processing</div>')
    for t in r["tuples"]:
        if not t["sentiment"]:
            out.append(kv(t["aspect"], esc(t["note"])))
            continue
        out.append(f'<div class="verdict">{esc(t["aspect"])} <span class="muted" style="font-weight:400">from '
                   f'{esc(t["source"])}</span></div>')
        out += [check_row(c["check"], c["passed"], c["detail"]) for c in t["checks"]]
    return "".join(out)


def show_trace(records: list[dict], tau: float) -> None:
    with st.container(key="trace"):
        st.markdown("#### Trace")
        st.caption("What each step decided, sentence by sentence.")
        for r in records:
            label = r["sentence"] if len(r["sentence"]) <= 46 else r["sentence"][:44] + "..."
            with st.expander(f"{r['sentence_index'] + 1}. {label}", expanded=len(records) == 1):
                md(trace_html(r, tau))


# ----------------------------------------------------------------------------------------------- shared views
def output_json(records: list[dict]) -> str:
    keep = ("aspect", "category", "model_category", "sentiment", "confidence", "source", "note")
    return json.dumps([{"product": r["product"], "review": r["review_id"], "sentence": r["sentence"],
                        "tuples": [{k: t[k] for k in keep if k in t} | ({"rating": rating(t), "stars": stars(t)} if t["sentiment"]
                                                              else {}) for t in r["tuples"]]} for r in records],
                      ensure_ascii=False, indent=2, default=str)


def show_results(records: list[dict], product: str, verbose: bool, key: str) -> None:
    main, side = st.columns([1.55, 1], gap="large") if verbose else (st.container(), None)
    with main:
        md(summary_html(records))
        t_sent, t_out, t_table = st.tabs(["Sentences", "Output", "Aspect table"])
        with t_sent:
            reviews = list(dict.fromkeys((r["product"], r["review_id"]) for r in records))
            for p, rid in reviews:
                rs = [r for r in records if (r["product"], r["review_id"]) == (p, rid)]
                if len(reviews) > 1:
                    st.markdown(f"**{p}**, review {rid + 1 if isinstance(rid, int) else rid}")
                md("".join(sentence_html(r) for r in rs))
        with t_out:
            st.caption("The brief's format, one line per sentence.")
            st.code("\n".join(f"{r['sentence']}\n  {brief(r['tuples'])}" for r in records), language="text")
            rows = opinion_rows(records)
            c1, c2 = st.columns(2)
            c1.download_button("Download CSV", rows.to_csv(index=False), "opinions.csv", "text/csv", key=f"csv_{key}",
                               width="stretch")
            c2.download_button("Download JSON", output_json(records), "output.json", "application/json",
                               key=f"json_{key}", width="stretch")
        with t_table:
            table = aspect_table(opinion_rows(records))
            if table.empty:
                st.info("No opinions, so there is no aspect table yet.")
            for p, t in table.groupby("product", sort=False):
                st.markdown(f"**{product or p}**")
                st.dataframe(t.drop(columns="product"), hide_index=True, width="stretch",
                             column_config={"rating": st.column_config.NumberColumn("rating", format="%.1f")})
            st.caption("Rating: 1 plus the mean expected star over the reviews that mention the category, one vote per "
                       "review.")
    if side is not None:
        with side:
            show_trace(records, load_pipeline().model.tau)


def page_header(title: str, lede: str, key: str) -> bool:
    left, right = st.columns([5, 1], vertical_alignment="bottom")
    left.title(title)
    verbose = right.toggle("Verbose", key=f"verbose_{key}",
                           help="Open the trace panel: preprocessing, gates, raw model output and post-processing checks")
    md(f'<p class="lede">{esc(lede)}</p>')
    return verbose


# ----------------------------------------------------------------------------------------------- pages
def analyse_page() -> None:
    verbose = page_header("Analyse a review", "Paste a laptop review. Each sentence gets its aspects, a sentiment, a 1 to "
                                              "5 rating and a confidence.", "one")
    pipe = load_pipeline()
    st.session_state.setdefault("text", EXAMPLES["Mixed review"])

    def pick() -> None:
        if st.session_state.example:
            st.session_state.text = EXAMPLES[st.session_state.example]
            st.session_state.pop("one", None)

    st.pills("Examples", list(EXAMPLES), key="example", on_change=pick)
    c1, c2 = st.columns([3, 1])
    c1.text_area("Review", key="text", height=130, placeholder="Type or paste a laptop review")
    product = c2.text_input("Product", value="Apple Mac mini", help="Names the aspect-sentiment table")
    run = st.button("Analyse", type="primary")
    text = st.session_state.text
    if text.strip() and (run or "one" not in st.session_state):
        with st.spinner("Analysing"):
            st.session_state.one = pipe.run(pd.DataFrame({"product": [product or "review"], "review_id": [0],
                                                          "text": [text]}))
    records = st.session_state.get("one")
    if not text.strip():
        st.info("Enter a review, or pick one of the examples above.")
    elif not records:
        st.warning("Nothing to analyse: the text has no sentence after preprocessing.")
    else:
        show_results(records, product, verbose, "one")


def batch_page() -> None:
    verbose = page_header("Batch CSV", "Upload a CSV with one review per row. Rows of one product share an aspect table.",
                          "batch")
    pipe = load_pipeline()
    f = st.file_uploader("CSV file", type=["csv"])
    if f is None and st.button("Use the sample file", help="Six reviews of two laptops"):
        st.session_state.sample = True
    if f is not None:
        try:
            data = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception as e:  # malformed upload: say what failed, keep the page usable
            st.error(f"Could not read the file as CSV: {e}")
            return
    elif st.session_state.get("sample"):
        data = SAMPLE_CSV
        with st.expander("Sample file"):
            st.dataframe(data, hide_index=True, width="stretch")
    else:
        st.info("No file yet. Upload a CSV, or try the sample file.")
        return
    if data.empty:
        st.warning("The file has no rows.")
        return
    cols = list(data.columns)
    c1, c2, c3 = st.columns([1.2, 1.2, .8])
    text_col = c1.selectbox("Text column", cols, index=next(
        (i for i, c in enumerate(cols) if any(k in c.lower() for k in ("review", "text"))), 0))
    prod_col = c2.selectbox("Product column", ["(none)", *cols],
                            index=next((i + 1 for i, c in enumerate(cols) if "product" in c.lower()), 0))
    limit = c3.number_input("Rows", 1, len(data), min(200, len(data)))
    if st.button("Analyse file", type="primary"):
        d = data.head(int(limit))
        reviews = pd.DataFrame({"product": d[prod_col] if prod_col != "(none)" else "reviews",
                                "review_id": range(len(d)), "text": d[text_col]})
        with st.spinner(f"Analysing {len(reviews)} reviews"):
            st.session_state.batch = pipe.run(reviews)
    records = st.session_state.get("batch")
    if records is None:
        return
    if not records:
        st.warning("No sentences found in that column.")
        return
    products = list(dict.fromkeys(r["product"] for r in records))
    chosen = st.selectbox("Product", ["All products", *products]) if len(products) > 1 else products[0]
    shown = records if chosen == "All products" else [r for r in records if r["product"] == chosen]
    show_results(shown, "" if chosen == "All products" else chosen, verbose, "batch")


def how_page() -> None:
    st.title("How it works")
    md('<p class="lede">A review passes through four steps. Each step can stop a sentence or change its answer, and the '
       'Verbose switch on the analysis pages shows every decision.</p>')
    steps = [
        ("Preprocess", "Clean the text, split it into sentences.",
         "HTML entities and tags, Unicode forms, curly quotes, invisible characters and stretched words (soooo) are "
         "repaired. Each sentence gets nuance tags (negation, contrast, comparison, question) and its mentions: the "
         "laptop parts it names, found by word lists or by word similarity (\"build\" is close to the hardware label)."),
        ("Gate", "Decide whether the sentence is about a laptop.",
         "The model was trained on laptop reviews only, so it reads any opinion as a laptop opinion. Seven gates run "
         "before it, cheapest first. A stopped sentence is shown as {NULL#NULL} with the reason."),
        ("Detect", "bge_detr reads the sentence with its neighbours.",
         "The previous and next sentence are its context, and negation scopes are marked. Sixteen queries each "
         "predict an entity, an attribute and a star from 0 to 4. A category counts when its score reaches τ = 0.85."),
        ("Decide", "Turn scores into the brief's tuples.",
         f"Stars become a sentiment and a 1 to 5 rating. Entity checks decide whether the entity is explicit or NULL. "
         f"A sentence that names a part but scores just under τ uses a second threshold, "
         f"{post_process.TAU_LOW:.2f}. A {{NULL#NULL}} sentence is tried again joined with a neighbour."),
    ]
    md("".join(f'<div class="flow"><h3>{a}</h3><div><p class="lead">{b}</p><p class="muted">{esc(c)}</p></div></div>'
               for a, b, c in steps))

    st.subheader("The gates")
    c1, c2, c3 = st.columns([1.15, .85, 1.35])
    c1.markdown('<div class="group"><h4>Stop</h4><p><b>tokens</b>: no real word ("!!!", "123")</p><p><b>filler</b>: '
                'greetings or thanks only</p><p><b>off-topic</b>: another object is the subject ("The pizza was cold")'
                '</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="group"><h4>Pass</h4><p><b>laptop named</b>: laptop, computer, MacBook or a brand. No '
                'further evidence needed.</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="group"><h4>Evidence</h4><p><b>domain</b> must pass: the sentence is closer to the 22 '
                f'laptop labels than to 22 other topics (margin {domain_gate.MARGIN:+.2f} or more)</p><p>and one of '
                f'<b>aspect terms</b> (a part is mentioned) or <b>category</b> (closest laptop label at '
                f'{domain_gate.CATEGORY_FLOOR:.2f} or more)</p></div>', unsafe_allow_html=True)
    st.caption("Thresholds are fitted on laptop training sentences only. On the SemEval-2016 test sets the gates keep "
               "98.6% of laptop opinion sentences and stop 59.5% of restaurant sentences.")

    st.subheader("When is the entity NULL?")
    st.markdown("The brief: *when there is no explicit mention of the entity or attribute, E or A takes the value NULL*. "
                "The model always names an entity, so each opinion goes through five checks. The entity stays when it "
                "is **named** in the sentence, or when the sentence has an **explicit target** (some laptop part is "
                "mentioned) and one more check agrees: the entity is mentioned in a neighbouring sentence (**review "
                "context**), it is the **whole laptop**, or it is among the sentence's three closest labels "
                "(**label similarity**).")
    st.dataframe(pd.DataFrame(HOW_EXAMPLES, columns=["Sentence", "Output", "Why"]), hide_index=True, width="stretch")

    st.subheader("Reading the output")
    md(f'<div class="legend">{icon(FACE["positive"], "pos")}<div>Positive: 4 or 5 stars</div>'
       f'{icon(FACE["neutral"], "neu")}<div>Neutral: 3 stars, or a mixed opinion</div>'
       f'{icon(FACE["negative"], "neg")}<div>Negative: 1 or 2 stars</div>'
       f'<span class="ms" style="color:var(--star)">star</span><div>Rating from 1 to 5, shown only for opinions. It '
       f'stays inside its sentiment, so stars and face always agree.</div>'
       f'<span class="ms muted">verified</span><div>Confidence: the probability of the sentiment. <b>high</b> at 80% or '
       f'more with the 90% star interval inside one sentiment, <b>medium</b> at 60% or more, otherwise <b>low</b>.</div>'
       f'<span class="ms muted">block</span><div><span class="mono">{{NULL#NULL}}</span>: stopped by a gate, or no '
       f'opinion and nothing named. <span class="mono">{{LAPTOP#NULL}}</span>: named, but no opinion.</div></div>')


HOW_EXAMPLES = [
    ("Do not buy it!", "{NULL#GENERAL, negative}", "No explicit target, as in the brief"),
    ("It is the worst laptop ever.", "{LAPTOP#GENERAL, negative}", "LAPTOP is named"),
    ('Yeah right, "premium" build.', "{LAPTOP#QUALITY, negative}", '"build" is an explicit target, and the whole laptop'),
    ("The hinge broke after two weeks.", "{LAPTOP#QUALITY, negative}",
     "Scores 0.82: under τ, over the second threshold. The hinge is an explicit target"),
    ("...and the 16GB of RAM handles everything.", "{LAPTOP#OPERATION_PERFORMANCE, positive}",
     "RAM is an explicit target, and the whole laptop"),
    ("Called tech support about the screen. Gave me the runaround for an entire day.",
     "{NULL#OPERATION_PERFORMANCE, negative}", "The second sentence is {NULL#NULL} alone; joined with the first, the "
     "model finds a new opinion. No part is named in it, so the entity is NULL"),
]


def models_page() -> None:
    st.title("Models and experiments")
    md('<p class="lede">Four models were trained and compared on the SemEval-2016 laptop reviews (two more ran out of GPU '
       'memory). bge_detr at epoch 9 was selected on validation and runs this app.</p>')
    md('<div class="stats"><div><div class="v">47.7</div><div class="l">test joint F1 of (E#A, sentiment)</div></div>'
       '<div><div class="v">88.6%</div><div class="l">sentiment accuracy</div></div>'
       '<div><div class="v">0.35</div><div class="l">star error (MAE, 0 to 4 scale)</div></div>'
       '<div><div class="v">0.984</div><div class="l">detection AUC</div></div></div>')
    st.subheader("The candidates")
    st.dataframe(pd.DataFrame([
        ("bge_detr (selected)", 9, 47.7, 88.6, 71.2, 0.35, 0.879, 0.984),
        ("lfm_base", 10, 46.1, 84.0, 62.9, 0.52, 0.806, 0.962),
        ("bge_tweaked", 9, 36.7, 86.0, 72.2, 0.37, 0.872, 0.895),
        ("bge_base", 9, 36.0, 82.8, 65.8, 0.53, 0.797, 0.932),
    ], columns=["run", "epoch", "joint F1", "sentiment acc", "sentiment macro-F1", "star MAE", "star QWK",
                "detection AUC"]), hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(c, format=f) for c, f in [
            ("joint F1", "%.1f"), ("sentiment acc", "%.1f"), ("sentiment macro-F1", "%.1f"), ("star MAE", "%.2f"),
            ("star QWK", "%.3f"), ("detection AUC", "%.3f")]})
    st.caption("Test set: 808 sentences, 29% without an opinion. Each run at its best validation epoch.")
    st.subheader("Why bge_detr")
    st.markdown(
        "- **Best test joint F1**, and it wins 10 of 15 extended metrics, including every per-sentiment F1.\n"
        "- **The set head is what lifts it.** bge_tweaked shares its backbone; swapping in the set head moves joint F1 "
        "from 36.7 to 47.7.\n"
        "- **Stable, not decaying.** Test F1 is flat from epoch 7 to 10 (48.4 to 48.9).\n"
        "- **Small.** 35.4M parameters against 229.7M for the LFM encoder, and about 0.05 s per sentence on a CPU.")
    st.subheader("The system on top of the model")
    st.markdown(SYSTEM_RESULTS)
    st.subheader("Further reading")
    st.markdown("- `models/notebooks/ABSA_model_selection_report.md`: the decision, learning curves, overfitting and the "
                "comparison with the Colab model\n"
                "- `models/notebooks/ABSA_experiments_base_tweak.ipynb`: all six experiments\n"
                "- `models/notebooks/ABSA_colab.ipynb`: the Colab three-head model\n"
                "- `scripts/evaluate_system.py`: the end-to-end evaluation behind the numbers above")


SYSTEM_RESULTS = """Joint F1 of (E#A, sentiment) on the gold sentences, measured by `scripts/evaluate_system.py`.
Validation is the 15% of training reviews the model never trained on; the second threshold was fitted there.

| Setting | Validation | Test |
|---|---|---|
| Model only, τ = 0.85 | 58.9 | 48.4 |
| + second threshold 0.80 for sentences that name a part | 59.3 | 48.4 |
| + merge of {NULL#NULL} sentences with a neighbour (this app) | **59.8** | 48.2 |
| This app with the gates off | 60.0 | 48.4 |
| This app + part re-map (an option, off) | 58.5 | 47.3 |

The rules change the form of the answer more than the score. Entities follow the brief's NULL rule, and a sentence that
names a part no longer comes out without an opinion. The gates cost about 0.2 points on laptop text and stop 59.5% of
restaurant sentences. Part re-map ("the hinge broke" as HARDWARE) is off because the gold data files broken parts under
LAPTOP#QUALITY, and it costs about 1 point."""


def main() -> None:
    st.set_page_config(page_title="Laptop ABSA", page_icon=":material/laptop_mac:", layout="wide",
                       initial_sidebar_state="collapsed")
    md(CSS)
    st.logo(":material/laptop_mac:")
    st.navigation({
        "": [st.Page(analyse_page, title="Analyse review", url_path="analyse", default=True),
             st.Page(batch_page, title="Batch CSV", url_path="batch")],
        "About system": [st.Page(how_page, title="How it works", url_path="how-it-works"),
                         st.Page(models_page, title="Models and experiments", url_path="models")],
    }, position="top").run()


main()
