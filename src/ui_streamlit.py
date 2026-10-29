"""Streamlit app: laptop reviews -> aspect-based sentiment, with every step of the system open for inspection.

Pages (top navigation): Analysis (default), Batch CSV, About System (how it works) and Model Experiments. The Verbose
switch on the two analysis pages opens a trace panel on the right: preprocessing, every gate and why it passed or failed,
the raw model output, and the post-processing checks that produced the final tuples.

Run from the repo root (so .streamlit/config.toml sets the theme and fonts):
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
BRAND = "CDU ITCF ABSA"
LOGO = ('<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">'
        '<rect width="36" height="36" rx="9" fill="#2f54c6"/><text x="18" y="24" text-anchor="middle" '
        'font-family="DM Sans, Arial, sans-serif" font-weight="700" font-size="16" fill="#fff">a.</text></svg>')

CSS = """
<style>
:root { --bg:#f5f7fa; --surface:#fff; --text:#1f2533; --muted:#5f6b7d; --line:#e3e7ee; --soft:rgba(31,37,51,.035);
        --accent:#2f54c6; --pos:#1c7a50; --neu:#8a5f0c; --neg:#b2312d; --star:#2f54c6;
        --shadow:0 1px 2px rgba(16,24,40,.04), 0 16px 40px -20px rgba(16,24,40,.16);
        --serif:"Fraunces", Georgia, serif; --mono:"DM Mono", ui-monospace, monospace; --r:16px; --nav:4.5rem;
        --wire:#c3cad6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1219; --surface:#161b25; --text:#e6e8ee; --muted:#9aa3b2; --line:#283041; --soft:rgba(230,232,238,.045);
          --accent:#8da4f0; --pos:#55c18f; --neu:#dcaa48; --neg:#f07d74; --star:#8da4f0; --shadow:none; --wire:#3b4559; } }

/* top bar: brand on the left, pages on the right, the active page underlined */
header[data-testid="stHeader"] { height: var(--nav); background: var(--surface); border-bottom: 1px solid var(--line); }
[data-testid="stToolbar"] { height: 100%; padding-inline: max(0rem, calc((100% - 1176px) / 2 - 1rem)); }
:has(> img[data-testid="stHeaderLogo"]) { display: flex; align-items: center; gap: .75rem; }
:has(> img[data-testid="stHeaderLogo"])::after { content: "CDU ITCF ABSA"; font-weight: 600; font-size: 1.12rem;
  color: var(--text); letter-spacing: -.01em; white-space: nowrap; }
img[data-testid="stHeaderLogo"] { height: 2.25rem; width: 2.25rem; max-height: none; }
[data-testid="stTopNav"] { justify-content: flex-end; }
[data-testid="stTopNavLink"] { background: none !important; height: var(--nav); border-radius: 0; padding: 0 .15rem;
  margin: 0 .8rem; color: var(--muted); box-shadow: inset 0 -2px 0 transparent; transition: color .15s, box-shadow .15s; }
[data-testid="stTopNavLink"] :is(span, p) { color: inherit !important; font-weight: 500 !important; font-size: .95rem; }
[data-testid="stTopNavLink"]:hover { color: var(--text); }
.rc-overflow-item:nth-last-child(2) [data-testid="stTopNavLink"] { margin-right: 0; }
[data-testid="stTopNavLink"][aria-current="page"] { color: var(--accent); box-shadow: inset 0 -2px 0 var(--accent); }

.block-container, [data-testid="stMainBlockContainer"] { max-width: 1240px; padding: calc(var(--nav) + 3.5rem) 2rem 2rem; }
.ms { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
      white-space: nowrap; letter-spacing: normal; text-transform: none; font-feature-settings: 'liga'; vertical-align: middle; }
.pos { color: var(--pos); } .neu { color: var(--neu); } .neg { color: var(--neg); } .muted { color: var(--muted); }
.mono { font-family: var(--mono); }

/* hero */
.eyebrow { color: var(--accent); font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
           margin: 0 0 .85rem; }
.display { font-family: var(--serif); font-weight: 700; font-size: clamp(2.1rem, 3.4vw, 2.65rem); line-height: 1.08;
           letter-spacing: -.02em; color: var(--text); margin: 0 0 .9rem; }
.display .dot { color: var(--accent); }
.lede { color: var(--muted); max-width: 64ch; font-size: 1.05rem; line-height: 1.6; margin: 0 0 1.4rem; }
.chip { display: inline-flex; align-items: center; gap: .45rem; border: 1px solid var(--line); background: var(--surface);
        border-radius: 999px; padding: .4rem .85rem; font-size: .8rem; color: var(--muted); white-space: nowrap;
        font-variant-numeric: tabular-nums; }
.chip .live { width: .45rem; height: .45rem; border-radius: 50%; background: var(--pos); }
.st-key-hero_tools { margin-bottom: 1.4rem; }

/* cards: Streamlit containers keyed card_* and plain .card blocks */
[class*="st-key-card"], .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--shadow); padding: 1.75rem 2rem; }
[class*="st-key-card"] [data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .92rem; color: var(--text); }
.ct { font-family: var(--serif); font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; margin: 0 0 .35rem;
      line-height: 1.25; }
.sec { margin: 2.4rem 0 .2rem; }
.sec .ct { font-size: 1.6rem; }
[data-testid="stBaseButton-primary"] { min-height: 2.8rem; padding-inline: 1.35rem; }
[data-testid="stBaseButton-primary"] p { font-weight: 600; font-size: 1rem; }
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-secondary"]:active { transform: translateY(1px); }

/* results */
.summary { margin-bottom: 1rem; display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 1.5rem; align-items: center; }
.face { width: 3.6rem; height: 3.6rem; border-radius: 50%; display: grid; place-items: center; background: var(--soft); }
.face .ms { font-size: 2.3rem; }
.face.pos { background: color-mix(in srgb, var(--pos) 11%, transparent); }
.face.neu { background: color-mix(in srgb, var(--neu) 11%, transparent); }
.face.neg { background: color-mix(in srgb, var(--neg) 11%, transparent); }
.summary .t { font-family: var(--serif); font-size: 1.55rem; font-weight: 700; letter-spacing: -.01em; line-height: 1.2; }
.summary .s { color: var(--star); font-size: 1.15rem; letter-spacing: .1em; }
.figs { display: flex; gap: 2.4rem; padding-left: 2rem; border-left: 1px solid var(--line); }
.figs .v { font-size: 1.65rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.15; }
.figs .l { color: var(--muted); font-size: .8rem; }

.rh { font-weight: 600; margin: .6rem 0 .1rem; }
.sent { display: grid; grid-template-columns: 2.3rem minmax(0,1fr); gap: 0 1rem; padding: 1.15rem 0;
        border-top: 1px solid var(--line); }
.sent:first-child { border-top: 0; padding-top: .5rem; }
.sent .n { font-family: var(--mono); font-size: .76rem; color: var(--muted); border: 1px solid var(--line);
           border-radius: 6px; text-align: center; padding: .12rem 0; height: fit-content; margin-top: .2rem;
           font-variant-numeric: tabular-nums; }
.sent .x { font-weight: 600; font-size: 1.02rem; line-height: 1.5; margin-bottom: .55rem; }
.sent.off .x { color: var(--muted); font-weight: 500; }
.op { display: grid; grid-template-columns: 2rem minmax(0,1.5fr) minmax(0,.9fr) minmax(0,1.1fr); gap: .8rem;
      align-items: center; padding: .5rem .8rem; margin-bottom: .4rem; border-radius: 10px; background: var(--soft); }
.op > .ms { font-size: 1.65rem; }
.op .a { overflow-wrap: break-word; min-width: 0; }
.op .asp { font-family: var(--mono); font-size: .84rem; font-weight: 500; }
.op .sub { display: block; font-size: .78rem; color: var(--muted); margin-top: .1rem; }
.op .r { white-space: nowrap; }
.op .r .st { color: var(--star); letter-spacing: .08em; font-size: 1.05rem; }
.op .r .sub { letter-spacing: 0; }
.op .k { font-size: .86rem; font-variant-numeric: tabular-nums; }
.tag { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0 .5rem; font-size: .72rem;
       color: var(--muted); margin-left: .35rem; line-height: 1.55; font-weight: 500; vertical-align: 1px; }
.tag.on { border-color: var(--accent); color: var(--accent); }
.none { color: var(--muted); padding: .1rem 0; font-size: .92rem; }
.brief { font-family: var(--mono); font-size: .76rem; color: var(--muted); padding-top: .4rem; overflow-wrap: anywhere; }

/* verbose trace */
.st-key-trace { position: sticky; top: calc(var(--nav) + 1rem); max-height: calc(100dvh - var(--nav) - 2rem);
                overflow-y: auto; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
                box-shadow: var(--shadow); padding: 1.4rem 1.25rem 1rem; }
[data-testid="stColumn"]:has(.st-key-trace) { align-self: stretch; }
.th { font-size: .7rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--accent);
      margin: 1rem 0 .35rem; }
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

/* about and experiments */
.steps { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 1rem; }
.step .num { font-family: var(--mono); color: var(--accent); font-size: .8rem; }
.step .ct { margin-top: .5rem; }
.step p { margin: 0 0 .5rem; }
.step .lead { font-weight: 600; }
.grid3 { display: grid; grid-template-columns: 1.15fr .85fr 1.35fr; gap: 1rem; margin-bottom: .6rem; }
.grid3 p { margin: .3rem 0; font-size: .92rem; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1rem; }
.stats .card { padding: 1.4rem 1.5rem; }
.stats .v { font-size: 2.2rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; letter-spacing: -.01em; }
.stats .l { color: var(--muted); font-size: .86rem; margin-top: .3rem; }
.legend { display: grid; grid-template-columns: 2.2rem minmax(0,72ch); gap: .7rem 1rem; align-items: center; }
.legend .ms { font-size: 1.6rem; }
/* the seven gates */
.gt { display: grid; grid-template-columns: 2.2rem 7.6rem minmax(0,1fr) 13.7rem 8.4rem; font-size: .9rem; }
.gt > div { padding: .8rem 1.2rem .8rem 0; border-top: 1px solid var(--line); }
.gt .hdr { border-top: 0; padding: 0 0 .5rem; font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
           color: var(--muted); }
.gt .gh { grid-column: 1 / -1; padding: 1.2rem 0 .35rem; display: flex; flex-wrap: wrap; gap: .3rem .8rem; align-items: baseline; }
.gt .gh b { font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; color: var(--accent); }
.gt .gh span { color: var(--muted); font-size: .84rem; }
.gt .gn { font-family: var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.gt .gname { font-weight: 600; }
.gt .gx { color: var(--muted); font-size: .84rem; }
.gt .gthr { font-family: var(--mono); font-size: .8rem; }
.f { display: inline-block; font-family: var(--mono); font-size: .79rem; background: var(--soft); border: 1px solid var(--line);
     border-radius: 6px; padding: .08rem .4rem; margin: .2rem 0; overflow-wrap: anywhere; }
.rule { margin-top: 1.2rem; padding: 1rem 1.2rem; border-radius: 10px; background: var(--soft); font-size: .92rem; }
.rule p { margin: .35rem 0 0; }

/* flow chart: lanes per component, nodes joined by wires, side exits for stops */
.lane { display: grid; grid-template-columns: 11rem minmax(0,1fr); gap: 2rem; padding: 1.5rem 0 1.4rem;
        border-top: 1px solid var(--line); }
.lane:first-child { border-top: 0; padding-top: 0; }
.lh .num { font-family: var(--mono); color: var(--accent); font-size: .8rem; }
.lh .ct { font-size: 1.2rem; margin: .3rem 0 .2rem; }
.lh .mono { font-size: .76rem; color: var(--muted); overflow-wrap: anywhere; }
.spine { display: flex; flex-direction: column; gap: 1.3rem; }
.fr { position: relative; display: grid; grid-template-columns: minmax(0,1fr) 14rem; gap: 0 2.2rem; align-items: center; }
.fr + .fr::before, .lane + .lane .fr:first-child::before { content: ""; position: absolute; left: 1.5rem;
  border-left: 2px solid var(--wire); }
.fr + .fr::before { top: -1.3rem; height: 1.3rem; }
.lane + .lane .fr:first-child::before { top: -2.9rem; height: 2.9rem; }
.nd { position: relative; border: 1px solid var(--line); border-radius: 10px; padding: .7rem .95rem; background: var(--surface);
      font-size: .88rem; line-height: 1.5; }
.fr + .fr > .nd::before, .lane + .lane .fr:first-child > .nd::before { content: ""; position: absolute;
  left: calc(1.5rem - 5px); top: -7px; border: 5px solid transparent; border-top: 6px solid var(--wire); border-bottom: 0; }
.fr + .fr > .nd.q::before, .lane + .lane .fr:first-child > .nd.q::before { left: calc(1.5rem - 7px); }
.nd b { font-weight: 600; }
.nd .d, .blk .d { color: var(--muted); }
.nd.q { border-left: 3px solid var(--accent); }
.nd.calc { background: var(--soft); }
.nd.out { border-color: var(--accent); }
.par { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: .45rem; margin-top: .45rem; }
.par > div { border: 1px dashed var(--line); border-radius: 8px; padding: .45rem .6rem; font-size: .82rem; background: var(--surface); }
.exit { position: relative; font-size: .8rem; line-height: 1.45; border: 1px solid var(--line); border-radius: 8px;
        padding: .45rem .7rem; color: var(--muted); background: var(--surface); }
.exit::before { content: ""; position: absolute; right: calc(100% + 1px); top: 50%; width: 2.2rem; border-top: 2px dashed var(--wire); }
.exit.stop { color: var(--neg); border-color: color-mix(in srgb, var(--neg) 45%, transparent); }
.exit.go { color: var(--pos); border-color: color-mix(in srgb, var(--pos) 45%, transparent); }

/* architecture diagrams: subgrid rows line up the same stage across models */
.archs { display: grid; gap: 1rem; grid-template-rows: repeat(8, auto); margin-bottom: 1rem; }
.archs.n2 { grid-template-columns: repeat(2, minmax(0,1fr)); }
.archs.n3 { grid-template-columns: repeat(3, minmax(0,1fr)); }
.arch { grid-row: span 8; display: grid; grid-template-rows: subgrid; row-gap: 1.1rem; background: var(--surface);
        border: 1px solid var(--line); border-radius: var(--r); box-shadow: var(--shadow); padding: 1.4rem 1.4rem 1.1rem; }
.arch .ah .ct { margin: .1rem 0 .15rem; }
.blk { position: relative; border: 1px solid var(--line); border-radius: 10px; padding: .6rem .8rem; background: var(--bg);
       font-size: .84rem; line-height: 1.45; }
.blk .sl { display: block; font-size: .62rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
           color: var(--muted); margin-bottom: .2rem; }
.blk.chg { border-color: var(--accent); }
.blk.chg .sl { color: var(--accent); }
.blk + .blk::before { content: ""; position: absolute; left: calc(50% - 1px); top: calc(-1.1rem - 1px); height: 1.1rem;
  border-left: 2px solid var(--wire); }
.blk + .blk::after { content: ""; position: absolute; left: calc(50% - 5px); top: -7px; border: 5px solid transparent;
  border-top: 6px solid var(--wire); border-bottom: 0; }
.blk .par > div { background: var(--surface); }
.af { border-top: 1px solid var(--line); padding-top: .75rem; font-size: .8rem; color: var(--muted);
      font-variant-numeric: tabular-nums; }
.af b { color: var(--text); }
.key { display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; font-size: .84rem; color: var(--muted); margin: .2rem 0 1rem; }
.key i { display: inline-block; width: .9rem; height: .9rem; border-radius: 4px; border: 1.5px solid var(--line);
         vertical-align: -2px; margin-right: .4rem; }
.key i.chg { border-color: var(--accent); }
.key i.q { border-left: 3px solid var(--accent); }
.key i.calc { background: var(--soft); }
.key i.stop { border-color: var(--neg); }
.key i.go { border-color: var(--pos); }
.mt { width: 100%; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; }
.mt, .mt th, .mt td { border: 0; }
.mt th, .mt td { padding: .5rem .6rem; border-top: 1px solid var(--line); text-align: right; }
.mt th:first-child, .mt td:first-child { text-align: left; padding-left: 0; }
.mt thead th { border-top: 0; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
               color: var(--muted); }
.mt .best { color: var(--accent); font-weight: 600; }
.card p.small { font-size: .86rem; margin: .2rem 0 .8rem; }

.foot { border-top: 1px solid var(--line); margin-top: 3rem; padding-top: 1.2rem; color: var(--muted); font-size: .82rem; }
.foot b { color: var(--text); font-weight: 600; }

@media (max-width: 768px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: calc(var(--nav) + 2rem) 1rem 1.5rem; }
  :has(> img[data-testid="stHeaderLogo"])::after { display: none; }
  [data-testid="stTopNavLink"] { margin: 0 .45rem; }
  [class*="st-key-card"], .card { padding: 1.25rem 1rem; }
  .summary { grid-template-columns: auto minmax(0,1fr); }
  .figs { grid-column: 1 / -1; border-left: 0; padding-left: 0; gap: 1.6rem; }
  .op { grid-template-columns: 2rem minmax(0,1fr); }
  .op .r, .op .k { grid-column: 2; }
  .sent { grid-template-columns: 1fr; gap: .4rem; }
  .sent .n { width: 2.3rem; }
  .steps, .grid3 { grid-template-columns: 1fr; }
  .stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .st-key-trace { position: static; max-height: none; }
  .gt { grid-template-columns: 2rem minmax(0,1fr); }
  .gt .hdr { display: none; }
  .gt > div { border-top: 0; padding: .15rem 0; grid-column: 2; }
  .gt .gn { grid-column: 1; }
  .gt .gn, .gt .gname { padding-top: .9rem; border-top: 1px solid var(--line); }
  .gt .gh { grid-column: 1 / -1; }
  .lane { grid-template-columns: 1fr; gap: .9rem; }
  .fr { grid-template-columns: 1fr; }
  .exit { margin: .45rem 0 0 2.4rem; }
  .exit::before, .lane + .lane .fr:first-child::before, .lane + .lane .fr:first-child > .nd::before { display: none; }
}
@media (max-width: 1100px) { .archs.n3 { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 700px) { .archs.n2, .archs.n3 { grid-template-columns: 1fr; } }
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


def plural(n: int, word: str) -> str:
    return f'{word}{"s" * (n != 1)}'


# ----------------------------------------------------------------------------------------------- results
def summary_html(records: list[dict]) -> str:
    ops = [t for r in records for t in r["tuples"] if t["sentiment"]]
    kept = sum(r["gate"]["about_laptop"] for r in records)
    figs = "".join(f'<div><div class="v">{v}</div><div class="l">{esc(l)}</div></div>' for v, l in [
        (len(records), plural(len(records), "sentence")), (kept, "passed the gates"),
        (len(ops), plural(len(ops), "opinion"))])
    figs = f'<div class="figs">{figs}</div>'
    if not ops:
        return (f'<div class="card summary"><div class="face">{icon("sentiment_neutral", "muted")}</div><div>'
                f'<div class="t">No laptop opinion found</div><div class="muted">Every sentence was stopped by a gate or '
                f'carries no opinion.</div></div>{figs}</div>')
    score = 1 + sum(t["expected_star"] for t in ops) / len(ops)
    mood = "positive" if score >= 3.5 else "negative" if score < 2.5 else "neutral"
    full = int(score + 0.5)
    return (f'<div class="card summary"><div class="face {MOOD[mood]}">{icon(FACE[mood], MOOD[mood])}</div><div>'
            f'<div class="t">Mostly {mood}</div><div><span class="s">{"★" * full}{"☆" * (5 - full)}</span> '
            f'<span class="muted">{score:.1f} out of 5</span></div></div>{figs}</div>')


def opinion_html(t: dict) -> str:
    m = MOOD[t["sentiment"]]
    sub = (f'<span class="sub">model said {breakable(t["model_category"])}</span>'
           if t["model_category"] != t["aspect"] else "")
    source = "" if t["source"] == "model" else f'<span class="tag on">{esc(t["source"])}</span>'
    return (f'<div class="op">{icon(FACE[t["sentiment"]], m)}<div class="a"><span class="asp">{breakable(t["aspect"])}</span>'
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
    return (f'<div class="sent{off}"><div class="n">{r["sentence_index"] + 1:02d}</div><div>'
            f'<div class="x">{esc(r["sentence"])}</div>{body}<div class="brief">{esc(brief(r["tuples"]))}</div></div></div>')


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
        md('<div class="eyebrow">Verbose</div><div class="ct">Trace</div>')
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
    md('<div class="sec"><div class="eyebrow">Breakdown</div></div>')
    main, side = st.columns([1.55, 1], gap="large") if verbose else (st.container(), None)
    with main:
        md(summary_html(records))
        with st.container(key=f"card_results_{key}"):
            t_sent, t_out, t_table = st.tabs(["Sentences", "Output", "Aspect table"])
            with t_sent:
                reviews = list(dict.fromkeys((r["product"], r["review_id"]) for r in records))
                for p, rid in reviews:
                    rs = [r for r in records if (r["product"], r["review_id"]) == (p, rid)]
                    if len(reviews) > 1:
                        md(f'<div class="rh">{esc(p)} <span class="muted">review '
                           f'{esc(rid + 1 if isinstance(rid, int) else rid)}</span></div>')
                    md("".join(sentence_html(r) for r in rs))
            with t_out:
                st.caption("The brief's format, one line per sentence.")
                st.code("\n".join(f"{r['sentence']}\n  {brief(r['tuples'])}" for r in records), language="text")
                rows = opinion_rows(records)
                c1, c2 = st.columns(2)
                c1.download_button("Download CSV", rows.to_csv(index=False), "opinions.csv", "text/csv", key=f"csv_{key}",
                                   icon=":material/download:", width="stretch")
                c2.download_button("Download JSON", output_json(records), "output.json", "application/json",
                                   key=f"json_{key}", icon=":material/data_object:", width="stretch")
            with t_table:
                table = aspect_table(opinion_rows(records))
                if table.empty:
                    st.info("No opinions, so there is no aspect table yet.")
                for p, t in table.groupby("product", sort=False):
                    st.markdown(f"**{product or p}**")
                    st.dataframe(t.drop(columns="product"), hide_index=True, width="stretch",
                                 column_config={"rating": st.column_config.NumberColumn("rating", format="%.1f")})
                st.caption("Rating: 1 plus the mean expected star over the reviews that mention the category, one vote "
                           "per review.")
    if side is not None:
        with side:
            show_trace(records, load_pipeline().model.tau)


def hero(eyebrow: str, title: str, lede: str) -> str:
    return (f'<div class="eyebrow">{esc(eyebrow)}</div><div class="display" role="heading" aria-level="1">{esc(title)}'
            f'<span class="dot">.</span></div><p class="lede">{esc(lede)}</p>')


def page_header(eyebrow: str, title: str, lede: str, key: str, pipe: Pipeline) -> bool:
    left, right = st.columns([3, 1.4], vertical_alignment="bottom")
    with left:
        md(hero(eyebrow, title, lede))
    m = pipe.model
    with right.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", key="hero_tools"):
        verbose = st.toggle("Verbose", key=f"verbose_{key}",
                            help="Open the trace panel: preprocessing, gates, raw model output and post-processing checks")
        st.markdown(f'<span class="chip"><span class="live"></span>bge_detr · epoch {m.epoch} · {m.device.type}</span>',
                    unsafe_allow_html=True, width="content")
    return verbose


def section(eyebrow: str, title: str) -> None:
    md(f'<div class="sec"><div class="eyebrow">{esc(eyebrow)}</div><div class="ct">{esc(title)}</div></div>')


# ----------------------------------------------------------------------------------------------- pages
def analyse_page() -> None:
    pipe = load_pipeline()
    verbose = page_header("Review intelligence / 01", "Analyse a review",
                          "Discover what people really think about a laptop, one detail at a time. Paste a review to see "
                          "each sentence's aspects, sentiment, a 1 to 5 rating and a confidence.", "one", pipe)
    st.session_state.setdefault("text", EXAMPLES["Mixed review"])

    def pick() -> None:
        if st.session_state.example:
            st.session_state.text = EXAMPLES[st.session_state.example]
            st.session_state.pop("one", None)

    with st.container(key="card_compose"):
        st.pills("Try an example", list(EXAMPLES), key="example", on_change=pick)
        c1, c2 = st.columns([2, 1], gap="large")
        c1.text_area("Review", key="text", height=160, placeholder="Type or paste a laptop review")
        c1.caption("Write naturally. A few sentences give the clearest breakdown.")
        product = c2.text_input("Product", value="Apple Mac mini", placeholder="e.g. Apple Mac mini")
        c2.caption("Optional. Names the aspect-sentiment table.")
        run = st.button("Analyse review →", type="primary")
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
    pipe = load_pipeline()
    verbose = page_header("Review intelligence / 02", "Analyse a file",
                          "Upload a CSV with one review per row. Rows of one product share an aspect-sentiment table.",
                          "batch", pipe)
    with st.container(key="card_batch"):
        f = st.file_uploader("CSV file", type=["csv"])
        if f is None and st.button("Use the sample file", help="Six reviews of two laptops", icon=":material/table:"):
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
            st.caption("No file yet. Upload a CSV, or try the sample file.")
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
        if st.button("Analyse file →", type="primary"):
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
    chosen = (st.columns([1, 2])[0].selectbox("Product", ["All products", *products]) if len(products) > 1
              else products[0])
    shown = records if chosen == "All products" else [r for r in records if r["product"] == chosen]
    show_results(shown, "" if chosen == "All products" else chosen, verbose, "batch")


def par(*cells: str) -> str:
    """Side-by-side boxes inside a node or block: parallel heads, parallel checks."""
    return '<div class="par">' + "".join(f"<div>{c}</div>" for c in cells) + "</div>"


# ----------------------------------------------------------------------------------------------- about: gates and flow
GATE_GROUPS = [
    ("Hard stops", "any failure stops the sentence", [
        ("tokens", "The bge tokenizer finds at least one word piece that contains a letter.",
         'stops "!!!", "???", "123"', "1 word piece or more"),
        ("filler", "At least one word is outside the filler list of greetings, thanks and stop words (hello, thanks, "
                   "cheers, everyone, ok, the, it, ...).", 'stops "Hello.", "Thanks everyone!"', "word list"),
        ("off-topic", "Another object is what the sentence is about: its subject, or what is loved, hated, bought or "
                      "recommended (phones, TVs, printers, food, restaurants, hotels, flights, films, cars, clothes, "
                      "weather). Skipped when the laptop is named or the sentence compares (than, versus, better). A "
                      'passing mention is fine: "great for watching movies" is about the laptop.',
         'stops "The pizza was cold", "I love my new phone"', "sentence pattern"),
    ]),
    ("Strong pass", "passing skips the evidence gates", [
        ("laptop named", "The laptop word list matches: laptop, computer, notebook, MacBook, Chromebook, ThinkPad, XPS, "
                         "or a brand (Apple, Dell, Lenovo, HP, Asus, ...). Failing it stops nothing: the evidence decides.",
         'passes "It is the worst laptop ever."', "word list"),
    ]),
    ("Evidence", "domain must pass, and aspect terms or category", [
        ("aspect terms", 'The sentence mentions one of the 22 laptop entities, by its word list ("trackpad" is MOUSE, '
                         '"RAM" is MEMORY) or by word similarity: each content word is embedded on its own and compared '
                         "with the 22 entity labels.", '"build" is HARDWARE at 0.85; "fast" at 0.67 is not',
         f"cosine ≥ {domain_gate.WORD_FLOOR:.2f}"),
        ("domain", "The sentence embedding (pretrained bge-small, [CLS]) is closer to the 22 laptop labels than to 22 "
                   "other topics: restaurant, food, hotel, flights, car, phone, TV, film, music, sports, ...",
         "margin = best laptop cosine - best other cosine", f"margin ≥ {domain_gate.MARGIN:+.2f}"),
        ("category", "The closest laptop label is close enough: the sentence falls inside one of the 22 categories.",
         "best laptop cosine", f"cosine ≥ {domain_gate.CATEGORY_FLOOR:.2f}"),
    ]),
]


def gates_html() -> str:
    out = ['<div class="gt">', *(f'<div class="hdr">{h}</div>' for h in ("#", "Gate", "What it checks", "Example",
                                                                          "Passes at"))]
    n = 0
    for group, rule, gates in GATE_GROUPS:
        out.append(f'<div class="gh"><b>{group}</b><span>{rule}</span></div>')
        for name, what, example, threshold in gates:
            n += 1
            out += [f'<div class="gn">{n:02d}</div>', f'<div class="gname">{name}</div>', f"<div>{esc(what)}</div>",
                    f'<div class="gx">{esc(example)}</div>', f'<div class="gthr">{esc(threshold)}</div>']
    return "".join(out) + "</div>"


def node(body: str, kind: str = "", branch: str = "", branch_kind: str = "stop") -> str:
    """One step of the flow chart; `branch` is a side exit (a stop, a pass, or another route)."""
    side = f'<div class="exit {branch_kind}">{branch}</div>' if branch else ""
    return f'<div class="fr"><div class="nd {kind}">{body}</div>{side}</div>'


def lane(num: int, title: str, module: str, nodes: list[str]) -> str:
    return (f'<div class="lane"><div class="lh"><div class="num">{num:02d}</div><div class="ct">{title}</div>'
            f'<div class="mono">{module}</div></div><div class="spine">{"".join(nodes)}</div></div>')


def flow_html() -> str:
    g, p = domain_gate, post_process
    return "".join([
        lane(1, "Preprocess", "src/preprocess.py", [
            node('<b>Review text</b> <span class="d">as typed or pasted, any length</span>'),
            node('<b>Normalise</b><div class="d">HTML entities and tags, Unicode NFKC, curly quotes made straight, '
                 'invisible characters, letter runs capped at three ("soooooo" becomes "sooo"). Case, emoji and '
                 "spelling stay.</div>"),
            node('<b>Split into sentences</b><div class="d">at . ! ? emoticons and line breaks, never after an '
                 "abbreviation (e.g., Mr., approx.) or inside a decimal</div>"),
            node("<b>Tag every sentence</b>" + par("<b>nuances</b><br>negation, contrast, comparison, question",
                                                   "<b>entities</b><br>named by the 22 word lists",
                                                   "<b>off-topic</b><br>another object it is about",
                                                   "<b>filler</b><br>greetings or thanks only",
                                                   "<b>context</b><br>the previous and next sentence")),
        ]),
        lane(2, "Gates", "src/domain_gate.py", [
            node('<b>01 tokens</b><div class="d">does the bge tokenizer find a word piece with a letter?</div>', "q",
                 'no: {NULL#NULL}, "no real word"'),
            node('<b>02 filler</b><div class="d">is any word outside the greetings, thanks and stop-word list?</div>',
                 "q", "no: {NULL#NULL}, filler only"),
            node('<b>03 off-topic</b><div class="d">is a phone, pizza, hotel, film ... the subject, or what is loved, '
                 "hated or bought, with no laptop named and no comparison?</div>", "q",
                 "yes: {NULL#NULL}, about that object"),
            node('<b>Mentions</b><div class="d">the word lists ("trackpad" is MOUSE), then every other content word '
                 f'against the 22 entity labels</div><span class="f">cos(word, entity label) ≥ {g.WORD_FLOOR:.2f}</span>',
                 "calc"),
            node('<b>04 laptop named</b><div class="d">laptop, computer, MacBook, ThinkPad, a brand ...?</div>', "q",
                 "yes: passes, straight to the model", "go"),
            node('<b>Sentence embedding</b><div class="d">pretrained bge-small [CLS], normalised; cosine to the 22 '
                 "laptop labels and to the 22 other topics</div>", "calc"),
            node("<b>Evidence</b>" + par(
                "<b>05 aspect terms</b><br>at least one mention",
                f'<b>06 domain</b><br><span class="f">max cos(laptop) - max cos(other) ≥ {g.MARGIN:+.2f}</span>',
                f'<b>07 category</b><br><span class="f">max cos(laptop) ≥ {g.CATEGORY_FLOOR:.2f}</span>')),
            node('<b>Rule</b><br><span class="f">domain AND (aspect terms OR category)</span>', "q",
                 "no: {NULL#NULL}, may still be merged (step 05)"),
        ]),
        lane(3, "Model", "src/model_inference.py<br>bge_detr, epoch 9", [
            node('<b>Input window</b><br><span class="f">[CLS] previous [SEP] sentence [SEP] next [SEP]</span>'
                 '<div class="d">token type 0 for the sentence, 1 for its context; a negation cue and up to 5 words after '
                 "it get a learned scope vector. Over 56 tokens: windows that overlap by a quarter.</div>"),
            node('<b>Encoder</b><div class="d">bge-small, 12 transformer layers: one 384-d state per token</div>'),
            node('<b>Memory</b><div class="d">the sentence tokens only, Linear 384 → 256 + LayerNorm</div>'),
            node('<b>16 opinion queries</b><div class="d">a 2-layer transformer decoder: the queries attend to each other '
                 "and to the words, so each one settles on one opinion</div>"),
            node("<b>Per query</b>" + par('<b>entity</b><br>softmax over 22 + "no opinion"',
                                          "<b>attribute</b><br>softmax over 9", "<b>star</b><br>softmax over 0 to 4")),
            node('<b>Category score</b><br><span class="f">score(E#A) = max over queries of p(E) · p(A)</span>'
                 '<div class="d">the star distribution comes from that query; long text takes the max over windows</div>',
                 "calc"),
        ]),
        lane(4, "Decide", "src/post_process.py", [
            node('<b>Detection</b><br><span class="f">score ≥ τ = 0.85</span>', "q",
                 f"nothing reaches τ: a sentence that mentions a part keeps scores ≥ {p.TAU_LOW:.2f}, otherwise no "
                 "opinion (step 05)", "alt"),
            node('<b>Expected star</b><br><span class="f">E = Σ k · p(k), k = 0 to 4</span>', "calc"),
            node(f'<b>{p.LEVEL:.0%} star interval</b><div class="d">the narrowest run of stars that holds at least '
                 f"{p.LEVEL:.0%} of the probability</div>", "calc"),
            node('<b>Sentiment brackets</b><br><span class="f">negative = p0 + p1 · neutral = p2 · positive = p3 + p4'
                 "</span>", "calc"),
            node('<b>Sentiment</b><div class="d">interval inside one bracket: that bracket. Otherwise the largest '
                 "bracket, but an interval from negative to positive stars with no bracket over 50% is a mixed opinion: "
                 "neutral.</div>", "q"),
            node('<b>Rating and confidence</b><br><span class="f">confidence = p(bracket) · star = round(E), kept inside '
                 f'the bracket · rating = star + 1</span><div class="d">high at {p.HIGH:.0%} or more with the interval '
                 f"inside one bracket, medium at {p.MEDIUM:.0%} or more, otherwise low</div>", "calc"),
            node('<b>Entity rule</b><br><span class="f">named OR (explicit target AND (review context OR whole laptop '
                 "OR top-3 label))</span>", "q", "no: the entity becomes NULL, {NULL#A}", "alt"),
        ]),
        lane(5, "Merge", "src/main.py", [
            node('<b>Still {NULL#NULL}?</b><div class="d">and no hard gate (01 to 03) stopped it: join it with the '
                 "previous and with the next sentence of the same review</div>", "q"),
            node('<b>Run again</b><div class="d">gates and model on the joined text. Opinions at τ or more that the '
                 "neighbour alone does not have belong to this sentence; the best-scoring join wins.</div>"),
            node('<b>No opinion</b><div class="d">each mentioned entity becomes {E#NULL}; nothing mentioned stays '
                 "{NULL#NULL}</div>"),
        ]),
        lane(6, "Output", "src/post_process.py", [
            node('<b>Tuples</b><br><span class="f">{E#A, sentiment}</span><div class="d">with the 1 to 5 rating, stars, '
                 "confidence and where it came from (model, second threshold or merge)</div>", "out"),
            node('<b>Aspect table</b><div class="d">per product and category: mentions by sentiment, and the reviews '
                 'that mention it</div><span class="f">rating = 1 + mean over reviews of each review\'s mean E</span>',
                 "out"),
        ]),
    ])


def how_page() -> None:
    md(hero("About the system / 03", "How it works",
            "A review passes through four steps. Each step can stop a sentence or change its answer, and the Verbose "
            "switch on the analysis pages shows every decision."))
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
    md('<div class="steps">' + "".join(
        f'<div class="card step"><div class="num">{i:02d}</div><div class="ct">{a}</div><p class="lead">{b}</p>'
        f'<p class="muted">{esc(c)}</p></div>' for i, (a, b, c) in enumerate(steps, 1)) + "</div>")

    section("Domain gates", "The seven gates")
    md(f'<div class="card">{gates_html()}<div class="rule"><b>Decision</b><br><span class="f">about laptop = hard gates '
       f"pass AND (laptop named OR (domain AND (aspect terms OR category)))</span><p class=\"muted\">A stopped sentence "
       f"gets {{NULL#NULL}} and the reason, and never reaches the model. If no hard gate stopped it, the merge step can "
       f"still join it with a neighbour.</p><p class=\"muted\">Fitted on laptop training sentences only: the domain and "
       f"category floors are the 1st percentile over the opinion sentences that reach gate 05 (each drops about 1% of "
       f"them); the word floor is the 99th percentile over the 300 most frequent training words the lists do not match."
       f"</p></div></div>")
    st.caption("On the SemEval-2016 test sets the gates keep 98.6% of laptop opinion sentences (565 of 573) and stop "
               "59.5% of restaurant sentences. This rule stops more restaurant sentences than a 2 of 3 vote (55.9%) or the "
               "margin alone (54.4%) at about the same laptop loss.")

    section("Entity rule", "When is the entity NULL?")
    with st.container(key="card_null"):
        st.markdown("The brief: *when there is no explicit mention of the entity or attribute, E or A takes the value "
                    "NULL*. The model always names an entity, so each opinion goes through five checks. The entity stays "
                    "when it is **named** in the sentence, or when the sentence has an **explicit target** (some laptop "
                    "part is mentioned) and one more check agrees: the entity is mentioned in a neighbouring sentence "
                    "(**review context**), it is the **whole laptop**, or it is among the sentence's three closest labels "
                    "(**label similarity**).")
        st.dataframe(pd.DataFrame(HOW_EXAMPLES, columns=["Sentence", "Output", "Why"]), hide_index=True, width="stretch")

    section("Legend", "Reading the output")
    md(f'<div class="card legend">{icon(FACE["positive"], "pos")}<div>Positive: 4 or 5 stars</div>'
       f'{icon(FACE["neutral"], "neu")}<div>Neutral: 3 stars, or a mixed opinion</div>'
       f'{icon(FACE["negative"], "neg")}<div>Negative: 1 or 2 stars</div>'
       f'<span class="ms" style="color:var(--star)">star</span><div>Rating from 1 to 5, shown only for opinions. It '
       f'stays inside its sentiment, so stars and face always agree.</div>'
       f'<span class="ms muted">verified</span><div>Confidence: the probability of the sentiment. <b>high</b> at 80% or '
       f'more with the 90% star interval inside one sentiment, <b>medium</b> at 60% or more, otherwise <b>low</b>.</div>'
       f'<span class="ms muted">block</span><div><span class="mono">{{NULL#NULL}}</span>: stopped by a gate, or no '
       f'opinion and nothing named. <span class="mono">{{LAPTOP#NULL}}</span>: named, but no opinion.</div></div>')

    section("In depth", "From review to tuple, step by step")
    md('<div class="key"><span><i class="q"></i>decision</span><span><i class="calc"></i>calculation</span>'
       '<span><i class="stop"></i>stops the sentence</span><span><i class="go"></i>passes it on</span>'
       '<span><i></i>another route</span></div>')
    md(f'<div class="card">{flow_html()}</div>')


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


# ----------------------------------------------------------------------------------------------- experiments
def blk(stage: str, body: str, changed: bool = False) -> str:
    return f'<div class="blk{" chg" if changed else ""}"><span class="sl">{stage}</span>{body}</div>'


def arch_html(kind: str, name: str, note: str, blocks: list[str], foot: str) -> str:
    return (f'<div class="arch"><div class="ah"><div class="eyebrow">{kind}</div><div class="ct">{name}</div>'
            f'<div class="muted small">{note}</div></div>{"".join(blocks)}<div class="af">{foot}</div></div>')


BGE = '<b>bge-small-en-v1.5</b><div class="d">12 transformer layers, 12 heads, feed-forward 1536, hidden 384</div>'
CONTEXT = ('<span class="f">[CLS] prev [SEP] sentence [SEP] next [SEP]</span><div class="d">context as token type 1, '
           "dropped 30% of the time in training; a negation cue and 5 words after it get a learned scope vector</div>")
TWEAKS = '<div class="d">layer-wise learning-rate decay 0.9; 25% extra long, multi-sentence inputs</div>'
ARCHS = {
    "bge_base": arch_html("Baseline", "bge_base", "The pretrained encoder with a linear head.", [
        blk("Input", '<span class="f">[CLS] sentence [SEP]</span><div class="d">up to 56 tokens; longer text in '
                     "overlapping windows</div>"),
        blk("Encoder", BGE),
        blk("Read-out", '<b>[CLS] vector</b><div class="d">one 384-d vector per sentence</div>'),
        blk("Heads", par("<b>detection</b><br>Linear 384 → 80, sigmoid", "<b>stars</b><br>Linear 384 → 80 × 5, softmax")),
        blk("Training", '<div class="d">BCE for detection; cross-entropy on smoothed star targets</div>'),
        blk("Output", '<div class="d">the 80 training categories with p ≥ τ = 0.20, a star 0 to 4 each</div>'),
    ], "<b>33.2M</b> parameters · test joint F1 <b>36.0</b>"),
    "bge_detr": arch_html("Selected", "bge_detr", "The tweaked encoder with a DETR-style set head.", [
        blk("Input", CONTEXT, True),
        blk("Encoder", BGE + TWEAKS, True),
        blk("Read-out", '<b>Memory</b><div class="d">the sentence tokens only, Linear 384 → 256 + LayerNorm</div>', True),
        blk("Heads", '<b>16 opinion queries</b><div class="d">2-layer transformer decoder (256-d, 8 heads, feed-forward '
                     "1024)</div>" + par("<b>entity</b><br>22 + no opinion", "<b>attribute</b><br>9",
                                         "<b>star</b><br>0 to 4"), True),
        blk("Training", '<div class="d">Hungarian matching of queries to gold opinions; cross-entropy, "no opinion" '
                        "weighted 0.1</div>", True),
        blk("Output", '<span class="f">score(E#A) = max over queries of p(E) · p(A)</span><div class="d">any of the 198 '
                      "pairs, τ = 0.85</div>", True),
    ], "<b>35.4M</b> parameters (head 2.2M) · test joint F1 <b>47.7</b>"),
    "lfm_base": arch_html("Larger encoder", "lfm_base", "A hybrid convolution and attention encoder, linear head.", [
        blk("Input", '<span class="f">&lt;|startoftext|&gt; sentence</span><div class="d">up to 56 tokens, no end token'
                     "</div>", True),
        blk("Encoder", '<b>LFM2.5-Encoder-230M</b><div class="d">14 blocks: 8 gated short convolutions (3 tokens, '
                       "centred) and 6 grouped-query attention (16 query / 8 key-value heads, RoPE); SwiGLU feed-forward "
                       "2560; hidden 1024</div>", True),
        blk("Read-out", '<b>mean of the sentence tokens</b><div class="d">one 1024-d vector (no CLS objective)</div>',
            True),
        blk("Heads", par("<b>detection</b><br>Linear 1024 → 80, sigmoid",
                         "<b>stars</b><br>Linear 1024 → 80 × 5, softmax")),
        blk("Training", '<div class="d">BCE for detection; cross-entropy on smoothed star targets</div>'),
        blk("Output", '<div class="d">the 80 training categories with p ≥ τ = 0.40, a star 0 to 4 each</div>'),
    ], "<b>229.7M</b> encoder parameters · test joint F1 <b>46.1</b>"),
    "bge_tweaked": arch_html("Tweaked", "bge_tweaked", "The same tweaked encoder as bge_detr, classifier heads.", [
        blk("Input", CONTEXT, True),
        blk("Encoder", BGE + TWEAKS, True),
        blk("Read-out", '<b>category attention</b><div class="d">one query per category, started from the embedding of '
                        "its name, attends over the sentence tokens: 80 vectors</div>", True),
        blk("Heads", par("<b>detection</b><br>per category, + E × A factorised: s(E) + s(A) + r(E#A)",
                         "<b>opinion gate</b><br>p(the sentence holds an opinion)",
                         "<b>stars</b><br>MLP 384 → 256 → 5"), True),
        blk("Training", '<div class="d">asymmetric loss for detection; focal, class-balanced star loss</div>', True),
        blk("Output", '<span class="f">p(E#A) = p(gate) · p(E#A | opinion)</span><div class="d">τ = 0.60</div>', True),
    ], "test joint F1 <b>36.7</b> · best star accuracy (79.8%) and empty-sentence accuracy (83%)"),
    "custom": arch_html("Custom architecture", "3 head transformer classifier",
                        "One sentence vector read by three heads.", [
        blk("Input", '<span class="f">[CLS] sentence [SEP]</span><div class="d">the sentence alone, no context; up to 128 '
                     "tokens</div>"),
        blk("Encoder", BGE),
        blk("Read-out", '<b>mean pooling</b><div class="d">over all tokens, dropout 0.1</div>', True),
        blk("Heads", par("<b>1 category</b><br>Linear 384 → 22", "<b>2 attribute</b><br>Linear 384 → 9",
                         "<b>3 pair</b><br>per E#A (22 × 9): presence MLP 384 → 384 → 198, rating MLP 384 → 384 → "
                         "198 × 5"), True),
        blk("Training", '<div class="d">BCE for category, attribute and presence; soft-ordinal cross-entropy for the '
                        "rating (σ 0.5)</div>", True),
        blk("Output", '<span class="f">r = Σ k · p(k)</span><div class="d">presence ≥ threshold is an opinion; r under lo '
                      "negative, over hi positive, else neutral; 6 thresholds tuned on validation</div>", True),
    ], "<b>34.1M</b> parameters (heads 0.76M) · triple F1 <b>57.4</b> on its own fold"),
}

CANDIDATES = pd.DataFrame([
    ("bge_detr (selected)", 9, "SemEval, 808", 47.7, 88.6, 71.2, 0.35, 0.879, 0.984),
    ("lfm_base", 10, "SemEval, 808", 46.1, 84.0, 62.9, 0.52, 0.806, 0.962),
    ("bge_tweaked", 9, "SemEval, 808", 36.7, 86.0, 72.2, 0.37, 0.872, 0.895),
    ("bge_base", 9, "SemEval, 808", 36.0, 82.8, 65.8, 0.53, 0.797, 0.932),
    ("Custom 3 head transformer classifier", 14, "own fold, 390", 57.4, 87.3, 71.5, 0.452, 0.860, None),
], columns=["run", "epoch", "test set", "joint F1", "sentiment acc", "sentiment macro-F1", "star MAE", "star QWK",
            "detection AUC"])

CUSTOM_TEST = [("Triple precision / recall / F1", "0.607 / 0.545 / 0.574"), ("Triple F1, default thresholds", "0.507"),
               ("Triple F1: negative / neutral / positive", "0.503 / 0.214 / 0.637"), ("Pair F1", "0.629"),
               ("Category F1, micro / macro", "0.781 / 0.256"), ("Attribute F1, micro / macro", "0.777 / 0.652"),
               ("Polarity accuracy / macro-F1", "0.873 / 0.715"), ("Rating accuracy / within ±1", "0.829 / 0.864"),
               ("Rating MAE / RMSE", "0.452 / 0.928"), ("Rating Spearman / QWK", "0.794 / 0.860"),
               ("Calibration error (ECE)", "0.046"), ("Sentence exact match", "0.415")]
TRANSFER = [("Sentiment accuracy", ("87.3", "82.8", "88.6"), 2), ("Sentiment macro-F1", ("71.5", "65.8", "71.2"), 0),
            ("Star MAE", ("0.452", "0.53", "0.35"), 2), ("Star within ±1", ("86.4", "86.3", "90.9"), 2),
            ("Star QWK", ("0.860", "0.797", "0.879"), 2), ("Star Spearman", ("0.794", "0.759", "0.811"), 2),
            ("Calibration error (ECE)", ("0.046", "0.073", "0.054"), 0),
            ("Test sentences / gold opinions", ("390 / 543", "808 / 801", "808 / 801"), None),
            ("Sentences without an opinion", ("0%", "29.1%", "29.1%"), None)]


def metric_table(head: list[str], rows: list[tuple[str, tuple, int | None]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr><td>{name}</td>" + "".join(f'<td{" class=best" if i == best else ""}>{v}</td>'
                                                    for i, v in enumerate(vals)) + "</tr>" for name, vals, best in rows)
    return f'<table class="mt"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def models_page() -> None:
    md(hero("Model experiments / 04", "Models and experiments",
            "Five models were trained on the SemEval-2016 laptop reviews: four in the experiments notebook and a custom "
            "3 head transformer classifier. bge_detr at epoch 9 was selected on validation and runs this app."))
    md('<div class="stats">' + "".join(f'<div class="card"><div class="v">{v}</div><div class="l">{l}</div></div>'
                                       for v, l in [("47.7", "test joint F1 of (E#A, sentiment)"),
                                                    ("88.6%", "sentiment accuracy"),
                                                    ("0.35", "star error (MAE, 0 to 4 scale)"),
                                                    ("0.984", "detection AUC")]) + "</div>")

    section("Architectures", "Block by block")
    md('<div class="key"><span><i class="chg"></i>differs from bge_base</span><span>Read each diagram top to bottom. In a '
       "row, the same stage sits at the same height, so the blocks compare side by side.</span></div>")
    md(f'<div class="archs n2">{ARCHS["bge_base"]}{ARCHS["bge_detr"]}</div>')
    md(f'<div class="archs n3">{ARCHS["lfm_base"]}{ARCHS["bge_tweaked"]}{ARCHS["custom"]}</div>')

    section("Comparison", "The candidates")
    with st.container(key="card_candidates"):
        st.dataframe(CANDIDATES, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format=f) for c, f in [
                         ("joint F1", "%.1f"), ("sentiment acc", "%.1f"), ("sentiment macro-F1", "%.1f"),
                         ("star MAE", "%.2f"), ("star QWK", "%.3f"), ("detection AUC", "%.3f")]})
        st.caption("Each run at its best validation epoch. SemEval: the official test set, 808 sentences, 29% without an "
                   "opinion. The custom classifier was scored on its own held-out fold: 390 sentences that all hold an "
                   "opinion, so it can never fire on an empty sentence and its joint F1 (triple F1) does not compare. "
                   "Compare it on the sentiment and star columns; detection AUC was not computed for it.")

    section("Custom architecture", "3 head transformer classifier")
    c1, c2 = st.columns([1, 1.15], gap="medium")
    c1.markdown('<div class="card"><div class="ct">Its own test</div><p class="muted small">Fold 0: 390 sentences, 543 '
                "gold triples. Best epoch 14 of 15, validation triple F1 0.582.</p>"
                + metric_table(["Metric", "Value"], [(m, (v,), None) for m, v in CUSTOM_TEST]) + "</div>",
                unsafe_allow_html=True)
    c2.markdown('<div class="card"><div class="ct">On the metrics that transfer</div><p class="muted small">Defined the '
                "same way on both test sets, whatever their share of empty sentences. Best in blue.</p>"
                + metric_table(["Metric", "Custom", "bge_base", "bge_detr"], TRANSFER)
                + '<p class="muted small" style="margin-top:.8rem">bge_base is its closest relative: the same encoder, '
                "flat heads, no context. bge_detr ties or beats the custom classifier on star quality while facing the "
                "harder test set.</p></div>", unsafe_allow_html=True)

    section("End to end", "The system on top of the model")
    with st.container(key="card_system"):
        st.markdown(SYSTEM_RESULTS)
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        section("Selection", "Why bge_detr")
        with st.container(key="card_why"):
            st.markdown(
                "- **Best test joint F1**, and it wins 10 of 15 extended metrics, including every per-sentiment F1.\n"
                "- **The set head is what lifts it.** bge_tweaked shares its backbone; swapping in the set head moves "
                "joint F1 from 36.7 to 47.7.\n"
                "- **The custom classifier's 57.4 is not a lead.** It is scored on sentences that all hold an opinion; "
                "on the metrics that transfer, bge_detr is equal or better.\n"
                "- **Stable, not decaying.** Test F1 is flat from epoch 7 to 10 (48.4 to 48.9).\n"
                "- **Small.** 35.4M parameters against 229.7M for the LFM encoder, and about 0.05 s per sentence on a "
                "CPU.")
    with c2:
        section("Sources", "Further reading")
        with st.container(key="card_reading"):
            st.markdown("- `models/notebooks/ABSA_model_selection_report.md`: the decision, learning curves, overfitting "
                        "and the comparison with the custom classifier\n"
                        "- `models/notebooks/ABSA_experiments_base_tweak.ipynb`: the bge and LFM experiments\n"
                        "- `models/notebooks/ABSA_colab.ipynb`: the custom architecture, 3 head transformer classifier\n"
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
    st.set_page_config(page_title=BRAND, page_icon=":material/laptop_mac:", layout="wide",
                       initial_sidebar_state="collapsed")
    md(CSS)
    st.logo(LOGO, size="large")
    st.navigation([
        st.Page(analyse_page, title="Analysis", url_path="analyse", default=True),
        st.Page(batch_page, title="Batch CSV", url_path="batch"),
        st.Page(how_page, title="About System", url_path="how-it-works"),
        st.Page(models_page, title="Model Experiments", url_path="models"),
    ], position="top").run()
    md(f'<div class="foot"><b>{BRAND}</b> · See past the reviews. Identify the user sentiments.</div>')
    
    md(f'<div class="foot"><b>Creators / Group: </b> · RUPESH GHIMIRE · SANDESH PRASAD PAUDEL </div>')



main()
