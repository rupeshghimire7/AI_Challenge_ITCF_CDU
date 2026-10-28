"""Data nuance checker: what a piece of review text contains that the pipeline has to handle.

ARTEFACTS  formatting that preprocess.normalise() repairs. Clean text has none (scripts/clean_data.py checks every
           cleaned sentence). The patterns live here and normalise() reuses them, so detection and repair cannot drift.
NUANCES    linguistic phenomena that stay in the text: the plan D5.1 slices that can be read off the text alone
           (implicit / multi / rare / unseen / empty need gold labels or the D6.1 lexicon). Rule-based candidates until
           D5.3 measures their precision on S1; the UI shows them as badges.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable

# ----------------------------------------------------------------------------------------------- artefacts
BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
TAG = re.compile(r"</?[A-Za-z][^<>]{0,200}>")  # "<p>", "<a href=..>"; leaves "<3" and "< $200" alone
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")  # tab and newline included: normalise() turns them into spaces
INVISIBLE = re.compile("[­​‌‎‏‪-‮⁠-⁤⁦-⁩﻿]")  # keeps U+200D (emoji ZWJ)
PUNCT_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
                           "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
                           "‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-", "—": " - ", "―": " - "})
TYPOGRAPHIC = re.compile("[" + "".join(map(chr, PUNCT_MAP)) + "]")  # any character PUNCT_MAP rewrites
# List markers opening a sentence ("- light", "* great screen", "2) Blue screen"). "+" (a "pro" marker) and ">"
# ("> 8 hours") carry meaning and are kept.
LIST_MARKER = re.compile(r"^(?:(?:[-*•‣◦▪●]|\d{1,2}[.)])\s+)+")
LETTER_RUN = re.compile(r"([^\W\d_])\1{3,}")  # "soooooo"; digits are never touched ("$1000")
PUNCT_RUN = re.compile(r"([!?.,;:*~=_+#-])\1{3,}")  # "!!!!!", "....."
_SPACE_ODD = re.compile(r"\s{2,}|[^\S ]|^\s|\s$")
_CONTROL_ODD = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")  # control characters other than whitespace

ARTEFACTS: dict[str, Callable[[str], bool]] = {
    "html_entity": lambda t: html.unescape(t) != t,
    "html_tag": lambda t: bool(BR.search(t) or TAG.search(t)),
    "unicode_compat": lambda t: unicodedata.normalize("NFKC", t) != t,  # full-width forms, ligatures, no-break space
    "typographic_punct": lambda t: bool(TYPOGRAPHIC.search(t)),  # curly quotes, long dashes, backtick
    "invisible_char": lambda t: bool(INVISIBLE.search(t) or _CONTROL_ODD.search(t)),
    "irregular_space": lambda t: bool(_SPACE_ODD.search(t)),
    "list_marker": lambda t: bool(LIST_MARKER.match(" ".join((t.translate(PUNCT_MAP) if TYPOGRAPHIC.search(t) else t).split()))),
    "elongation": lambda t: bool(LETTER_RUN.search(t)),  # also the D2.4 / D6.6 `elongated` flag
    "punct_run": lambda t: bool(PUNCT_RUN.search(t)),
}

# ----------------------------------------------------------------------------------------------- nuances (D5.1)
_AUX = r"is|are|am|was|were|do|does|did|can|could|will|would|should|has|have"
_SUBJECT = r"i|you|he|she|it|we|they|this|that|these|those|there|the|a|an|my|your|its|their|anyone|anybody|someone"
NUANCES: dict[str, re.Pattern] = {
    "negation": re.compile(r"\b(?:not|no|never|nothing|none|nobody|neither|nor|without|hardly|barely|cannot)\b|n't\b", re.I),
    "contrast": re.compile(r"\b(?:but|however|although|though|yet|whereas|except|despite)\b", re.I),
    "comparison": re.compile(r"\b(?:than|compared?|comparison|versus|vs\.?|unlike|better|worse|best|worst)\b|\blike my\b"
                             r"|\bmy (?:old|previous|last|other)\b", re.I),
    # lexicon-free part of the D5.1 rule; "positive word + negative event" needs the D6.2 lexicon
    "sarcasm_cand": re.compile(r"\byeah,? right\b|\bjust (?:great|perfect|wonderful|fantastic|brilliant|lovely)\b|\bso-called\b"
                               r"|(?<!\w)[\"'][^\"'\s]+(?: [^\"'\s]+)?[\"'](?!\w)", re.I),  # scare quotes: 'premium', "new"
    "numeric": re.compile(r"\$\s?\d|\b\d+(?:\.\d+)?\s?(?:gb|tb|mb|ghz|mhz|hours?|hrs?|mins?|minutes|lbs?|pounds|inch(?:es)?|mah"
                          r"|watts?|%)(?![a-z])|\b\d+(?:\.\d+)?\"", re.I),
    # "?" anywhere, or inverted word order with no "." / "!" at the end: "Is it worth it", "Why does it lag"
    # ("Would definitely recommend", "Is just amazing" are not questions: no subject after the auxiliary)
    "question": re.compile(rf"\?|^(?:(?:{_AUX})\s+(?:{_SUBJECT})|(?:who|what|when|where|why|how|which)\s+(?:{_AUX}))\b(?!.*[.!]$)", re.I),
    "conditional": re.compile(r"\b(?:if|would|wish|hope|should)\b", re.I),
}


def artefacts(text: str) -> list[str]:
    """Names of the ARTEFACTS present in `text`."""
    return [name for name, found in ARTEFACTS.items() if found(text)]


def nuances(text: str) -> list[str]:
    """Names of the NUANCES present in `text` (normalised text expected)."""
    return [name for name, rx in NUANCES.items() if rx.search(text)]
