"""Text preprocessing: one implementation for training, evaluation and serving, so the text a model trains on and the
text it serves are byte-identical (plan D2.4, 6.2.2).

normalise(text)         D2.4 normalisation: repairs every nuance_checker artefact; case, emoji and spelling are kept
split_sentences(text)   review text -> normalised sentences (abbreviation-safe splitter)
preprocess(review)      serving entry point: one record per sentence with its text, artefacts and nuances
match_key(text)         comparison key for duplicates and train/test leakage (case, spacing and punctuation ignored)
unescape_html(text)     HTML entities, decoded until stable
"""

from __future__ import annotations

import html
import re
import unicodedata

from src.nuance_checker import (BR, CONTROL, INVISIBLE, LETTER_RUN, LIST_MARKER, PUNCT_MAP, PUNCT_RUN, TAG, TYPOGRAPHIC,
                                artefacts, nuances)

_SPACE = re.compile(r"\s+")


def unescape_html(text: str) -> str:
    """HTML entities, repeated until stable: the SemEval test XML is double-escaped ("&amp;#34;" -> "&#34;" -> '"')."""
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def normalise(text: str) -> str:
    """HTML entities and tags, Unicode NFKC, typographic quotes and dashes, invisible / control characters, whitespace,
    leading list markers, and character runs longer than 3 (capped at 3: "soooooo" -> "sooo"). Idempotent."""
    t = unescape_html(text)
    t = TAG.sub(" ", BR.sub(" ", t))
    t = _typographic(unicodedata.normalize("NFKC", _typographic(t)))  # again after NFKC: it turns "″" into "′′"
    t = LIST_MARKER.sub("", _SPACE.sub(" ", CONTROL.sub(" ", INVISIBLE.sub("", t))).strip())
    return PUNCT_RUN.sub(r"\1\1\1", LETTER_RUN.sub(r"\1\1\1", t))


def _typographic(t: str) -> str:
    return t.translate(PUNCT_MAP) if TYPOGRAPHIC.search(t) else t  # translate() is slow; most text has nothing to map


def match_key(text: str) -> str:
    """Lowercase letters and digits only, single-spaced: "I LOVE it!" and "i love it !" share one key."""
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", unescape_html(text)).lower()))


# ----------------------------------------------------------------------------------------------- sentence splitting
# After these, a full stop never ends a sentence ("e.g. the fan", "Mr. Jobs", "approx. 5 hours", "no. 3").
_NEVER_SPLIT = {"e.g", "i.e", "eg", "ie", "vs", "v", "mr", "mrs", "ms", "dr", "st", "jr", "sr", "approx", "appx", "no", "nos",
                "cf", "ca", "fig", "vol", "pg", "p", "pp"}
# After these, a full stop ends a sentence only if the next word is capitalised ("5 hrs. Great screen" splits).
_MAYBE_SPLIT = {"etc", "hr", "hrs", "min", "mins", "sec", "secs", "yr", "yrs", "mo", "mos", "wk", "wks", "lb", "lbs", "oz",
                "ft", "in", "inc", "ltd", "co", "corp", "dept", "misc", "est", "avg", "max", "jan", "feb", "mar", "apr",
                "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "a.m", "p.m", "u.s", "u.k"}
_CANDIDATE = re.compile(r"""(?:[.!?]+|(?<!\S)[:;]-?[()DPp])["')\]]*(?=\s+(\S))""")  # sentence-final punctuation or emoticon
_BULLET = re.compile(r"^\s*(?:[-*•>]|\d+[.)])\s+")
_LAST_WORD = re.compile(r"[^\s(\[\"']*\Z")


def _ends_sentence(line: str, m: re.Match) -> bool:
    end, nxt = m.group(0), m.group(1)
    if end[0] in ":;":  # emoticon: ends a sentence only before a capital ("works great :) The keys ...")
        return nxt[0].isupper()
    if "!" in end or "?" in end:
        return True
    if end.startswith(".."):  # ellipsis: "Being a PC user my whole life.... This computer" splits; "slow... and hot" not
        return nxt[0].isupper()
    word = _LAST_WORD.search(line, max(0, m.start() - 32), m.start())[0].lower()  # bounded window: no rescan of the line
    if word in _NEVER_SPLIT:
        return False
    if word in _MAYBE_SPLIT or len(word) == 1 or "." in word:  # "etc.", single initials, dotted tokens ("u.s")
        return nxt[0].isupper()
    return True


def _segments(text: str) -> list[tuple[str, str]]:
    """Review text -> (raw, normalised) sentence pairs. Line breaks and <br> tags are hard boundaries (Amazon reviews use
    them for lists); inside a line, `.` `!` `?` and emoticons end a sentence unless an abbreviation or decimal says
    otherwise. Fragments with no letter or digit (a stray "!!") join the previous sentence."""
    lines = BR.sub("\n", unescape_html(text)).splitlines()
    pieces: list[str] = []
    for line in lines:
        line = _BULLET.sub("", line) if _BULLET.match(line) and len(lines) > 1 else line
        start = 0
        for m in _CANDIDATE.finditer(line):
            if _ends_sentence(line, m):
                pieces.append(line[start : m.end()])
                start = m.end()
        pieces.append(line[start:])
    out: list[tuple[str, str]] = []
    for p in pieces:
        clean = normalise(p)
        if not clean:
            continue
        if out and not re.search(r"[^\W_]", clean):
            raw = f"{out[-1][0]} {p.strip()}"
            out[-1] = (raw, normalise(raw))
        else:
            out.append((p.strip(), clean))
    return out


def split_sentences(text: str) -> list[str]:
    return [clean for _, clean in _segments(text)]


def preprocess(review: str) -> list[dict]:
    """Serving entry point: review text -> one record per sentence with `text` (normalised), `artefacts` (what
    normalisation repaired in it) and `nuances` (the D5.1 slices it contains)."""
    return [{"text": clean, "artefacts": artefacts(raw), "nuances": nuances(clean)} for raw, clean in _segments(review)]
