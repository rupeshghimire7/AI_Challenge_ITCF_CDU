"""Aspect terms: which laptop entities a sentence names, and whether it is about something other than the laptop.

The brief (Slot 1): "When there is no explicit mention of the entity or attribute, E or A takes the value NULL". The model
always answers with one of the 22 entities, so whether the entity is *named* has to be read off the text; and the model
was trained on laptop reviews only, so it reads a sentence about a phone or a pizza as a laptop opinion. These patterns
let the post-processing catch both.

ENTITY_TERMS          per SemEval-2016 laptop entity, the words that name it (entity definitions of the annotation
                      guidelines, gaps filled from the train sentences); brand names name LAPTOP and COMPANY
OTHER_TERMS           objects outside the laptop domain: other devices, food, travel, media, clothes, vehicles
FILLER                words that carry no opinion and name no aspect (greetings, thanks, stop words)

named_entities(text)  entities the text names, in ENTITY_TERMS order
entity_matches(text)  the same, each with the word that names it: {"MOUSE": "trackpad", "MEMORY": "RAM"}
off_topic(text)       the non-laptop object the sentence is about, or "": the object is the subject ("My printer keeps
                      jamming") or what is loved / hated / bought ("I love my new phone"), the laptop is not named, and the
                      sentence compares nothing. Merely mentioning an object is not enough: laptop reviews talk about
                      watching movies, printers and flights, and those opinions are about the laptop.
is_filler(text)       the text has words and every one is FILLER ("Hello.", "Thanks everyone!")
"""

from __future__ import annotations

import re

_BRANDS = (r"(?:apple|hp|dell|lenovo|toshiba|asus|acer|samsung|sony|msi|alienware|razer|gateway|compaq|fujitsu|huawei|lg"
           r"|microsoft|google|ibm)s?")

ENTITY_TERMS: dict[str, str] = {
    "LAPTOP": rf"laptops?|notebooks?|computers?|pcs?|macbooks?|macs?|chromebooks?|ultrabooks?|netbooks?|machines?"
              rf"|devices?|units?|products?|models?|systems?|equipment|thinkpads?|ideapad|inspiron|xps|pavilion|envy"
              rf"|spectre|aspire|vaio|zenbook|vivobook|latitude|elitebook|probook|{_BRANDS}",
    "DISPLAY": r"screens?|displays?|monitors?|resolution|retina|lcd|touch ?screens?|brightness|pixels?|glare|bezels?"
               r"|picture|high definition|hd|visuals?|fonts?",
    "KEYBOARD": r"key ?boards?|keys?|key ?pads?|number ?pad|10-key|typed|typing|backlit|backlight(?:ing)?"
                r"|keystrokes?",
    "MOUSE": r"mouse|mice|mouse ?pads?|touch ?pads?|track ?pads?|click ?pads?|track ?buttons?|trackpoint|cursor"
             r"|pointer|scrolling",
    "MOTHERBOARD": r"mother ?boards?|main ?boards?|logic ?boards?|system ?boards?",
    "CPU": r"cpus?|processors?|i[3579]|ryzen|celeron|pentium|atom|ghz|(?:dual|quad)[ -]core|intel|amd|chips?",
    "FANS_COOLING": r"fans?|cooling|cooler|heat|hot|overheat(?:s|ed|ing)?|temperatures?|vents?|warm|thermal",
    "PORTS": r"ports?|usb(?:-?c)?|hdmi|thunderbolt|ethernet|vga|sd ?cards?|card ?readers?|slots?|jacks?|displayport"
             r"|firewire|dongles?",
    "MEMORY": r"memory|ram|ddr\d?",
    "POWER_SUPPLY": r"power ?supply|chargers?|charging|adapters?|power ?cords?|cords?|power ?cables?|cables?|bricks?"
                    r"|plugs?|magsafe",
    "OPTICAL_DRIVES": r"dvds?|cds?|optical|disc ?drives?|blu-?ray|burners?|cd-?rom|superdrive",
    "BATTERY": r"batter(?:y|ies)|charge|juice|unplugged",
    "GRAPHICS": r"graphics?|gpus?|video ?cards?|nvidia|geforce|gtx|rtx|radeon|quadro|iris|frame ?rates?|fps",
    "HARD_DISC": r"hard ?d?rives?|hard ?(?:disks?|discs?)|hdds?|ssds?|storage|drives?|disks?|solid[ -]state|emmc"
                 r"|tb|capacity|gigs",
    "MULTIMEDIA_DEVICES": r"speakers?|sound|audio|web ?cams?|cameras?|microphones?|mics?|volume|bass|head ?phones?",
    "HARDWARE": r"hardware|hinges?|casing|chassis|body|lid|shell|components?|parts|construction",
    "SOFTWARE": r"software|programs?|apps?|applications?|bloatware|office|excel|powerpoint|antivirus|norton|mcafee"
                r"|drivers?|browsers?|safari|firefox|itunes|photoshop|iwork|iphoto|imovie|garageband|skype|games?"
                r"|subscriptions?|trial",
    "OS": r"os|operating ?systems?|windows?(?: ?\d+(?:\.\d)?)?|win ?\d+|w ?\d+|mac ?os(?: ?x)?|osx|macos|linux"
          r"|ubuntu|chrome ?os|vista|xp|yosemite|mavericks|sierra|catalina|big sur|el capitan|(?:mountain )?lion"
          r"|snow leopard|interface|start ?(?:up|menu|screen)",
    "WARRANTY": r"warrant(?:y|ies)|guarantee|apple ?care|protection ?plans?",
    "SHIPPING": r"shipping|shipped|ship|ships|deliver(?:y|ed)?|arrived|arrival|packag(?:e|es|ing)|fedex|courier",
    "SUPPORT": r"support|customer ?service|service|tech ?support|help ?desk|technicians?|customer ?care|geek ?squad"
               r"|repairs?|reps?|representatives?|call ?cent(?:er|re)|agents?|staff|associates?|employees?",
    "COMPANY": rf"company|companies|brands?|manufacturers?|makers?|corporation|{_BRANDS}",
}

OTHER_TERMS = (r"(?<!head )(?:phones?|smart ?phones?|iphones?|cell ?phones?|ipads?|kindles?|tvs?|televisions?|printers?"
               r"|scanners?|smart ?watch(?:es)?|xbox|playstation|ps[45]|nintendo|consoles?|cars?|trucks?|bikes?|food"
               r"|pizzas?|burgers?|meals?|dinner|lunch|breakfast|coffee|restaurants?|waiters?|waitress(?:es)?|hotels?"
               r"|flights?|movies?|films?|novels?|songs?|albums?|shoes|shirts?|dress(?:es)?|weather)")

FILLER = frozenset("""hello hi hey hiya greetings thanks thank thx cheers regards bye goodbye update edit ps note yes no
    ok okay well so um uh oh lol everyone everybody all guys folks you your i me my we our it this that the a an and or
    to for of in on at is was be been am are just again here there""".split())

_ENTITY_RX = {e: re.compile(rf"\b(?:{p})\b", re.I) for e, p in ENTITY_TERMS.items()}
_DET = r"(?:(?:the|my|our|your|his|her|their|this|that|these|those|a|an)\s+)?"
_MOD = r"(?:(?![\w'-]*ing\s)[\w'-]+\s+){0,2}"  # up to two modifiers ("brand new"), not a verb ("enjoy watching movies")
_TARGET = [  # the non-laptop object is the subject of the sentence, or the object of an evaluation / purchase verb
    re.compile(rf"^{_DET}{_MOD}\b(?P<x>{OTHER_TERMS})(?:'s)?\s+(?:[\w'-]+\s+){{0,3}}(?:is|was|are|were|has|had|keeps?|kept"
               rf"|looks?|looked|seems?|seemed|works?|worked|tastes?|tasted|feels?|felt|broke|died|sucks?|rocks?)\b", re.I),
    re.compile(rf"\b(?:love|like|hate|enjoy|adore|dislike|recommend|prefer|buy|bought|purchased?|ordered?|got)[sd]?\s+"
               rf"{_DET}{_MOD}\b(?P<x>{OTHER_TERMS})\b", re.I),
]
_COMPARE = re.compile(r"\b(?:than|compared?|comparison|versus|vs\.?|unlike|similar|better|worse|best|worst)\b", re.I)
_WORD = re.compile(r"[^\W\d_]+")


def entity_matches(text: str) -> dict[str, str]:
    return {e: m.group(0) for e, rx in _ENTITY_RX.items() if (m := rx.search(text))}


def named_entities(text: str) -> list[str]:
    return list(entity_matches(text))


def off_topic(text: str) -> str:
    if _COMPARE.search(text) or _ENTITY_RX["LAPTOP"].search(text):
        return ""
    m = next((m for rx in _TARGET if (m := rx.search(text))), None)
    return m.group("x").lower() if m else ""


def is_filler(text: str) -> bool:
    words = _WORD.findall(text.lower())
    return bool(words) and all(w in FILLER for w in words)
