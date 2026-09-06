"""Arabic-script normalisation and entity folding.

Sorani Kurdish and Arabic share a script but not orthographic norms. The same
word routinely appears under several code points that render near-identically
(Kurdish U+06CC vs Arabic U+064A), so an unnormalised model treats them as
unrelated tokens. Scam authors exploit that on purpose: tatweel padding,
stretched letters and mixed digit systems all split a blocked keyword without
changing how the message looks to a human.

Everything here is character-wise, which lets `normalize_traced` return an
offset map from each normalised character back to the character in the raw
message that produced it. The demo needs that map to highlight the span that
actually triggered a decision -- an explanation that points at the wrong
characters is worse than no explanation.

KLPT (github.com/sinaahmadi/klpt) covers the Kurdish half of this and would be
the better citation, but it does not install on Python 3.14: every release
pins cyhunspell/chunspell, which have no wheels for this interpreter and need
a hunspell toolchain to build. This module is the hand-rolled replacement.
"""

from __future__ import annotations

import re
import sys
import unicodedata


def console_utf8() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, which cannot encode Arabic script:
    printing a Kurdish message raises UnicodeEncodeError and kills the script.
    Every entry point in this project calls this first.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# --- character classes -------------------------------------------------------

# Zero-width and bidi controls. Splitting a keyword with an invisible character
# is the cheapest keyword-filter evasion there is, so these are removed before
# anything else looks at the text.
INVISIBLE = (
    "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"
    "\u202a\u202b\u202c\u202d\u202e"
    "\u2066\u2067\u2068\u2069"
)

TATWEEL = "\u0640"

# Arabic diacritics (tashkil) plus superscript alef. Optional in both languages
# and near-absent from real SMS, so they carry no signal and only fragment
# n-grams when a scammer sprinkles them in.
DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")

# Fold toward Kurdish forms. U+06D5 (ە) and U+0626 (ئ) are deliberately absent:
# they are distinct Kurdish letters, not variants of anything.
LETTER_FOLD = {
    "\u064a": "\u06cc",  # ي  Arabic yeh      -> ی Farsi yeh
    "\u0649": "\u06cc",  # ى  alef maksura    -> ی
    "\u06d2": "\u06cc",  # ے  yeh barree      -> ی
    "\u06cd": "\u06cc",  # ۍ  yeh with tail   -> ی
    "\u0643": "\u06a9",  # ك  Arabic kaf      -> ک Keheh
    "\u0629": "\u0647",  # ة  teh marbuta     -> ه
    "\u0623": "\u0627",  # أ  alef+hamza above-> ا
    "\u0625": "\u0627",  # إ  alef+hamza below-> ا
    "\u0622": "\u0627",  # آ  alef madda      -> ا
    "\u0671": "\u0627",  # ٱ  alef wasla      -> ا
    "\u0624": "\u0648",  # ؤ  waw+hamza       -> و
}

# Arabic-Indic (U+0660) and Extended Arabic-Indic (U+06F0) digits -> ASCII.
# Iraqi phones are quoted in both systems, sometimes within one message.
for _i in range(10):
    LETTER_FOLD[chr(0x0660 + _i)] = str(_i)
    LETTER_FOLD[chr(0x06F0 + _i)] = str(_i)
del _i


def _strip_invisible(text: str) -> str:
    return text.translate({ord(c): None for c in INVISIBLE})


def normalize_traced(text: str) -> tuple[str, list[int]]:
    """Normalise `text`, returning the result and a per-character offset map.

    ``offsets[i]`` is the index in `text` of the character that produced
    ``result[i]``. Characters dropped entirely contribute nothing; a character
    that NFKC expands maps every output character back to the same source
    index, which is what a highlighter wants.
    """
    out: list[str] = []
    offsets: list[int] = []

    for idx, ch in enumerate(text):
        if ch in INVISIBLE or ch == TATWEEL:
            continue
        if DIACRITICS.match(ch):
            continue

        # NFKC per character: keeps the offset map exact. Composition across a
        # character boundary is not something Arabic-script SMS relies on.
        folded = unicodedata.normalize("NFKC", ch)
        folded = "".join(LETTER_FOLD.get(c, c) for c in folded)
        folded = folded.lower()

        for c in folded:
            # Collapse letter runs of 3+ to 2: "فوووووري" and "فووري" are the
            # same word wearing different amounts of urgency. Digits are
            # exempt -- collapsing them turns 10,000,000 into 100 and shreds
            # phone numbers before entity folding ever sees them.
            if (
                not c.isdigit()
                and len(out) >= 2
                and out[-1] == c
                and out[-2] == c
            ):
                continue
            out.append(c)
            offsets.append(idx)

    # Whitespace collapse, still tracking offsets.
    result: list[str] = []
    result_offsets: list[int] = []
    prev_space = False
    for c, off in zip(out, offsets):
        if c.isspace():
            if prev_space or not result:
                continue
            result.append(" ")
            result_offsets.append(off)
            prev_space = True
        else:
            result.append(c)
            result_offsets.append(off)
            prev_space = False
    while result and result[-1] == " ":
        result.pop()
        result_offsets.pop()

    return "".join(result), result_offsets


def normalize(text: str) -> str:
    """Normalised form of `text`. Use `normalize_traced` when spans matter."""
    return normalize_traced(text)[0]


# --- entity folding ----------------------------------------------------------

URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"(?:com|net|org|iq|krd|info|biz|xyz|top|icu|tk|ml|ga|cf|click|link|vip|shop|online|site|live|app|co|ru|cn|test)"
    r"(?:/[^\s<>\"']*)?",
    re.IGNORECASE,
)

# Iraqi mobile numbers: 07[5789] + 8 more digits, optionally +964/00964 with the
# trunk 0 dropped. Separators are whatever the sender felt like using.
PHONE_RE = re.compile(
    r"(?:(?:\+|00)\s?964[\s\-.]?|0)\s?7[5789][\s\-.]?\d[\s\-.]?\d{3}[\s\-.]?\d{4}"
)

CURRENCY_WORDS = (
    r"دینار|دينار|د\.ع|iqd|دۆلار|دولار|\$|usd|هەزار|الف|ألف|ملیۆن|مليون|ملیون|میلیۆن|k|m"
)
MONEY_RE = re.compile(
    r"\d[\d,،.\s]{0,15}\d\s*(?:" + CURRENCY_WORDS + r")\b"
    r"|\b(?:" + CURRENCY_WORDS + r")\s*\d[\d,،.\s]{0,15}\d",
    re.IGNORECASE,
)

LONG_NUM_RE = re.compile(r"\b\d{4,}\b")

URL_TOKEN = " __url__ "
PHONE_TOKEN = " __phone__ "
MONEY_TOKEN = " __money__ "
NUM_TOKEN = " __num__ "


def fold_entities(text: str) -> str:
    """Replace URLs, phones, money amounts and long digit runs with placeholders.

    For the n-gram model only. The *presence* of a link is the signal;
    memorising one campaign's short URL is not learning, and on a corpus this
    size it would quietly inflate the held-out score. `signals.py` reads the raw
    text so it can still inspect the actual domain.

    Order matters: URLs contain digits and dots, phones contain digit runs, so
    the most specific pattern has to claim its span first.
    """
    text = URL_RE.sub(URL_TOKEN, text)
    text = PHONE_RE.sub(PHONE_TOKEN, text)
    text = MONEY_RE.sub(MONEY_TOKEN, text)
    text = LONG_NUM_RE.sub(NUM_TOKEN, text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str, *, normalise: bool = True, fold: bool = True) -> str:
    """Full pipeline used by the model. Flags exist for the ablation study."""
    if normalise:
        text = normalize(text)
    if fold:
        text = fold_entities(text)
    return text


if __name__ == "__main__":
    console_utf8()
    samples = [
        "بەڕێـــز، ژمارەت بردویەتیەوە ١٠٠٠٠٠٠٠ دینار! پەیوەندی بکە ٠٧٥٠ ٠٠٠ ٠٠٠٠",
        "عزيــزي المشترك حسابك في زين كاش موقـ__وف، فعّله عبر http://zaincash-iq.top/a",
        "کۆدەکەت 458213 ە. لەگەڵ کەسێکدا بەشی مەکە.",
    ]
    for s in samples:
        n = normalize(s)
        print("raw  :", s)
        print("norm :", n)
        print("fold :", fold_entities(n))
        print()
