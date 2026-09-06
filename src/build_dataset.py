"""Expand hand-authored seeds into a labelled corpus.

Scope note: this is a *training-data* generator and nothing else. It has no
send path, no recipient list, no delivery integration, and it will not be given
one. See the ethics section of the README.

Two rules keep the expansion honest:

1. Every generated row keeps its `seed_id`. The splitter in `evaluate.py` is
   group-aware, so variants of one seed can never straddle train and test. Skip
   this and template leakage hands you a fake 99%.

2. Scam and ham get the *same* perturbations at the same rates. If ham were
   left clean while scam got noised, the model would learn the noise and the
   held-out score would measure nothing at all.

Perturbations here model natural variation only: people typing Kurdish on an
Arabic keyboard, mixed digit systems, sloppy spacing. The deliberate evasions
(zero-width injection, tatweel stretching, Latin transliteration) are held back
for the adversarial probe in `evaluate.py` -- training on them would make that
probe measure nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from normalize import console_utf8

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "seeds"
OUT_CSV = ROOT / "data" / "corpus.csv"

# All placeholders below are invented. No observed attacker infrastructure is
# reproduced here, and none of these hosts was resolved or visited.
FAKE_PHONES = [
    "0750 000 0000", "0751 000 0000", "0770 000 0000", "0771 000 0000",
    "0780 000 0000", "0781 000 0000", "0790 000 0000", "0751000 0000",
    "+964 750 000 0000", "07500000000", "0770-000-0000",
]

SCAM_HOSTS = [
    "zaincash-iq.top", "zaincash-win.top", "asiacell-prize.xyz",
    "korek-account.vip", "fib-secure.icu", "secure-iq-bank.click",
    "zain-login.ml", "iq-verify.link", "wallet-iq.online",
    "delivery-iq.click", "parcel-fee.top", "charity-iq.online",
    "support-iq.link", "iq-telecom.ga", "job-iq.site", "work-abroad.xyz",
    "bit.ly/2xQvR", "cutt.ly/aB3dE", "tinyurl.com/y7kp2",
    "verify-account.tk", "my-wallet.cf", "http://185.203.11.42/login",
]

# Legitimate hosts, keyed by the brand token that appears in the message. A
# legitimate Zain Cash SMS links to zaincash.iq, never to korek.com, so these
# may only be swapped within a brand. Picking from one flat pool produced rows
# like "زين كاش ... حمل التطبيق من www.asiacell.com", which is not a message
# any operator has ever sent and quietly taught the model that a brand paired
# with a mismatched domain is normal.
LEGIT_HOSTS_BY_BRAND = {
    "asiacell": ["asiacell.com", "www.asiacell.com"],
    "korek": ["korek.com", "www.korek.com"],
    "zain": ["zaincash.iq", "www.zaincash.iq", "zain.com"],
    "fib": ["fib.iq", "www.fib.iq"],
}

# Brand tokens as they appear in the message body, normalised forms included.
BRAND_MARKERS = {
    "asiacell": ["ئاسیاسێل", "اسياسيل", "آسياسيل", "asiacell", "ئاسیا سێل"],
    "korek": ["کۆرەک", "كورك", "korek", "کورەک"],
    "zain": ["زین", "زين", "zain", "zaincash"],
    "fib": ["fib", "بانکی یەکەم", "المصرف الاول"],
}

AMOUNTS = [
    "250000", "500000", "750000", "1000000", "1500000", "2000000",
    "3000000", "5000000", "10000000", "15000000", "25000", "15000",
    "50000", "100000", "20000", "12500", "45000", "27500", "340000",
]

CKB_GREETINGS = ["بەڕێز، ", "سڵاو، ", "بەڕێز بەکارهێنەر، ", "", "", ""]
AR_GREETINGS = ["عزيزي المشترك، ", "عزيزي الزبون، ", "هلا، ", "", "", ""]

# Kurdish written on an Arabic keyboard: extremely common in real messages and
# the exact orthographic drift `normalize.py` exists to collapse.
#
# ه -> ة is deliberately absent. Teh marbuta is only ever word-final in Arabic
# and never a Kurdish variant at all; applying it blindly turned هلا into ةلا
# and هێڵەکەت into ةێڵەکەت, which no sender would ever type. Word-final ه -> ة
# in Arabic is handled separately in `_de_normalise`.
DE_NORMALISE = {"ی": "ي", "ک": "ك"}

ARABIC_INDIC = {str(d): chr(0x0660 + d) for d in range(10)}
EXT_ARABIC_INDIC = {str(d): chr(0x06F0 + d) for d in range(10)}

PHONE_ANY = re.compile(r"(?:\+964\s?|00964\s?)?0?7[5789][\d\s\-]{7,12}\d")
URL_ANY = re.compile(r"(?:https?://)?(?:[\w-]+\.)+[a-z]{2,6}(?:/[\w\-/]*)?", re.I)
DIGITS_RUN = re.compile(r"\b\d{4,}\b")


def _brand_in(text: str) -> str | None:
    for brand, markers in BRAND_MARKERS.items():
        if any(mk in text for mk in markers):
            return brand
    return None


def _swap_host(text: str, rng: random.Random, is_scam: bool) -> str:
    if is_scam:
        pool = SCAM_HOSTS
    else:
        # Only ever swap a legitimate link for another of the same brand's.
        brand = _brand_in(text)
        pool = LEGIT_HOSTS_BY_BRAND.get(brand) if brand else None
        if not pool:
            return text

    def repl(m):
        url = m.group(0)
        # Leave USSD-ish and bare numeric tokens alone.
        if not re.search(r"[a-z]", url, re.I):
            return url
        host = rng.choice(pool)
        if host.startswith("http"):
            return host
        scheme = "http://" if is_scam and rng.random() < 0.7 else ""
        tail = ""
        if is_scam and rng.random() < 0.6:
            tail = "/" + rng.choice(["verify", "login", "a", "x", "pay", "reg", "c"])
        return f"{scheme}{host}{tail}"

    return URL_ANY.sub(repl, text, count=1)


def _swap_amount(text: str, rng: random.Random) -> str:
    """Swap a number for another of the same magnitude.

    Magnitude-matched on purpose. Rewriting a legitimate "30GB for 20,000 IQD"
    promo into "for 1,000,000 IQD" produces a message no operator would send,
    and it quietly couples the money_magnitude signal to the perturbation
    rather than to the label.
    """
    def repl(m):
        same_width = [a for a in AMOUNTS if len(a) == len(m.group(0))]
        return rng.choice(same_width) if same_width else m.group(0)

    return DIGITS_RUN.sub(repl, text, count=1)


def _swap_phone(text: str, rng: random.Random) -> str:
    return PHONE_ANY.sub(lambda m: rng.choice(FAKE_PHONES), text, count=1)


def _de_normalise(text: str, rng: random.Random, lang: str) -> str:
    """Rewrite some letters in their Arabic-keyboard forms."""
    out = []
    for i, ch in enumerate(text):
        if ch in DE_NORMALISE and rng.random() < 0.7:
            out.append(DE_NORMALISE[ch])
            continue
        # Word-final ه -> ة, Arabic only: "ساعه" for "ساعة" is ordinary
        # sloppy typing. Mid-word it is not a thing, and in Kurdish it is
        # never a thing.
        if (
            lang == "ar"
            and ch == "ه"
            and i > 0
            and (i + 1 == len(text) or not text[i + 1].isalpha())
            and rng.random() < 0.3
        ):
            out.append("ة")
            continue
        out.append(ch)
    return "".join(out)


def _swap_digits(text: str, rng: random.Random) -> str:
    """Swap digit systems outside URLs.

    URLs are skipped: a sender who writes ١٥٠٠ in the message body still types
    the link with ASCII digits, and rewriting them produced dead hosts like
    tinyurl.com/y۷kp۲ that exist nowhere outside this generator.
    """
    table = rng.choice([ARABIC_INDIC, EXT_ARABIC_INDIC])
    parts = []
    last = 0
    for m in URL_ANY.finditer(text):
        if not re.search(r"[a-z]", m.group(0), re.I):
            continue
        parts.append("".join(table.get(c, c) for c in text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append("".join(table.get(c, c) for c in text[last:]))
    return "".join(parts)


def _spacing_noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        text = re.sub(r"، ", "،", text, count=1)
    if rng.random() < 0.5:
        text = re.sub(r" ", "  ", text, count=1)
    if rng.random() < 0.3:
        text = text.replace(".", " .", 1)
    return text


def _punct_noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.4:
        text = text.rstrip(".") + rng.choice(["!", "!!", ".", "", "..."])
    return text


def _greeting(text: str, rng: random.Random, lang: str) -> str:
    pool = CKB_GREETINGS if lang == "ckb" else AR_GREETINGS
    g = rng.choice(pool)
    if not g:
        return text
    # Don't stack a second greeting on a message that already opens with one.
    if text.startswith(("بەڕێز", "سڵاو", "عزيزي", "هلا")):
        return text
    return g + text


def expand_seed(seed: dict, n: int, rng: random.Random) -> list[dict]:
    """Return `n` rows for one seed: the original plus n-1 perturbed variants."""
    is_scam = seed["label"] == "scam"
    lang = seed["lang"]
    rows = [dict(seed, text=seed["text"], seed_id=seed["id"], variant=0)]
    seen = {seed["text"]}

    attempts = 0
    while len(rows) < n and attempts < n * 12:
        attempts += 1
        t = seed["text"]

        # Slot fills. Applied to both classes at identical rates.
        if rng.random() < 0.7:
            t = _swap_phone(t, rng)
        if rng.random() < 0.7:
            t = _swap_host(t, rng, is_scam)
        if rng.random() < 0.6:
            t = _swap_amount(t, rng)
        if rng.random() < 0.5:
            t = _greeting(t, rng, lang)

        # Orthographic and typographic drift.
        if rng.random() < 0.45:
            t = _de_normalise(t, rng, lang)
        if rng.random() < 0.35:
            t = _swap_digits(t, rng)
        if rng.random() < 0.5:
            t = _spacing_noise(t, rng)
        if rng.random() < 0.4:
            t = _punct_noise(t, rng)

        if t in seen:
            continue
        seen.add(t)
        rows.append(dict(seed, text=t, seed_id=seed["id"], variant=len(rows)))

    return rows


def concept_id(seed_id: str) -> str:
    """Language-independent id shared by a seed and its translation.

    The Kurdish and Arabic seed files are parallel: ckb-scam-017 and
    ar-scam-017 are the same message in two languages, down to the amounts.
    Grouping the cross-lingual split on `seed_id` alone would therefore train
    on the Arabic version of the exact message being tested in Kurdish and
    report it as transfer. `evaluate.py` groups on this instead.
    """
    return re.sub(r"^(?:ckb|ar)-", "", seed_id)


def load_seeds() -> list[dict]:
    seeds = []
    for path in sorted(SEED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                s = json.loads(line)
                s["concept_id"] = concept_id(s["id"])
                seeds.append(s)
    return seeds


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Build the labelled corpus from seeds.")
    ap.add_argument("--per-seed", type=int, default=16,
                    help="rows generated per seed, original included")
    ap.add_argument("--seed", type=int, default=20260101, help="RNG seed")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = load_seeds()
    if not seeds:
        raise SystemExit(f"no seeds found in {SEED_DIR}")

    rows = []
    for s in seeds:
        rows.extend(expand_seed(s, args.per_seed, rng))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["text", "label", "lang", "category", "source", "seed_id",
              "concept_id", "variant"]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    by = Counter((r["lang"], r["label"]) for r in rows)
    print(f"seeds        : {len(seeds)}")
    print(f"rows         : {len(rows)}")
    for (lang, label), n in sorted(by.items()):
        print(f"  {lang:4s} {label:5s} {n:5d}")
    print(f"unique texts : {len({r['text'] for r in rows})}")
    print(f"sources      : {dict(Counter(r['source'] for r in rows))}")
    print(f"written to   : {args.out}")


if __name__ == "__main__":
    main()
