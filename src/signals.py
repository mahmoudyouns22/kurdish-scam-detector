"""Hand-built, interpretable signals.

These run on the *raw* message, not the entity-folded form the n-gram model
sees. Folding throws away the actual domain on purpose; this module needs it,
because "there is a link" and "there is a link to zaincash-iq.top" are very
different claims.

Every signal is a plain number and most carry the character spans that fired,
so `predict.py` can point at the part of the message that caused the verdict.

Lexicon entries are written in natural orthography and normalised at import,
so matching happens in the same space as `normalize.normalize()` output. Adding
a word here does not require knowing which yeh or kaf the sender typed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from normalize import MONEY_RE, PHONE_RE, URL_RE, normalize, normalize_traced

# --- lexicons ----------------------------------------------------------------
# ckb = Sorani Kurdish, ar = Iraqi Arabic, en = English/transliterated.
# NOTE: drafted by a non-native speaker; the Kurdish rows need review.

URGENCY = [
    # ckb
    "خێرا", "بەپەلە", "دەستبەجێ", "زوو", "ماوەکە کۆتایی", "تەنها ئەمڕۆ",
    "دواجار", "ئاگاداری", "کۆتا هەل", "لە ماوەی", "ئەگەرنا",
    "هەڵدەوەشێتەوە", "دادەخرێت", "پەلە بکە",
    # ar
    "عاجل", "فوري", "فورا", "بسرعة", "الان", "خلال 24 ساعة", "اخر فرصة",
    "قبل انتهاء", "سيتم ايقاف", "سيتم حذف", "تنبيه", "انذار", "المهلة",
    "خلال ساعة", "وإلا",
    # en
    "urgent", "immediately", "expires", "act now", "last chance",
    "final notice", "within 24 hours", "will be suspended",
]

PRIZE = [
    # ckb
    "براوە", "بردویەتەوە", "خەڵات", "دیاری", "پیرۆزە", "بەختەوەر",
    "ژمارەکەت هەڵبژێردرا", "کێبڕکێ", "بردتەوە",
    # ar
    "فزت", "ربحت", "فائز", "رابح", "جائزة", "مبروك", "مبارك", "سحب",
    "تم اختيار رقمك", "مسابقة", "هدية",
    # en
    "winner", "you won", "congratulations", "prize", "lucky",
]

SENSITIVE_REQUEST = [
    # ckb
    "وشەی نهێنی", "ژمارەی ئەکاونت", "زانیاری بانکی", "ژمارەی کارت",
    "زانیارییەکانت بنێرە", "ناسنامەکەت",
    # ar
    "كلمة السر", "كلمة المرور", "الرقم السري", "رقم البطاقة", "بيانات حسابك",
    "معلومات البطاقة", "رقم الهوية", "ارسل بياناتك",
    # en
    "password", "pin code", "card number", "account details",
    "verify your identity",
]

# A genuine OTP *gives* you a code and tells you not to share it. A scam *asks
# you to send one back*. This pair is one of the sharpest signals available and
# is worth encoding explicitly rather than hoping n-grams find it.
ASKS_FOR_CODE = [
    # ckb
    "کۆدەکە بنێرە", "کۆدەکەم بۆ بنێرە", "کۆدەکە پێمان بڵێ", "کۆدەکە بنووسە",
    "کۆدەکەت بنێرە", "ژمارەی پشتڕاستکردنەوە بنێرە", "کۆدەکە بۆم بنێرە",
    # ar
    "ارسل الرمز", "ارسل الكود", "زودنا بالرمز", "اخبرنا بالكود", "اكتب الرمز",
    "ارسل رمز التحقق", "شارك الرمز", "ارسل لنا الكود",
    # en
    "send the code", "send us the code", "share the code", "forward the code",
    "reply with the code", "tell us the otp",
]

PROVIDES_CODE = [
    # ckb
    "کۆدەکەت", "کۆدی پشتڕاستکردنەوەت", "بەشی مەکە", "لەگەڵ کەس", "مەیدە بە کەس",
    # ar
    "رمز التحقق الخاص بك", "كودك", "رمزك هو", "لا تشاركه", "لا تشارك الرمز",
    "لا تعطي الرمز", "لا تشاركه مع",
    # en
    "your verification code", "your otp is", "do not share",
    "never share this code",
]

ACTION_LINK = [
    # ckb
    "کلیک بکە", "لینکەکە بکەرەوە", "بچۆ بۆ", "داگرە", "تۆمار بکە",
    # ar
    "اضغط", "انقر", "افتح الرابط", "سجل الان", "حمل التطبيق", "ادخل الى",
    # en
    "click here", "tap the link", "open the link", "register now", "download",
]

PAY_FEE = [
    # ckb
    "پارە بنێرە", "کرێی", "باج", "پارەدان", "بڕی کەم", "کرێی گەیاندن",
    # ar
    "ادفع", "رسوم", "الرسوم الجمركية", "اجور التوصيل", "مبلغ رمزي",
    "حول المبلغ", "رسوم بسيطة",
    # en
    "pay a fee", "customs fee", "delivery fee", "small fee",
    "transfer the amount",
]

BRANDS = {
    "zain": ["زین", "زين", "zain", "زین کاش", "زين كاش", "zaincash", "zain cash"],
    "asiacell": ["ئاسیاسێل", "اسياسيل", "آسياسيل", "asiacell", "ئاسیا سێل"],
    "korek": ["کۆرەک", "كورك", "korek", "کورەک"],
    "fastpay": ["فاست پەی", "فاست باي", "fastpay", "fast pay"],
    "nasswallet": ["ناس والێت", "ناس واليت", "nasswallet", "nass wallet"],
    "fib": ["fib", "بانکی یەکەم", "المصرف الاول", "first iraqi bank"],
    "bank": ["بانک", "بانك", "مصرف", "bank"],
}

# Domains the brands actually use. Anything else carrying a brand token is a
# lookalike, which is the most reliable phishing tell in this corpus.
LEGIT_DOMAINS = {
    "zaincash.iq", "iq.zain.com", "zain.com", "asiacell.com", "korek.com",
    "fib.iq", "nasswallet.com", "fastpaycash.com", "fastpay.iq",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly",
    "rb.gy", "shorturl.at", "rebrand.ly", "bit.do", "s.id",
}

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".icu", ".tk", ".ml", ".ga", ".cf", ".click", ".link",
    ".vip", ".shop", ".online", ".live", ".site",
}

BRAND_TOKENS = ("zain", "asiacell", "korek", "fastpay", "fib", "nass")

IPV4_URL_RE = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}")
ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿ]")
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)


def _norm_lexicon(words):
    seen = {}
    for w in words:
        n = normalize(w)
        if n:
            seen[n] = None
    return list(seen)


URGENCY_N = _norm_lexicon(URGENCY)
PRIZE_N = _norm_lexicon(PRIZE)
SENSITIVE_N = _norm_lexicon(SENSITIVE_REQUEST)
ASKS_CODE_N = _norm_lexicon(ASKS_FOR_CODE)
PROVIDES_CODE_N = _norm_lexicon(PROVIDES_CODE)
ACTION_LINK_N = _norm_lexicon(ACTION_LINK)
PAY_FEE_N = _norm_lexicon(PAY_FEE)
BRANDS_N = {k: _norm_lexicon(v) for k, v in BRANDS.items()}


@dataclass
class SignalResult:
    features: dict
    spans: list = field(default_factory=list)


def _lexicon_hits(norm_text, offsets, lexicon, label):
    """Count lexicon matches in normalised text, mapped back to raw spans."""
    hits = 0
    spans = []
    for phrase in lexicon:
        start = 0
        while True:
            i = norm_text.find(phrase, start)
            if i < 0:
                break
            hits += 1
            if offsets:
                lo = offsets[i]
                hi = offsets[min(i + len(phrase) - 1, len(offsets) - 1)] + 1
                spans.append((lo, hi, label))
            start = i + len(phrase)
    return hits, spans


def _domains(text):
    out = []
    for m in URL_RE.finditer(text):
        host = re.sub(r"^https?://", "", m.group(0), flags=re.I).split("/")[0]
        host = host.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        out.append(host)
    return out


def analyse(text: str) -> SignalResult:
    """Compute every signal for one message."""
    norm, offsets = normalize_traced(text)
    f = {}
    spans = []

    # --- links ---
    urls = list(URL_RE.finditer(text))
    domains = _domains(text)
    f["url_present"] = 1.0 if urls else 0.0
    f["url_count"] = float(len(urls))
    for m in urls:
        spans.append((m.start(), m.end(), "link"))

    f["url_shortener"] = float(any(d in SHORTENERS for d in domains))
    f["url_suspicious_tld"] = float(
        any(any(d.endswith(t) for t in SUSPICIOUS_TLDS) for d in domains)
    )
    f["url_raw_ip"] = float(bool(IPV4_URL_RE.search(text)))

    # A brand token inside a domain that is not the brand's real domain. This
    # catches zaincash-iq.top, asiacell-win.xyz and the rest of the family.
    lookalike = 0.0
    for d in domains:
        if d in LEGIT_DOMAINS or any(d.endswith("." + g) for g in LEGIT_DOMAINS):
            continue
        flat = re.sub(r"[^a-z]", "", d)
        if any(b in flat for b in BRAND_TOKENS):
            lookalike = 1.0
    f["url_brand_lookalike"] = lookalike

    # --- phones and money ---
    phones = list(PHONE_RE.finditer(norm))
    f["phone_present"] = float(bool(phones))
    f["phone_count"] = float(len(phones))
    for m in phones:
        if offsets:
            lo = offsets[m.start()]
            hi = offsets[min(m.end() - 1, len(offsets) - 1)] + 1
            spans.append((lo, hi, "phone"))

    # Money requires a currency term, not just a big number. A bare 6-digit
    # run is far more likely to be an OTP code, and counting that as "money"
    # would blur the one distinction this classifier most needs to keep sharp.
    money_spans = list(MONEY_RE.finditer(norm))
    amounts = []
    for m in money_spans:
        digits = re.sub(r"[^\d]", "", m.group(0))
        if digits:
            amounts.append(int(digits))
        if offsets:
            lo = offsets[m.start()]
            hi = offsets[min(m.end() - 1, len(offsets) - 1)] + 1
            spans.append((lo, hi, "money"))
    big = max(amounts) if amounts else 0
    f["money_present"] = float(bool(money_spans))
    # Magnitude bucket, not the raw value: the model should learn "implausibly
    # large" rather than one campaign's favourite number.
    f["money_magnitude"] = 0.0 if big < 1000 else float(min(len(str(big)) - 3, 6))

    # --- lexicons ---
    for name, lex in (
        ("urgency", URGENCY_N),
        ("prize", PRIZE_N),
        ("sensitive", SENSITIVE_N),
        ("asks_for_code", ASKS_CODE_N),
        ("provides_code", PROVIDES_CODE_N),
        ("action_link", ACTION_LINK_N),
        ("pay_fee", PAY_FEE_N),
    ):
        hits, sp = _lexicon_hits(norm, offsets, lex, name)
        f["lex_" + name] = float(hits)
        spans.extend(sp)

    # The distinction, not just the two counts.
    f["otp_asks_not_provides"] = float(
        f["lex_asks_for_code"] > 0 and f["lex_provides_code"] == 0
    )

    # --- brands ---
    brand_hits = 0
    for name, lex in BRANDS_N.items():
        hits, sp = _lexicon_hits(norm, offsets, lex, "brand")
        if hits:
            brand_hits += 1
            spans.extend(sp)
    f["brand_mention"] = float(brand_hits)
    # Interaction terms: a brand name is normal and a link is normal, but a
    # brand name *plus* a link in an unsolicited SMS is the phishing shape.
    f["brand_and_link"] = float(brand_hits > 0 and f["url_present"] > 0)
    f["brand_and_lookalike"] = float(brand_hits > 0 and lookalike > 0)
    f["brand_and_urgency"] = float(brand_hits > 0 and f["lex_urgency"] > 0)

    # --- surface shape ---
    n = max(len(text), 1)
    f["length"] = float(len(text))
    f["digit_ratio"] = sum(c.isdigit() for c in norm) / max(len(norm), 1)
    f["exclaim_count"] = float(text.count("!"))
    f["punct_density"] = sum(c in "!?.,:*#" for c in text) / n
    f["upper_ratio"] = sum(c.isupper() for c in text) / n

    ar = len(ARABIC_SCRIPT_RE.findall(text))
    la = len(LATIN_RE.findall(text))
    total = ar + la
    # Script mixing: Latin transliteration inside an Arabic-script message is
    # normal in Kurdish chat, but heavy mixing also shows up in evasion.
    f["script_mix_ratio"] = (min(ar, la) / total) if total else 0.0

    return SignalResult(features=f, spans=spans)


FEATURE_NAMES = list(analyse("test").features.keys())


def vectorise(texts):
    return [[analyse(t).features[k] for k in FEATURE_NAMES] for t in texts]


if __name__ == "__main__":
    from normalize import console_utf8

    console_utf8()
    demo = [
        "پیرۆزە! ژمارەکەت 5000000 دینار بردویەتەوە. خێرا پەیوەندی بکە 0750 000 0000",
        "کۆدی پشتڕاستکردنەوەت 458213 ە. لەگەڵ هیچ کەسێکدا بەشی مەکە.",
        "عزيزي المشترك، حسابك في زين كاش موقوف. فعّله فورا عبر http://zaincash-iq.top/x",
    ]
    for t in demo:
        r = analyse(t)
        on = {k: v for k, v in r.features.items() if v}
        print(t)
        print("  fired:", ", ".join(f"{k}={v:g}" for k, v in on.items()))
        print()
    print("total features:", len(FEATURE_NAMES))
