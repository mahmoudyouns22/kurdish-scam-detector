"""Inference with a per-message explanation.

The explanation is not a bolt-on. For a linear model the contribution of a
feature to one decision is exactly ``coefficient * feature_value``, so the
numbers below are the model's actual reasoning rather than a story told about
it afterwards. That is the whole argument for shipping logistic regression
here: a tool that tells a non-technical person *why* a message is a scam
teaches them to spot the next one.

Spans are mapped back to the original message through the offset map in
`normalize.normalize_traced`, so a highlight points at the characters the user
actually typed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

import signals
from normalize import console_utf8

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"

# Human-readable glosses for the hand-built signals, in all three languages the
# demo speaks.
SIGNAL_LABELS = {
    "url_present": ("Contains a link", "لینکی تێدایە", "يحتوي على رابط"),
    "url_count": ("Number of links", "ژمارەی لینکەکان", "عدد الروابط"),
    "url_shortener": ("Shortened link", "لینکی کورتکراوە", "رابط مختصر"),
    "url_suspicious_tld": ("Suspicious domain ending", "دوماینی گومانلێکراو", "نطاق مشبوه"),
    "url_raw_ip": ("Link is a raw IP address", "لینک ژمارەی IP ـە", "الرابط عنوان IP"),
    "url_brand_lookalike": ("Domain imitates a real brand", "دوماین لاسایی براندێک دەکاتەوە", "نطاق يقلد علامة حقيقية"),
    "phone_present": ("Contains a phone number", "ژمارەی مۆبایلی تێدایە", "يحتوي على رقم هاتف"),
    "phone_count": ("Number of phone numbers", "ژمارەی مۆبایلەکان", "عدد ارقام الهاتف"),
    "money_present": ("Mentions a money amount", "باسی بڕی پارە دەکات", "يذكر مبلغ مالي"),
    "money_magnitude": ("Size of the amount", "قەبارەی بڕەکە", "حجم المبلغ"),
    "lex_urgency": ("Urgency wording", "وشەی پەلەکردن", "الفاظ استعجال"),
    "lex_prize": ("Prize or winning wording", "باسی خەڵات و بردنەوە", "الفاظ جوائز وربح"),
    "lex_sensitive": ("Asks for sensitive data", "داوای زانیاری هەستیار دەکات", "يطلب معلومات حساسة"),
    "lex_asks_for_code": ("Asks you to send a code", "داوای ناردنی کۆد دەکات", "يطلب ارسال الرمز"),
    "lex_provides_code": ("Gives you a code (normal OTP)", "کۆدت پێدەدات (ئاسایی)", "يعطيك رمزا (طبيعي)"),
    "lex_action_link": ("Pushes you to open a link", "پاڵت پێوەدەنێت لینک بکەیتەوە", "يحثك على فتح رابط"),
    "lex_pay_fee": ("Asks for a payment or fee", "داوای پارە یان کرێ دەکات", "يطلب دفع مبلغ او رسوم"),
    "otp_asks_not_provides": ("Asks for a code instead of giving one", "کۆد دەخوازێت لەبری ئەوەی بیدات", "يطلب الرمز بدل ان يعطيه"),
    "brand_mention": ("Mentions a company", "ناوی کۆمپانیایەک", "يذكر شركة"),
    "brand_and_link": ("Company name plus a link", "ناوی کۆمپانیا + لینک", "اسم شركة مع رابط"),
    "brand_and_lookalike": ("Company name plus a fake domain", "ناوی کۆمپانیا + دوماینی ساختە", "اسم شركة مع نطاق مزيف"),
    "brand_and_urgency": ("Company name plus urgency", "ناوی کۆمپانیا + پەلەکردن", "اسم شركة مع استعجال"),
    "length": ("Message length", "درێژی نامەکە", "طول الرسالة"),
    "digit_ratio": ("Proportion of digits", "ڕێژەی ژمارەکان", "نسبة الارقام"),
    "exclaim_count": ("Exclamation marks", "هێمای سەرسوڕمان", "علامات تعجب"),
    "punct_density": ("Punctuation density", "چڕی خاڵبەندی", "كثافة علامات الترقيم"),
    "upper_ratio": ("Capital letters", "پیتی گەورە", "حروف كبيرة"),
    "script_mix_ratio": ("Mixes two scripts", "تێکەڵکردنی دوو ئەلفوبێ", "خلط بين نصين"),
}

VERDICTS = {
    "scam": {
        "en": "Likely a SCAM",
        "ckb": "بە گومانەوە ساختەیە",
        "ar": "على الارجح احتيال",
    },
    "ham": {
        "en": "Looks legitimate",
        "ckb": "ئاسایی دیارە",
        "ar": "تبدو رسالة عادية",
    },
}


@dataclass
class Contribution:
    name: str
    kind: str          # "signal" or "ngram"
    value: float
    weight: float
    contribution: float


@dataclass
class Prediction:
    text: str
    label: str
    probability: float
    contributions: list
    spans: list

    @property
    def verdict(self) -> dict:
        return VERDICTS[self.label]


_CACHE: dict = {}


def load_model(path: Path = MODEL_PATH):
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise SystemExit(
                f"{path} not found -- run `python src/train.py` first."
            )
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def _feature_names(pipe) -> np.ndarray:
    """Names for every column of the feature union, in order."""
    names = []
    for block_name, block in pipe.named_steps["features"].transformer_list:
        if block_name == "signals":
            names.extend(f"signal::{n}" for n in signals.FEATURE_NAMES)
        else:
            vocab = block.named_steps["tfidf"].get_feature_names_out()
            names.extend(f"{block_name}::{v}" for v in vocab)
    return np.asarray(names, dtype=object)


def predict(text: str, *, model_path: Path = MODEL_PATH, top_k: int = 8) -> Prediction:
    bundle = load_model(model_path)
    pipe = bundle["pipeline"]
    clf = pipe.named_steps["clf"]

    scam_idx = list(clf.classes_).index("scam")
    proba = float(pipe.predict_proba([text])[0][scam_idx])
    label = "scam" if proba >= 0.5 else "ham"

    # coefficient * value is the exact per-feature contribution to the log-odds.
    X = pipe.named_steps["features"].transform([text])
    coef = clf.coef_[0]
    names = _feature_names(pipe)

    x = np.asarray(X.todense()).ravel() if hasattr(X, "todense") else np.asarray(X).ravel()
    contrib = coef * x

    # Rank by contribution toward the predicted class.
    order = np.argsort(-contrib if label == "scam" else contrib)
    contributions = []
    for i in order[: top_k * 4]:
        if abs(contrib[i]) < 1e-9:
            continue
        raw = names[i]
        kind, _, feat = raw.partition("::")
        contributions.append(Contribution(
            name=feat, kind="signal" if kind == "signal" else kind,
            value=float(x[i]), weight=float(coef[i]),
            contribution=float(contrib[i]),
        ))
        if len(contributions) >= top_k:
            break

    return Prediction(
        text=text, label=label, probability=proba,
        contributions=contributions, spans=signals.analyse(text).spans,
    )


def describe(c: Contribution, lang: str = "en") -> str:
    if c.kind == "signal":
        gloss = SIGNAL_LABELS.get(c.name)
        if gloss:
            return {"en": gloss[0], "ckb": gloss[1], "ar": gloss[2]}[lang]
        return c.name
    return f"text pattern {c.name!r}"


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Classify a message and explain why.")
    ap.add_argument("text", nargs="*", help="message text; omit to read stdin")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    texts = [" ".join(args.text)] if args.text else [
        l.rstrip("\n") for l in sys.stdin if l.strip()
    ]
    if not texts or not texts[0]:
        raise SystemExit("no input text")

    for t in texts:
        r = predict(t, model_path=args.model, top_k=args.top)
        print("\n" + "-" * 70)
        print(t)
        print(f"\n  {r.verdict['en']}   ({r.probability:.1%} scam)")
        print(f"  {r.verdict['ckb']}")
        print(f"  {r.verdict['ar']}")
        print("\n  why:")
        for c in r.contributions:
            arrow = "scam" if c.contribution > 0 else "ham "
            print(f"    {arrow} {c.contribution:+.3f}  {describe(c)}"
                  + (f"  (value {c.value:g})" if c.kind == "signal" else ""))


if __name__ == "__main__":
    main()
