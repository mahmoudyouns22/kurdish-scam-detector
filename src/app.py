"""Agadar — Streamlit desktop app.

Run with:  python -m streamlit run src/app.py

The browser demo in web/ is the one to show people; this is the local tool that
runs directly against models/model.joblib, so it always reflects whatever you
just trained rather than the last exported snapshot.

Fully offline. Nothing is uploaded, logged or sent anywhere -- which is the only
honest way to ask someone to paste an SMS that may contain their bank details.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import MODEL_PATH, SIGNAL_LABELS, describe, predict  # noqa: E402

st.set_page_config(
    page_title="Agadar — Kurdish & Iraqi Arabic Scam Detector",
    page_icon="🛡️",
    layout="centered",
)

# Palette mirrors web/template.html so the desktop tool and the browser demo
# read as one product. Streamlit's own theme lives in .streamlit/config.toml;
# this covers what the theme file cannot reach.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600;700&display=swap');

:root{
  --accent:#C08A2E; --danger:#D9584A; --safe:#4E9B72;
  --arab:"IBM Plex Sans Arabic","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --serif:"IBM Plex Serif",Georgia,serif;
}
html, body, [class*="css"]{font-family:"IBM Plex Sans",system-ui,sans-serif}
#MainMenu, footer{visibility:hidden}
.block-container{padding-top:2.2rem;max-width:820px}

.brand{display:flex;align-items:baseline;gap:10px;margin-bottom:2px}
.brand .ar{font-family:var(--arab);font-size:27px;font-weight:700;color:var(--accent);line-height:1}
.brand .lat{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;opacity:.6}
.tagline{font-family:var(--serif);font-size:23px;font-weight:600;line-height:1.25;margin:10px 0 4px}
.sub{opacity:.72;font-size:14.5px;margin-bottom:6px}
.privacy{font-family:var(--mono);font-size:11px;color:var(--safe);margin-bottom:2px}

/* RTL input: Arabic script left-aligned is the first thing a Kurdish speaker
   notices about a tool built by someone who does not read it. */
.stTextArea textarea{
  direction:rtl;text-align:right;font-family:var(--arab)!important;
  font-size:16.5px!important;line-height:1.9!important
}

.stamp{font-family:var(--serif);font-size:26px;font-weight:700;margin:0 0 2px}
.stamp.scam{color:var(--danger)}
.stamp.ham{color:var(--safe)}
.conf{font-family:var(--mono);font-size:12.5px;opacity:.65;margin-bottom:10px}

.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:6px 0 4px}
.vbox{padding:9px 12px;border-radius:9px;border:1px solid rgba(128,128,128,.22)}
.vbox b{display:block;font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;opacity:.55;font-weight:400;margin-bottom:3px}
.vbox span{font-family:var(--arab);font-size:16px;font-weight:600}
.vbox.en span{font-family:"IBM Plex Sans",sans-serif;font-size:15px}
.vbox.rtl{direction:rtl;text-align:right}
.vbox.scam span{color:var(--danger)}
.vbox.ham span{color:var(--safe)}

.bubble{
  border:1px solid rgba(128,128,128,.22);border-radius:14px 14px 14px 4px;
  padding:14px 16px;font-family:var(--arab);font-size:16.5px;line-height:2.05;
  direction:rtl;text-align:right;overflow-wrap:anywhere
}
mark{padding:1px 3px;border-radius:4px;color:inherit!important}

.rrow{display:flex;align-items:center;gap:12px;padding:8px 0;
  border-bottom:1px solid rgba(128,128,128,.14)}
.rrow:last-child{border-bottom:none}
.rmain{flex:1;min-width:0}
.rname{font-size:14px;font-weight:500}
.rtrans{font-family:var(--arab);font-size:12.5px;opacity:.6;direction:rtl;text-align:right}
.rtrack{width:80px;height:5px;border-radius:99px;background:rgba(128,128,128,.2);overflow:hidden;flex:none}
.rtrack i{display:block;height:100%;border-radius:99px}
.rval{font-family:var(--mono);font-size:11.5px;opacity:.65;min-width:46px;text-align:right;flex:none}
.lgd{font-family:var(--mono);font-size:10.5px;opacity:.6;margin-top:9px}
</style>
""",
    unsafe_allow_html=True,
)

SPAN_COLOURS = {
    "link": "rgba(217,88,74,.28)",
    "phone": "rgba(192,138,46,.30)",
    "money": "rgba(192,138,46,.30)",
    "brand": "rgba(128,128,128,.28)",
    "urgency": "rgba(217,88,74,.28)",
    "prize": "rgba(217,88,74,.28)",
    "sensitive": "rgba(217,88,74,.34)",
    "asks_for_code": "rgba(217,88,74,.38)",
    "provides_code": "rgba(78,155,114,.32)",
    "action_link": "rgba(217,88,74,.24)",
    "pay_fee": "rgba(192,138,46,.30)",
}
SPAN_NAMES = {
    "link": "link", "phone": "phone number", "money": "amount", "brand": "company",
    "urgency": "urgency", "prize": "prize wording", "sensitive": "asks for private data",
    "asks_for_code": "asks for a code", "provides_code": "gives a code",
    "action_link": "pushes a link", "pay_fee": "asks for payment",
}

EXAMPLES = [
    ("Kurdish · prize scam", "scam",
     "پیرۆزە! ژمارەکەت 10000000 دینار بردویەتییەوە. خێرا پەیوەندی بکە 0750 000 0000"),
    ("Kurdish · fake Zain Cash", "scam",
     "ئاگاداری: ئەکاونتی زین کاشەکەت ڕاگیراوە. بۆ چالاککردنەوەی خێرا کلیک بکە لەسەر http://zaincash-iq.top/verify"),
    ("Kurdish · OTP theft", "scam",
     "کۆدێکت بۆ هات، تکایە کۆدەکە بنێرەوە بۆم تاکو گواستنەوەکە تەواو بکەم"),
    ("Kurdish · real OTP", "ham",
     "کۆدی پشتڕاستکردنەوەت 458213 ە. لەگەڵ هیچ کەسێکدا بەشی مەکە"),
    ("Kurdish · real promo", "ham",
     "ئاسیاسێل: پاکێجی نوێی ئینتەرنێت 20 گیگا بە 15000 دینار. بۆ زانیاری زیاتر سەردانی asiacell.com بکە"),
    ("Arabic · account frozen", "scam",
     "تنبيه: حسابك بزين كاش موقوف. لاعادة التفعيل اضغط فورا http://zaincash-iq.top/verify"),
    ("Arabic · customs fee", "scam",
     "وصلك طرد بس متوقف بالكمرك. للافراج عنه ادفع 25000 دينار على 0750 000 0000"),
    ("Arabic · real OTP", "ham",
     "رمز التحقق الخاص بك هو 458213. لا تشاركه مع اي شخص"),
]

# url_present/url_count carry identical weight on a one-link message, so showing
# both prints the same reason twice. Keep the one a person would say out loud.
DEDUPE = {"url_count": "url_present", "phone_count": "phone_present"}


def merge_spans(spans):
    """Flatten overlapping spans, keeping the first label for each region."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    out = []
    for lo, hi, label in ordered:
        if out and lo < out[-1][1]:
            if hi > out[-1][1]:
                out.append((out[-1][1], hi, label))
            continue
        out.append((lo, hi, label))
    return out


def highlight(text, spans):
    parts, cursor = [], 0
    for lo, hi, label in merge_spans(spans):
        lo, hi = max(lo, cursor), min(hi, len(text))
        if hi <= lo:
            continue
        parts.append(html.escape(text[cursor:lo]))
        colour = SPAN_COLOURS.get(label, "rgba(128,128,128,.25)")
        parts.append(
            f'<mark style="background:{colour}" title="{html.escape(label)}">'
            f"{html.escape(text[lo:hi])}</mark>"
        )
        cursor = hi
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


st.markdown(
    '<div class="brand"><span class="ar">ئاگادار</span>'
    '<span class="lat">Agadar</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="tagline">Scam messages, caught in the language they arrive in.</div>'
    '<div class="sub">Sorani Kurdish and Iraqi Arabic. Paste a message to see the verdict '
    "and the exact words behind it.</div>"
    '<div class="privacy">● Runs offline against your local model — nothing is uploaded.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. From the project root run:\n\n"
        "```\npython src/build_dataset.py\npython src/train.py\n```"
    )
    st.stop()

if "message" not in st.session_state:
    st.session_state.message = EXAMPLES[0][2]

st.write("")
cols = st.columns(2)
for i, (label, _kind, text) in enumerate(EXAMPLES):
    if cols[i % 2].button(label, use_container_width=True, key=f"ex{i}"):
        st.session_state.message = text

message = st.text_area(
    "Message", key="message", height=125,
    help="Sorani Kurdish, Iraqi Arabic, or a mix.",
)

if not message.strip():
    st.info("Paste a message, or pick one of the examples above.")
    st.stop()

result = predict(message, top_k=10)
scam = result.label == "scam"
kind = "scam" if scam else "ham"
conf = result.probability if scam else 1 - result.probability

st.divider()
st.markdown(
    f'<div class="stamp {kind}">{"Likely a scam" if scam else "Looks legitimate"}</div>'
    f'<div class="conf">{conf:.1%} confidence · P(scam) {result.probability:.3f}</div>',
    unsafe_allow_html=True,
)
st.progress(result.probability)

st.markdown(
    f'<div class="verdicts">'
    f'<div class="vbox en {kind}"><b>English</b><span>'
    f'{"Likely a SCAM" if scam else "Looks legitimate"}</span></div>'
    f'<div class="vbox rtl {kind}"><b>کوردی</b><span>'
    f'{"بە گومانەوە ساختەیە" if scam else "ئاسایی دیارە"}</span></div>'
    f'<div class="vbox rtl {kind}"><b>العربية</b><span>'
    f'{"على الأرجح احتيال" if scam else "تبدو رسالة عادية"}</span></div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("##### What triggered it")
st.markdown(f'<div class="bubble">{highlight(message, result.spans)}</div>',
            unsafe_allow_html=True)

fired = {label for _, _, label in result.spans}
if fired:
    st.markdown(
        '<div class="lgd">highlighted: '
        + " · ".join(SPAN_NAMES.get(f, f) for f in sorted(fired))
        + "</div>",
        unsafe_allow_html=True,
    )

st.markdown("##### Why")
shown = {c.name for c in result.contributions if c.kind == "signal"}
signal_rows = [
    c for c in result.contributions
    if c.kind == "signal" and not (DEDUPE.get(c.name) and DEDUPE[c.name] in shown)
][:7]

if signal_rows:
    peak = max(abs(c.contribution) for c in signal_rows)
    rows = []
    for c in signal_rows:
        gloss = SIGNAL_LABELS.get(c.name)
        width = max(4, abs(c.contribution) / peak * 100) if peak else 0
        colour = "var(--danger)" if c.contribution > 0 else "var(--safe)"
        trans = f"{gloss[1]} · {gloss[2]}" if gloss else ""
        rows.append(
            f'<div class="rrow"><div class="rmain">'
            f'<div class="rname">{html.escape(describe(c))}</div>'
            f'<div class="rtrans">{html.escape(trans)}</div></div>'
            f'<span class="rtrack"><i style="width:{width:.0f}%;background:{colour}"></i></span>'
            f'<span class="rval">{c.contribution:+.2f}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
else:
    st.write("No named signal fired — the verdict rests on text patterns alone.")

with st.expander("Character and word patterns"):
    ngrams = [c for c in result.contributions if c.kind != "signal"]
    if ngrams:
        st.markdown(
            " ".join(
                f"`{c.name}` {c.contribution:+.3f}" for c in ngrams[:14]
            )
        )
    else:
        st.write("None in the top contributors.")

st.caption(
    "Trained on an authored synthetic corpus — no real intercepted messages yet. "
    "Treat this as a second opinion that explains itself, not a verdict. "
    "See the README's limitations section."
)
