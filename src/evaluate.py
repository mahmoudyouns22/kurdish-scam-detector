"""Evaluation harness: held-out, cross-lingual transfer, ablations, adversarial.

Accuracy alone is not a result, so nothing here reports it alone. The three
experiments that matter:

* **Transfer** -- train on Arabic, test on Kurdish, and the reverse. Arabic is
  the well-resourced language and Kurdish the low-resource target, so how far a
  character n-gram model carries across the shared script is the actual
  research question this project can answer.
* **Ablation** -- normalisation on/off, char/word/hybrid, n-grams with and
  without the hand-built signals. If normalisation shows no gain, that is worth
  knowing and worth saying.
* **Adversarial probe** -- the evasions a real scammer uses. These are held out
  of `build_dataset.py` on purpose; training on them would make this section
  meaningless.

Every split is group-aware on `seed_id`.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from normalize import console_utf8
from train import CORPUS, build_pipeline, group_split, load_corpus

ZERO_WIDTH = ["​", "‌", "‍", "⁠"]
TATWEEL = "ـ"
ARABIC_INDIC = {str(d): chr(0x0660 + d) for d in range(10)}

# Approximate Sorani/Arabic -> Latin. Not a scholarly romanisation; the point
# is to leave the Arabic script entirely and see what survives.
TRANSLIT = {
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "c",
    "چ": "ch", "ح": "h", "خ": "x", "د": "d", "ذ": "z", "ر": "r", "ڕ": "rr",
    "ز": "z", "ژ": "j", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "3", "غ": "gh", "ف": "f", "ڤ": "v", "ق": "q", "ک": "k",
    "ك": "k", "گ": "g", "ل": "l", "ڵ": "ll", "م": "m", "ن": "n", "و": "w",
    "ۆ": "o", "ھ": "h", "ه": "h", "ە": "e", "ی": "y", "ي": "y", "ێ": "e",
    "ئ": "", "ء": "", "ة": "a", "ى": "a", "ژ": "j",
}


# --- adversarial transforms --------------------------------------------------

def ev_zero_width(text: str, rng: random.Random) -> str:
    """Split words with invisible characters -- the cheapest filter evasion."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.strip() and rng.random() < 0.25:
            out.append(rng.choice(ZERO_WIDTH))
    return "".join(out)


def ev_tatweel(text: str, rng: random.Random) -> str:
    return "".join(
        ch + (TATWEEL * rng.randint(1, 3)) if ("ؠ" < ch < "ە" and rng.random() < 0.3) else ch
        for ch in text
    )


def ev_stretch(text: str, rng: random.Random) -> str:
    return "".join(
        ch * rng.randint(3, 5) if (ch.isalpha() and rng.random() < 0.15) else ch
        for ch in text
    )


def ev_digits(text: str, rng: random.Random) -> str:
    return "".join(ARABIC_INDIC.get(c, c) for c in text)


def ev_translit(text: str, rng: random.Random) -> str:
    return "".join(TRANSLIT.get(c, c) for c in text)


def ev_all(text: str, rng: random.Random) -> str:
    return ev_stretch(ev_tatweel(ev_zero_width(text, rng), rng), rng)


EVASIONS = {
    "zero-width insertion": ev_zero_width,
    "tatweel padding": ev_tatweel,
    "letter stretching": ev_stretch,
    "digit-system swap": ev_digits,
    "Latin transliteration": ev_translit,
    "combined (zw+tatweel+stretch)": ev_all,
}


# --- helpers -----------------------------------------------------------------

def scam_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, pos_label="scam", zero_division=0)


def fit_eval(train_df, test_df, **kw):
    pipe = build_pipeline(kw.pop("model", "logreg"), **kw)
    pipe.fit(train_df["text"], train_df["label"])
    pred = pipe.predict(test_df["text"])
    return pipe, pred


def header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --- experiments -------------------------------------------------------------

def exp_heldout(train, test):
    header("1. HELD-OUT (group-aware split on seed_id)")
    pipe, pred = fit_eval(train, test)
    print(classification_report(test["label"], pred, digits=3, zero_division=0))
    cm = confusion_matrix(test["label"], pred, labels=["ham", "scam"])
    print("confusion matrix        pred_ham  pred_scam")
    print(f"  true_ham               {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"  true_scam              {cm[1][0]:6d}    {cm[1][1]:6d}")

    proba = pipe.predict_proba(test["text"])[:, list(pipe.classes_).index("scam")]
    ap = average_precision_score((test["label"] == "scam").astype(int), proba)
    print(f"\naverage precision (PR-AUC, scam): {ap:.3f}")

    print("\nper language:")
    for lang in sorted(test["lang"].unique()):
        m = (test["lang"] == lang).to_numpy()
        p, r, f, _ = precision_recall_fscore_support(
            test["label"][m], np.asarray(pred)[m],
            labels=["scam"], zero_division=0,
        )
        print(f"  {lang:4s} n={int(m.sum()):4d}  P {p[0]:.3f}  R {r[0]:.3f}  F1 {f[0]:.3f}")

    print("\nper scam category (recall):")
    scam_mask = (test["label"] == "scam").to_numpy()
    for cat in sorted(test.loc[scam_mask, "category"].unique()):
        m = scam_mask & (test["category"] == cat).to_numpy()
        rec = (np.asarray(pred)[m] == "scam").mean()
        print(f"  {cat:10s} n={int(m.sum()):4d}  recall {rec:.3f}")
    return pipe, pred


def exp_cv(df, n_splits=5):
    header("1b. CROSS-VALIDATED (5 group-aware folds)")
    print("One fold holds out ~42 seeds, so single-fold numbers swing several")
    print("points and per-category cells can be 2 seeds wide. These are the")
    print("numbers to quote: mean +/- population std over all five folds.\n")
    rows = {}
    for name, kw in (
        ("logreg (shipped)", {"model": "logreg"}),
        ("linear SVC", {"model": "svc"}),
        ("random forest", {"model": "rf"}),
        ("logreg, signals centred", {"model": "logreg", "scaler": "standard"}),
        ("logreg, no normalisation", {"model": "logreg", "normalise": False}),
    ):
        overall, per_lang = [], {"ar": [], "ckb": []}
        for fold in range(n_splits):
            tr, te = group_split(df, n_splits=n_splits, fold=fold)
            _, pred = fit_eval(tr, te, **dict(kw))
            overall.append(scam_f1(te["label"], pred))
            for lang in ("ar", "ckb"):
                m = (te["lang"] == lang).to_numpy()
                per_lang[lang].append(scam_f1(te["label"][m], np.asarray(pred)[m]))
        rows[name] = (overall, per_lang)
        print(f"{name:26s} scam-F1 {np.mean(overall):.3f} +/- {np.std(overall):.3f}"
              f"   ar {np.mean(per_lang['ar']):.3f}"
              f"   ckb {np.mean(per_lang['ckb']):.3f}")
    return rows


def exp_transfer(df):
    header("2. CROSS-LINGUAL TRANSFER (Arabic <-> Kurdish)")
    print("Train on one language, test on the other.\n")
    print("Split is grouped on concept_id, not seed_id. The seed files are")
    print("parallel translations (ar-scam-017 is ckb-scam-017 in Arabic, same")
    print("amounts), so grouping on seed_id alone would put the Arabic twin of")
    print("a Kurdish test message into training and call the result transfer.")
    print("One concept split is reused for every row below, so the six numbers")
    print("are measured on the same held-out concepts and are comparable.\n")

    # One split over the whole corpus, then sliced by language. This keeps the
    # held-out concept set identical across all six train/test combinations.
    tr_all, te_all = group_split(df, group_col="concept_id")
    ar_tr = tr_all[tr_all.lang == "ar"]
    ckb_tr = tr_all[tr_all.lang == "ckb"]
    ar_te = te_all[te_all.lang == "ar"]
    ckb_te = te_all[te_all.lang == "ckb"]
    joint_tr = tr_all

    leak = set(tr_all.concept_id) & set(te_all.concept_id)
    assert not leak, f"concept leakage: {leak}"
    print(f"held-out concepts: {te_all.concept_id.nunique()} "
          f"(train {tr_all.concept_id.nunique()}), concept overlap: none\n")

    rows = []
    for name, tr, te in [
        ("ar  -> ar   (in-language)", ar_tr, ar_te),
        ("ckb -> ckb  (in-language)", ckb_tr, ckb_te),
        ("ar  -> ckb  (transfer)", ar_tr, ckb_te),
        ("ckb -> ar   (transfer)", ckb_tr, ar_te),
        ("joint -> ckb", joint_tr, ckb_te),
        ("joint -> ar", joint_tr, ar_te),
    ]:
        _, pred = fit_eval(tr, te)
        p, r, f, _ = precision_recall_fscore_support(
            te["label"], pred, labels=["scam"], zero_division=0
        )
        acc = (np.asarray(pred) == te["label"].to_numpy()).mean()
        rows.append((name, len(tr), len(te), p[0], r[0], f[0], acc))
        print(f"{name:28s} train {len(tr):5d} test {len(te):5d}  "
              f"P {p[0]:.3f}  R {r[0]:.3f}  scam-F1 {f[0]:.3f}  acc {acc:.3f}")

    # The signals module is language-independent by construction: its lexicons
    # cover Kurdish and Arabic regardless of which language was trained on. So
    # the table above credits hand-built cross-lingual rules as "transfer".
    # Disabling signals isolates what the shared script actually carries -- and
    # the in-language rows have to be rerun the same way to compare fairly.
    print("\nn-grams only (signals disabled) -- isolates what the script carries:")
    for name, tr, te in [
        ("ar  -> ar   (in-language)", ar_tr, ar_te),
        ("ckb -> ckb  (in-language)", ckb_tr, ckb_te),
        ("ar  -> ckb  (transfer)", ar_tr, ckb_te),
        ("ckb -> ar   (transfer)", ckb_tr, ar_te),
    ]:
        _, pred = fit_eval(tr, te, use_signals=False)
        print(f"  {name:28s} scam-F1 {scam_f1(te['label'], pred):.3f}")
    return rows


def exp_ablation(train, test):
    header("3. ABLATION")
    configs = [
        ("full (char+word+signals, normalised)", {}),
        ("no normalisation", {"normalise": False}),
        ("no entity folding", {"fold": False}),
        ("no normalisation, no folding", {"normalise": False, "fold": False}),
        ("char n-grams only", {"use_word": False, "use_signals": False}),
        ("word n-grams only", {"use_char": False, "use_signals": False}),
        ("char+word, no signals", {"use_signals": False}),
        ("signals only", {"use_char": False, "use_word": False}),
        ("signals centred (StandardScaler)", {"scaler": "standard"}),
    ]
    rows = []
    for name, kw in configs:
        _, pred = fit_eval(train, test, **kw)
        f = scam_f1(test["label"], pred)
        ckb = (test["lang"] == "ckb").to_numpy()
        f_ckb = scam_f1(test["label"][ckb], np.asarray(pred)[ckb])
        rows.append((name, f, f_ckb))
        print(f"{name:38s} scam-F1 {f:.3f}   (ckb {f_ckb:.3f})")
    return rows


def exp_adversarial(pipe, test, seed=11):
    header("4. ADVERSARIAL PROBE")
    print("Evasions are applied to the WHOLE held-out set, both classes.")
    print("Scoring only evaded scam messages is misleading: mangling text")
    print("shreds it into rare n-grams that look nothing like clean ham, so")
    print("the model drifts toward 'scam' and recall appears to *improve*")
    print("while precision quietly collapses. The false-positive rate on")
    print("evaded ham is the column that exposes that.")
    print("These transforms never appear in training.\n")
    rng = random.Random(seed)
    y = test["label"].to_numpy()
    ham_m, scam_m = y == "ham", y == "scam"

    def row(name, texts):
        pred = np.asarray(pipe.predict(texts))
        rec = (pred[scam_m] == "scam").mean()
        fpr = (pred[ham_m] == "scam").mean()
        f1 = scam_f1(y, pred)
        print(f"{name:32s} F1 {f1:.3f}  recall {rec:.3f}  FPR(ham) {fpr:.3f}")
        return name, f1, rec, fpr

    base = row("baseline (no evasion)", list(test["text"]))
    rows = [base]
    for name, fn in EVASIONS.items():
        adv = [fn(t, rng) for t in test["text"]]
        r = row(name, adv)
        rows.append(r)
    print(f"\n{'':32s} {'F1 drop vs baseline':>24s}")
    for name, f1, _, _ in rows[1:]:
        print(f"{name:32s} {base[1] - f1:+.3f}")
    return rows


def exp_normalisation_under_shift(train, test, seed=13):
    header("3b. NORMALISATION UNDER ORTHOGRAPHIC SHIFT")
    print("The i.i.d. ablation shows normalisation doing nothing, and that is")
    print("expected: train and test are perturbed by the same generator, so")
    print("the model simply memorises both spellings. Normalisation earns its")
    print("place when test-time orthography differs from training-time, which")
    print("is what this measures -- train on the normal training set, test on")
    print("a test set pushed into script variants the model has not seen.\n")
    rng = random.Random(seed)
    shifted = [ev_tatweel(ev_zero_width(t, rng), rng) for t in test["text"]]

    for label, kw in (("with normalisation", {}), ("without normalisation", {"normalise": False})):
        pipe = build_pipeline("logreg", **kw)
        pipe.fit(train["text"], train["label"])
        clean = scam_f1(test["label"], pipe.predict(test["text"]))
        shift = scam_f1(test["label"], pipe.predict(shifted))
        print(f"{label:24s} clean F1 {clean:.3f}   shifted F1 {shift:.3f}   "
              f"delta {shift - clean:+.3f}")


def exp_collected(df):
    header("5. COLLECTED-DATA EVALUATION")
    col = df[df["source"] == "collected"]
    if col.empty:
        print("No rows with source='collected' in the corpus.")
        print("Every number above is measured on AUTHORED synthetic data and")
        print("therefore reports pattern separability, not field performance.")
        print("See data/README.md for how to contribute real messages.")
        return None
    auth = df[df["source"] == "authored"]
    pipe = build_pipeline("logreg")
    pipe.fit(auth["text"], auth["label"])
    pred = pipe.predict(col["text"])
    print(f"trained on {len(auth)} authored rows, tested on {len(col)} collected rows")
    print(classification_report(col["label"], pred, digits=3, zero_division=0))
    return scam_f1(col["label"], pred)


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Run the evaluation suite.")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--collected-only", action="store_true",
                    help="evaluate only on rows with source='collected'")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["heldout", "transfer", "ablation", "adversarial"])
    args = ap.parse_args()

    df = load_corpus(args.corpus)
    if args.collected_only:
        df = df[df["source"] == "collected"].reset_index(drop=True)
        if df.empty:
            raise SystemExit("no collected rows in the corpus; nothing to evaluate")

    train, test = group_split(df)
    print(f"corpus {len(df)} rows / {df.seed_id.nunique()} seeds")
    print(f"train {len(train)} ({train.seed_id.nunique()} seeds)  "
          f"test {len(test)} ({test.seed_id.nunique()} seeds)")

    pipe = None
    if "heldout" not in args.skip:
        pipe, _ = exp_heldout(train, test)
        exp_cv(df)
    if "transfer" not in args.skip:
        exp_transfer(df)
    if "ablation" not in args.skip:
        exp_ablation(train, test)
        exp_normalisation_under_shift(train, test)
    if "adversarial" not in args.skip:
        if pipe is None:
            pipe, _ = fit_eval(train, test)
        exp_adversarial(pipe, test)
    exp_collected(df)


if __name__ == "__main__":
    main()
