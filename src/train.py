"""Train, compare and persist the scam classifier.

Logistic regression is the default and not as a placeholder. It trains in
seconds on CPU, it produces probabilities worth showing a user, and its
coefficients *are* the explanation -- `predict.py` reads them directly to say
which n-grams and which signals moved a verdict. LinearSVC and a random forest
are trained alongside for comparison; whichever wins is reported honestly, but
the demo ships the model that can explain itself.

`build_pipeline` and `group_split` are imported by `evaluate.py`. Every
transformer here is a module-level function or a functools.partial over one,
because joblib cannot pickle a lambda.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MaxAbsScaler, StandardScaler,
)
from sklearn.svm import LinearSVC

import signals
from normalize import console_utf8, preprocess

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"


# --- transformer callables (module level so joblib can pickle them) ----------

def _prep(texts, normalise: bool = True, fold: bool = True):
    return [preprocess(str(t), normalise=normalise, fold=fold) for t in texts]


def _sigs(texts):
    return np.asarray(signals.vectorise([str(t) for t in texts]), dtype=float)


def build_pipeline(
    model: str = "logreg",
    *,
    use_char: bool = True,
    use_word: bool = True,
    use_signals: bool = True,
    normalise: bool = True,
    fold: bool = True,
    scaler: str = "maxabs",
) -> Pipeline:
    """Assemble the feature union and classifier.

    Character n-grams are the primary representation. Kurdish is morphologically
    rich, spelling is inconsistent, the script is mixed, and the corpus is
    small -- word features fail outright where sub-word features only degrade.
    """
    prep = partial(_prep, normalise=normalise, fold=fold)
    parts = []

    if use_char:
        parts.append((
            "char",
            Pipeline([
                ("prep", FunctionTransformer(prep)),
                ("tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5),
                    min_df=2, sublinear_tf=True, max_features=60000,
                )),
            ]),
        ))
    if use_word:
        parts.append((
            "word",
            Pipeline([
                ("prep", FunctionTransformer(prep)),
                ("tfidf", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    min_df=2, sublinear_tf=True,
                )),
            ]),
        ))
    if use_signals:
        # Signals read the raw message, never the folded form -- they need the
        # real domain that entity folding deliberately throws away.
        # MaxAbsScaler, not StandardScaler, and the reason is explainability
        # rather than accuracy. Centering gives an *absent* feature a negative
        # value, which multiplied by a negative coefficient contributes
        # *towards* scam -- so the demo would cite "company name plus a link"
        # as evidence on a message containing no link. Scaling without
        # centering keeps absent signals at exactly zero, so they contribute
        # nothing and every reason shown is a thing actually in the message.
        scale = MaxAbsScaler() if scaler == "maxabs" else StandardScaler()
        parts.append((
            "signals",
            Pipeline([
                ("extract", FunctionTransformer(_sigs)),
                ("scale", scale),
            ]),
        ))

    if not parts:
        raise ValueError("at least one feature block must be enabled")

    if model == "logreg":
        clf = LogisticRegression(
            class_weight="balanced", solver="liblinear", C=4.0, max_iter=2000,
        )
    elif model == "svc":
        clf = LinearSVC(class_weight="balanced", C=0.5, max_iter=5000)
    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1, random_state=0,
        )
    else:
        raise ValueError(f"unknown model: {model}")

    return Pipeline([("features", FeatureUnion(parts)), ("clf", clf)])


def load_corpus(path: Path = CORPUS) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run `python src/build_dataset.py` first."
        )
    df = pd.read_csv(path, encoding="utf-8")
    df["text"] = df["text"].astype(str)
    return df


def group_split(df: pd.DataFrame, *, n_splits: int = 5, fold: int = 0,
                group_col: str = "seed_id"):
    """Group-aware, doubly-stratified split.

    Grouping defaults to `seed_id`: all variants generated from one seed land
    in the same fold. Without that, a paraphrase of a training message shows up
    in test and the score measures the template expander, not the model.

    For cross-lingual work pass ``group_col="concept_id"``. The two seed files
    are parallel translations, so `seed_id` alone still lets the Arabic twin of
    a Kurdish test message sit in the training set.

    Stratification is on label *and* language jointly, so each fold keeps the
    scam/ham and ckb/ar balance.
    """
    strata = df["label"] + "_" + df["lang"]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (tr, te) in enumerate(splitter.split(df, strata, groups=df[group_col])):
        if i == fold:
            return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    raise ValueError("fold out of range")


def _report(name, y_true, y_pred, langs, elapsed):
    from sklearn.metrics import f1_score

    print(f"\n### {name}  (fit+predict {elapsed:.1f}s)")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "scam"])
    print("confusion matrix (rows true ham/scam, cols pred ham/scam):")
    print(cm)
    for lang in sorted(set(langs)):
        m = np.asarray(langs) == lang
        f1 = f1_score(y_true[m], np.asarray(y_pred)[m], pos_label="scam", zero_division=0)
        print(f"  {lang}: scam-F1 {f1:.3f}  (n={int(m.sum())})")
    return f1_score(y_true, y_pred, pos_label="scam", zero_division=0)


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Train the scam classifier.")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--ship", default="logreg", choices=["logreg", "svc", "rf"],
                    help="model to persist for the demo")
    args = ap.parse_args()

    df = load_corpus(args.corpus)
    train, test = group_split(df)
    print(f"corpus {len(df)}  ->  train {len(train)}  test {len(test)}")
    print(f"train seeds {train.seed_id.nunique()}  test seeds {test.seed_id.nunique()}")
    overlap = set(train.seed_id) & set(test.seed_id)
    assert not overlap, f"seed leakage across the split: {overlap}"
    print("seed overlap: none")

    scores = {}
    total = time.perf_counter()
    for name in ("logreg", "svc", "rf"):
        t0 = time.perf_counter()
        pipe = build_pipeline(name)
        pipe.fit(train["text"], train["label"])
        pred = pipe.predict(test["text"])
        scores[name] = _report(name, test["label"], pred, test["lang"], time.perf_counter() - t0)
    print(f"\ntotal training time: {time.perf_counter() - total:.1f}s")

    print("\nscam-F1 on held-out seeds:")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:7s} {v:.3f}")

    # Refit the shipped model on everything: the split existed to measure, and
    # the demo should use all the data available to it.
    ship = build_pipeline(args.ship)
    ship.fit(df["text"], df["label"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": ship, "signal_names": signals.FEATURE_NAMES,
                 "model": args.ship, "n_rows": len(df)}, args.out)
    print(f"\nshipped {args.ship} refit on all {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    # Run through the imported module rather than __main__. The pipeline holds
    # FunctionTransformers wrapping `_prep` and `_sigs`; pickled from __main__
    # they are recorded as `__main__._prep`, and predict.py -- a different
    # __main__ -- then cannot resolve them when it loads the model.
    import train

    train.main()
