"""Verify the JavaScript engine agrees with the Python pipeline.

The web demo reimplements normalisation, entity folding, both TF-IDF analyzers
and the linear model in JavaScript. That is a second implementation of the same
maths, and two implementations drift unless something checks them.

This runs every corpus row through both and reports the maximum absolute
difference in P(scam) plus any label disagreement. A demo whose numbers differ
from the reported evaluation is worse than no demo, so this is a gate, not a
formality.

    python web/verify_parity.py            # full corpus
    python web/verify_parity.py --n 500    # quick check

Requires Node.js on PATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from normalize import console_utf8  # noqa: E402

WEB = ROOT / "web"

RUNNER = r"""
const fs = require('fs');
const path = require('path');
global.window = global;
require(path.join(__dirname, 'engine.js'));
const model = JSON.parse(fs.readFileSync(path.join(__dirname, 'model.json'), 'utf8'));
const eng = new ScamEngine.Engine(model);
const texts = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = texts.map(t => {
  const r = eng.predict(t);
  return {p: r.probability, label: r.label, norm: r.normalized, fold: r.folded};
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Python vs JavaScript parity check.")
    ap.add_argument("--n", type=int, default=0, help="limit rows (0 = all)")
    ap.add_argument("--tol", type=float, default=1e-6, help="max allowed |dP|")
    args = ap.parse_args()

    corpus = ROOT / "data" / "corpus.csv"
    if not corpus.exists():
        raise SystemExit("run `python src/build_dataset.py` first")
    model_json = WEB / "model.json"
    if not model_json.exists():
        raise SystemExit("run `python src/export_web.py` first")

    df = pd.read_csv(corpus, encoding="utf-8")
    if args.n:
        df = df.sample(n=min(args.n, len(df)), random_state=0)
    texts = [str(t) for t in df["text"]]

    # Python side
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    pipe = bundle["pipeline"]
    scam_i = list(pipe.named_steps["clf"].classes_).index("scam")
    py = pipe.predict_proba(texts)[:, scam_i]

    # JavaScript side
    with tempfile.TemporaryDirectory() as td:
        runner = WEB / "_parity_runner.js"
        tin = Path(td) / "in.json"
        tout = Path(td) / "out.json"
        runner.write_text(RUNNER, encoding="utf-8")
        tin.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        try:
            proc = subprocess.run(
                ["node", str(runner), str(tin), str(tout)],
                capture_output=True, text=True, encoding="utf-8",
            )
        finally:
            runner.unlink(missing_ok=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit("node failed")
        js = json.loads(tout.read_text(encoding="utf-8"))

    diffs = [abs(py[i] - js[i]["p"]) for i in range(len(texts))]
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    py_lab = ["scam" if p >= 0.5 else "ham" for p in py]
    mismatch = [i for i in range(len(texts)) if py_lab[i] != js[i]["label"]]

    print(f"rows compared        : {len(texts)}")
    print(f"max |dP(scam)|       : {max_d:.3e}")
    print(f"mean |dP(scam)|      : {mean_d:.3e}")
    print(f"label disagreements  : {len(mismatch)}")

    if mismatch:
        print("\nfirst disagreements:")
        for i in mismatch[:5]:
            print(f"  py={py[i]:.4f} js={js[i]['p']:.4f}  {texts[i][:70]}")

    worst = max(range(len(texts)), key=lambda i: diffs[i])
    if max_d > args.tol:
        print(f"\nworst row (|dP| {diffs[worst]:.3e}):")
        print(f"  text : {texts[worst][:90]}")
        print(f"  py   : {py[worst]:.10f}")
        print(f"  js   : {js[worst]['p']:.10f}")

    ok = max_d <= args.tol and not mismatch
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
