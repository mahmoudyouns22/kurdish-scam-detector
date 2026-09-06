"""Bake the browser demo into one self-contained HTML file.

The published artifact cannot fetch a sidecar file, so the model, the inference
engine and the trilingual signal glossary are all inlined into a single page.
That is also what makes the privacy claim true: with nothing to fetch, there is
nothing to leak. A user pastes an SMS containing their bank details and it stays
in their tab.

    python src/train.py
    python src/export_web.py
    python web/build_demo.py      ->  web/demo.html

The signal labels are read from src/predict.py rather than duplicated here, so
the demo and the CLI cannot disagree about what a feature means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from normalize import console_utf8  # noqa: E402
from predict import SIGNAL_LABELS  # noqa: E402

WEB = ROOT / "web"


def build(out: Path) -> None:
    template = (WEB / "template.html").read_text(encoding="utf-8")
    model = (WEB / "model.json").read_text(encoding="utf-8")
    engine = (WEB / "engine.js").read_text(encoding="utf-8")

    # A literal </script> anywhere inside an inlined block would close the tag
    # early and silently break the page.
    for name, blob in (("model.json", model), ("engine.js", engine)):
        if "</script" in blob.lower():
            raise SystemExit(f"{name} contains a </script> sequence; cannot inline")

    labels = json.dumps(
        {k: list(v) for k, v in SIGNAL_LABELS.items()},
        ensure_ascii=False, separators=(",", ":"),
    )

    body = (
        template
        .replace("__MODEL_JSON__", model)
        .replace("__ENGINE_JS__", engine)
        .replace("__SIGNAL_LABELS__", labels)
    )
    for token in ("__MODEL_JSON__", "__ENGINE_JS__", "__SIGNAL_LABELS__"):
        if token in body:
            raise SystemExit(f"placeholder {token} was not substituted")

    # Standalone copy: needs its own <meta charset>. Without one the browser
    # guesses Latin-1, every Arabic string becomes mojibake, and the engine
    # dies at parse time on a now-malformed character range -- which is exactly
    # what happened the first time this was opened locally.
    doc = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
    out.write_text(doc, encoding="utf-8")

    # GitHub Pages copy. Pages can serve from /docs, so writing the same
    # document to docs/index.html publishes the demo at
    # https://<user>.github.io/kurdish-scam-detector/ with no build step and no
    # hosting to pay for.
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(doc, encoding="utf-8")

    # Body-only copy for embedding somewhere that supplies its own <head>.
    (out.parent / "artifact.html").write_text(body, encoding="utf-8")


def main() -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description="Build the self-contained web demo.")
    ap.add_argument("--out", type=Path, default=WEB / "demo.html")
    args = ap.parse_args()

    if not (WEB / "model.json").exists():
        raise SystemExit("run `python src/export_web.py` first")

    build(args.out)
    kb = args.out.stat().st_size / 1024
    print(f"written to : {args.out}")
    print(f"pages copy : {ROOT / 'docs' / 'index.html'}")
    print(f"size       : {kb:.0f} KB ({kb/1024:.2f} MB)")
    print("open it directly in a browser -- no server, no network.")


if __name__ == "__main__":
    main()
