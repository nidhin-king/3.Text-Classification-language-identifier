#!/usr/bin/env python3
"""Command-line interface for language identification.

Usage::

    python cli.py "Bonjour tout le monde"
    python cli.py --interactive
    python cli.py --batch file.csv --text-column text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import FLAG_EMOJI
from src.predict import PredictionError, get_predictor, load_pipeline
from src.utils import get_logger, setup_logging

logger = get_logger("cli")

_FLAGS = FLAG_EMOJI


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print a prediction dict."""
    flag = _FLAGS.get(result["language"], "")
    print(f"\n  {flag} {result['language']}")
    print(f"  confidence : {result['confidence'] * 100:.1f}%")
    print(f"  source     : {result['source']}")
    print(f"  latency    : {result['prediction_time_ms']:.1f} ms")
    if verbose:
        print("  top-5      :")
        for item in result["top_k"]:
            print(f"    - {item['language']:<14} {item['confidence'] * 100:5.1f}%")


def cmd_single(text: str, verbose: bool) -> None:
    """Predict one text and print the result."""
    predictor = get_predictor()
    try:
        result = predictor.predict(text)
    except PredictionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"input  : {text!r}")
    print_result(result, verbose=verbose)


def cmd_interactive(verbose: bool) -> None:
    """Start an interactive REPL for quick predictions."""
    predictor = get_predictor()
    print("Language Identification CLI - type 'exit' or press Ctrl-C to quit.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        try:
            print_result(predictor.predict(text), verbose=verbose)
        except PredictionError as exc:
            print(f"error: {exc}")


def cmd_batch(csv_path: str, text_column: str, out_path: str | None) -> None:
    """Run predictions over a CSV file and save the augmented result."""
    predictor = get_predictor()
    df = predictor.predict_from_csv(csv_path, text_column=text_column)
    out = Path(out_path or csv_path.replace(".csv", "_predicted.csv"))
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Predicted {len(df)} rows -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="language-cli",
        description="Detect the language of a text using a trained ML model.",
    )
    parser.add_argument("text", nargs="*", help="Text to classify.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive mode.")
    parser.add_argument("-b", "--batch", metavar="CSV", help="Batch-predict rows of a CSV file.")
    parser.add_argument("--text-column", default="text", help="CSV column holding the text (default: 'text').")
    parser.add_argument("-o", "--output", help="Output CSV path for batch mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show top-5 predictions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.batch:
        cmd_batch(args.batch, args.text_column, args.output)
        return 0
    if args.interactive:
        cmd_interactive(args.verbose)
        return 0
    if args.text:
        cmd_single(" ".join(args.text), args.verbose)
        return 0

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
