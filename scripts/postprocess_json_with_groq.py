from __future__ import annotations

"""
Post-process existing .json corpus files with Groq to reduce OCR noise.

This does NOT re-run PDF conversion; it rewrites the `text` field of each
paragraph/window in-place using the same Groq refinement used during conversion.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from convert_all_to_json import _refine_with_groq, _select_best_groq_model

try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None  # type: ignore


def iter_json(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.json"):
        if p.is_file():
            yield p


def refine_file(path: Path, client: Groq, model: str, max_chars: int) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {path}: could not read json ({e})")
        return False

    paras = data.get("paragraphs")
    if not isinstance(paras, list):
        return False

    changed = False
    new_paras = []
    for w in paras:
        if not isinstance(w, dict):
            new_paras.append(w)
            continue
        txt = str(w.get("text") or "")
        if not txt.strip():
            new_paras.append(w)
            continue
        refined = _refine_with_groq(txt, client=client, model=model, max_chars=max_chars)
        if refined and refined.strip() and refined.strip() != txt.strip():
            w = dict(w)
            w["text"] = refined
            changed = True
        new_paras.append(w)

    if changed:
        data["paragraphs"] = new_paras
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Groq post-process existing JSON corpus")
    ap.add_argument("--root", default="pdfs", help="Root directory to scan for .json")
    ap.add_argument("--include-sc", action="store_true", help="Include sc_judgments directory")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of files processed")
    ap.add_argument(
        "--groq-model",
        default="meta-llama/llama-3-8b-instruct",
        help="Groq model name (use 'auto' to auto-pick); default is a smaller Llama model to reduce token burn",
    )
    ap.add_argument(
        "--groq-max-chars",
        type=int,
        default=8000,
        help="Max chars per Groq chunk (lower to reduce token usage and rate-limit hits)",
    )
    args = ap.parse_args(argv)

    if Groq is None:
        print("Groq client not available; install `groq` package.")
        return 1

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.")
        return 1

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Could not init Groq client: {e}")
        return 1

    model = _select_best_groq_model(client, args.groq_model)
    print(f"Using Groq model: {model}")

    root = Path(args.root).resolve()
    files = list(iter_json(root))
    if not args.include_sc:
        files = [p for p in files if "sc_judgments" not in p.parts]
    if args.max_files:
        files = files[: args.max_files]

    total = len(files)
    changed = 0
    for i, path in enumerate(files, start=1):
        try:
            if refine_file(path, client, model, args.groq_max_chars):
                changed += 1
        except Exception as e:
            print(f"[error] {path}: {e}")
        if i % 25 == 0 or i == total:
            print(f"[progress] {i}/{total} processed, {changed} changed")

    print(f"Done. {changed}/{total} files updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
