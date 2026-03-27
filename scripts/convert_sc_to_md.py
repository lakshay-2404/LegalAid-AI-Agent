from __future__ import annotations

"""
Batch converter for the Supreme Court judgments dataset.

- Runs the robust PDF -> Markdown converter with OCR fallback.
- Writes outputs to `pdfs/sc_judgments_md/` by default so ingestion can prefer
  the cleaned `.md` instead of re-OCR'ing the original PDFs.
- Optionally emits plain-text `.txt` alongside the Markdown for debugging.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import time

from generation import GenerationConfig, convert_file_to_markdown, write_txt_from_md

DEFAULT_INPUT = Path("pdfs/sc_judgments/supreme_court_judgments")
DEFAULT_OUTPUT = Path("pdfs/sc_judgments_md")


def iter_pdfs(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".pdf":
        yield root
        return
    for path in root.rglob("*.pdf"):
        if path.is_file():
            yield path


def _source_signature(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    h.update(str(int(path.stat().st_mtime)).encode())
    return h.hexdigest()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    buf = []
    count = 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars and buf:
            out.append("\n".join(buf).strip())
            buf, count = [], 0
        buf.append(line)
        count += len(line) + 1
    if buf:
        out.append("\n".join(buf).strip())
    return [c for c in out if c]


def _refine_with_groq(text: str, *, model: str, max_chars: int) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set; skipping Groq refinement.")
        return text
    try:
        from groq import Groq  # type: ignore
    except Exception as e:  # pragma: no cover - dependency issue
        print(f"Groq client unavailable: {e}; skipping refinement.")
        return text

    client = Groq(api_key=api_key)
    chunks = _chunk_text(text, max_chars=max_chars)
    refined_parts: list[str] = []
    system = (
        "You are a precise legal transcription formatter. "
        "Rewrite the provided text into clean Markdown, preserving every sentence and number. "
        "Do NOT summarize or omit content. Keep section numbers, party names, dates, citations. "
        "Insert headings when obvious, keep page breaks as '---' if present, and avoid inventing text."
    )
    for idx, chunk in enumerate(chunks, start=1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": chunk},
                ],
            )
            refined_parts.append(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"Groq refinement failed on chunk {idx}: {e}")
            refined_parts.append(chunk)
    return "\n\n".join(refined_parts).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SC judgment PDFs to Markdown for ingestion.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="PDF file or directory to process")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Directory to write .md outputs")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for conversion")
    parser.add_argument(
        "--ocr-mode",
        choices=["auto", "never", "always", "hybrid"],
        default="hybrid",
        help="OCR strategy (passed through to generation.GenerationConfig)",
    )
    parser.add_argument("--txt", action="store_true", help="Also write plain-text .txt next to each .md")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip files that already have .md outputs")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N files")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages per PDF (speeds up long docs)")
    parser.add_argument("--ocr-workers", type=int, default=1, help="Limit OCR worker threads inside generation")
    parser.add_argument("--groq-model", default="meta-llama/llama-4-scout-17b-16e-instruct", help="Groq model name")
    parser.add_argument("--groq-max-chars", type=int, default=12000, help="Max characters per Groq chunk")
    args = parser.parse_args(argv)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pdfs = list(iter_pdfs(input_root))
    if not pdfs:
        print(f"No PDFs found under {input_root}")
        return 1

    cfg = GenerationConfig(
        ocr_mode=args.ocr_mode,
        also_write_txt=args.txt,
        max_pages=args.max_pages,
        ocr_workers=args.ocr_workers,
    )

    successes = 0
    failures = 0
    seen = 0

    report_path = output_root / "conversion_report.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_file = report_path.open("a", encoding="utf-8")

    def convert_one(pdf_path: Path):
        nonlocal seen
        rel = pdf_path.relative_to(input_root)
        out_md = (output_root / rel).with_suffix(".md")
        out_md.parent.mkdir(parents=True, exist_ok=True)

        sig = _source_signature(pdf_path)
        if args.no_overwrite and out_md.exists():
            # Quick skip if unchanged
            meta_ok = False
            try:
                for line in report_path.read_text(encoding="utf-8").splitlines():
                    entry = json.loads(line)
                    if entry.get("pdf") == str(pdf_path) and entry.get("sig") == sig and entry.get("status") == "ok":
                        meta_ok = True
                        break
            except Exception:
                meta_ok = False
            if meta_ok:
                return None, "skip"

        res = convert_file_to_markdown(
            pdf_path,
            output_dir=out_md.parent,
            config=cfg,
            overwrite=not args.no_overwrite,
        )
        if res.ok:
            try:
                original_md = out_md.read_text(encoding="utf-8")
                refined_md = _refine_with_groq(
                    original_md,
                    model=args.groq_model,
                    max_chars=max(1000, args.groq_max_chars),
                )
                out_md.write_text(refined_md + "\n", encoding="utf-8")
            except Exception as e:  # pragma: no cover - best effort
                print(f"Groq refinement skipped for {out_md.name}: {e}")

        if res.ok:
            res.output_path = out_md  # ensure mirrored path
            if cfg.also_write_txt:
                try:
                    write_txt_from_md(out_md)
                except Exception as e:  # pragma: no cover - best-effort
                    print(f"Warning: could not write TXT for {out_md.name}: {e}")

        status = "ok" if res.ok else "fail"
        report = {
            "pdf": str(pdf_path),
            "md": str(out_md),
            "status": status,
            "sig": sig,
            "used_ocr": res.used_ocr,
            "warnings": res.warnings,
            "errors": res.errors,
            "timestamp": int(time.time()),
        }
        report_file.write(json.dumps(report, ensure_ascii=False) + "\n")
        report_file.flush()

        seen += 1
        if args.progress_every and seen % args.progress_every == 0:
            print(f"[progress] {seen}/{len(pdfs)} processed ({successes} ok, {failures} fail)")
        return res, status

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        future_map = {pool.submit(convert_one, pdf): pdf for pdf in pdfs}
        for future in as_completed(future_map):
            pdf = future_map[future]
            try:
                res, status = future.result()
            except Exception as e:
                failures += 1
                print(f"[FAIL] {pdf.name}: {e}")
                continue

            if status == "skip":
                continue

            if res and res.ok:
                successes += 1
                print(f"[OK]   {pdf.name} -> {res.output_path.name} (OCR: {res.used_ocr})")
            else:
                failures += 1
                err_msg = "; ".join(res.errors) if res else "conversion failed"
                print(f"[FAIL] {pdf.name}: {err_msg}")

    total = successes + failures
    report_file.close()
    print(f"Done. {successes}/{total} succeeded. Output dir: {output_root}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
