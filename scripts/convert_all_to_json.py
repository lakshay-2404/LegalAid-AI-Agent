from __future__ import annotations

"""
Convert all PDFs/MDs in the corpus to paragraph JSON + Markdown.

- PDFs: runs generation.convert_file_to_markdown (PyMuPDF -> PyPDF -> OCR).
  The PDF conversion already writes:
    - <name>.md   (cleaned text)
    - <name>.json (paragraph array with page + paragraph_number)
- MDs: converts existing .md into paragraph JSON (paragraph array).

This is meant to be a one-shot corpus normalizer so ingestion can rely on JSON.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Optional Groq dependency; imported lazily.
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover - optional
    Groq = None

from generation import GenerationConfig, convert_file_to_markdown, _md_to_paragraphs

SUPPORTED = {".pdf", ".md"}


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def md_to_paragraph_json(md_path: Path) -> Path:
    md = md_path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = []
    for i, para in enumerate(_md_to_paragraphs(md), start=1):
        p = para.strip()
        if len(p) < 20:
            continue
        paragraphs.append(
            {
                "id": i,
                "paragraph_number": i,
                "page": None,
                "text": p,
                "case_id": md_path.stem,
                "source_md": md_path.name,
            }
        )
    out = {
        "case_id": md_path.stem,
        "source": str(md_path),
        "paragraphs": paragraphs,
    }
    out_path = md_path.with_suffix(".json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars chunks without losing lines."""
    if not text:
        return []
    out: list[str] = []
    buf: list[str] = []
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


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of Groq rate-limit errors without depending on client internals."""
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg or "quota" in msg


def _retry_after_seconds(msg: str, default: float = 1.5) -> float:
    """
    Parse "Please try again in Xms/s/m" hints from Groq errors.
    Caps to 15s to avoid stalling the pipeline for too long.
    """
    m = re.search(r"try again in ([0-9.]+)\s*(ms|s|m)", msg.lower())
    if not m:
        return default
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        val /= 1000.0
    elif unit == "m":
        val *= 60.0
    return min(val, 15.0)


def _rank_groq_models(client: Groq, exclude: set[str] | None = None) -> list[str]:
    """
    Return available Groq models sorted best-first for legal text cleanup,
    excluding any models already tried.
    """
    exclude = exclude or set()
    try:
        models = getattr(client.models.list(), "data", []) or []
    except Exception:
        return []

    def model_id(m):
        return getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else "")  # type: ignore[arg-type]

    def ctx(m):
        return getattr(m, "context_window", None) or (m.get("context_window") if isinstance(m, dict) else 0) or 0  # type: ignore[arg-type]

    scored = []
    for m in models:
        mid = model_id(m)
        if not mid or mid in exclude:
            continue
        score = 0
        if "instruct" in mid:
            score += 100
        if "llama-4" in mid or "llama3" in mid:
            score += 50
        if "vision" in mid or "scout" in mid:
            score += 25
        score += ctx(m) / 1000.0
        scored.append((score, ctx(m), mid))

    scored.sort(reverse=True)
    return [mid for _, __, mid in scored]


def _refine_with_groq(text: str, *, client: Groq, model: str, max_chars: int) -> str:
    """Post-process text with Groq, swapping to another model if rate-limited."""

    chunks = _chunk_text(text, max_chars=max_chars)
    refined_parts: list[str] = []
    system = (
        "You are a precise legal transcription formatter. "
        "Rewrite the provided text into clean Markdown, preserving every sentence and number. "
        "Do NOT summarize or omit content. Keep section numbers, party names, dates, citations. "
        "Insert headings when obvious, keep page breaks as '---' if present, and avoid inventing text."
    )

    active_model = model  # mutable so we can switch after a rate limit
    fallback_queue: list[str] | None = None
    tried_models: set[str] = {active_model}

    for chunk in chunks:
        handled = False
        try:
            resp = client.chat.completions.create(
                model=active_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": chunk},
                ],
            )
            refined_parts.append(resp.choices[0].message.content or "")
            handled = True
            continue
        except Exception as e:  # pragma: no cover - network/client error
            print(f"[Groq] refinement failed on model '{active_model}'; keeping chunk for now: {e}")
            msg = str(e).lower()
            rate_limited = _is_rate_limit_error(e)

            # If it's a per-day quota hit, don't bother retrying; keep the chunk.
            if "per day" in msg or "tpd" in msg:
                refined_parts.append(chunk)
                continue

            # Lightweight retry on the same model for transient per-minute limits.
            if rate_limited:
                wait = _retry_after_seconds(msg)
                time.sleep(wait)
                try:
                    resp = client.chat.completions.create(
                        model=active_model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": chunk},
                        ],
                    )
                    refined_parts.append(resp.choices[0].message.content or "")
                    handled = True
                    continue
                except Exception as e_retry:
                    print(f"[Groq] retry after {wait:.2f}s failed; {e_retry}")
                    rate_limited = _is_rate_limit_error(e_retry)

            # Switch through all remaining models when we keep hitting rate limits.
            if rate_limited:
                if fallback_queue is None:
                    fallback_queue = _rank_groq_models(client, exclude=tried_models)

                while fallback_queue:
                    candidate = fallback_queue.pop(0)
                    tried_models.add(candidate)
                    active_model = candidate
                    try:
                        resp = client.chat.completions.create(
                            model=active_model,
                            temperature=0.0,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": chunk},
                            ],
                        )
                        refined_parts.append(resp.choices[0].message.content or "")
                        handled = True
                        break
                    except Exception as e_fb:  # pragma: no cover - fallback failure
                        print(f"[Groq] fallback '{active_model}' failed; {e_fb}")
                        rate_limited = _is_rate_limit_error(e_fb)
                        continue

        if not handled:
            refined_parts.append(chunk)

    return "\n\n".join(refined_parts).strip()


def window_paragraphs(paragraphs: list[dict], *, min_chars: int = 800, max_chars: int = 2000, overlap: int = 2):
    """Create overlapping windows of paragraphs to preserve dense legal meaning."""
    paras = [p for p in paragraphs if isinstance(p, dict) and str(p.get("text") or "").strip()]
    windows: list[dict] = []
    i = 0
    while i < len(paras):
        start = i
        length = 0
        while i < len(paras) and length < min_chars:
            length += len(str(paras[i].get("text", "")))
            i += 1
        end = max(i, start + 1)
        while end - start > 1 and length > max_chars:
            end -= 1
            length -= len(str(paras[end].get("text", "")))

        window = paras[start:end]
        text = "\n\n".join(str(p.get("text", "")).strip() for p in window if str(p.get("text", "")).strip())
        if not text or len(text) < 50:
            i = end
            continue

        pages = sorted({p.get("page") for p in window if p.get("page") is not None})
        para_nums = [p.get("paragraph_number") for p in window if p.get("paragraph_number") is not None]

        windows.append(
            {
                "text": text,
                "paragraph_start": para_nums[0] if para_nums else None,
                "paragraph_end": para_nums[-1] if para_nums else None,
                "pages": pages if pages else None,
            }
        )

        # overlap
        i = max(start + 1, end - overlap)
    return windows


def _refine_windows_with_groq(windows: list[dict], *, groq_client: Groq | None, groq_model: str, groq_max_chars: int) -> list[dict]:
    if not groq_client or not groq_model:
        return windows
    refined: list[dict] = []
    for w in windows:
        text = str(w.get("text", ""))
        if not text.strip():
            refined.append(w)
            continue
        refined_text = _refine_with_groq(
            text,
            client=groq_client,
            model=groq_model,
            max_chars=max(1000, groq_max_chars),
        )
        w2 = dict(w)
        w2["text"] = refined_text
        refined.append(w2)
    return refined


def _select_best_groq_model(client: Groq, requested: str, exclude: set[str] | None = None) -> str:
    """
    Pick the best available Groq model for legal text cleanup.
    Prefers instruct-style models with large context.
    """
    if requested and requested != "auto":
        return requested
    exclude = exclude or set()
    try:
        models = getattr(client.models.list(), "data", []) or []
        if not models:
            return "meta-llama/llama-4-scout-17b-16e-instruct"

        def model_id(m):
            return getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else "")

        def ctx(m):
            return getattr(m, "context_window", None) or (m.get("context_window") if isinstance(m, dict) else 0) or 0

        scored = []
        for m in models:
            mid = model_id(m)
            if not mid or mid in exclude:
                continue
            score = 0
            if "instruct" in mid:
                score += 100
            if "llama-4" in mid or "llama3" in mid:
                score += 50
            if "vision" in mid or "scout" in mid:
                score += 25
            score += ctx(m) / 1000.0
            scored.append((score, ctx(m), mid))
        if not scored:
            return "meta-llama/llama-4-scout-17b-16e-instruct"
        scored.sort(reverse=True)
        return scored[0][2]
    except Exception:
        return "meta-llama/llama-4-scout-17b-16e-instruct"


def rewrite_with_windows(json_path: Path, *, groq_client: Groq | None, groq_model: str, groq_max_chars: int) -> None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read {json_path.name}: {e}")
        return
    if not isinstance(data, dict) or not isinstance(data.get("paragraphs"), list):
        return
    windows = window_paragraphs(data["paragraphs"])
    windows = _refine_windows_with_groq(
        windows,
        groq_client=groq_client,
        groq_model=groq_model,
        groq_max_chars=groq_max_chars,
    )
    data["paragraphs"] = windows
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert PDFs/MDs to paragraph JSON + Markdown.")
    ap.add_argument("--root", default="pdfs", help="Root directory to scan")
    ap.add_argument("--max-workers", type=int, default=2, help="Workers passed through to PDF OCR")
    ap.add_argument("--include-sc", action="store_true", help="Include sc_judgments directory (huge)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--ocr-mode", choices=["auto", "never", "always", "hybrid"], default="hybrid", help="OCR strategy")
    ap.add_argument(
        "--groq-model",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help='Groq model for noise reduction (use "auto" to pick the best available instruct model)',
    )
    ap.add_argument("--groq-max-chars", type=int, default=12000, help="Max chars per Groq call")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        return 1

    groq_client = None
    groq_model = args.groq_model
    if Groq is None:
        print("Groq client library not available; skipping Groq refinement.")
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not set; skipping Groq refinement.")
        else:
            try:
                groq_client = Groq(api_key=api_key)
                groq_model = _select_best_groq_model(groq_client, groq_model)
                print(f"Using Groq model: {groq_model}")
            except Exception as e:  # pragma: no cover - init error
                print(f"Could not init Groq client: {e}; skipping Groq refinement.")
                groq_client = None

    cfg = GenerationConfig(ocr_mode=args.ocr_mode, ocr_workers=args.max_workers)

    files = list(iter_files(root))
    if not args.include_sc:
        files = [p for p in files if "sc_judgments" not in p.parts]
    total = len(files)
    print(f"Found {total} files to process (include_sc={args.include_sc})")

    ok = 0
    fail = 0
    for idx, path in enumerate(files, start=1):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                res = convert_file_to_markdown(path, output_dir=path.parent, config=cfg, overwrite=args.overwrite)
                if res.ok:
                    rewrite_with_windows(
                        path.with_suffix(".json"),
                        groq_client=groq_client,
                        groq_model=groq_model,
                        groq_max_chars=args.groq_max_chars,
                    )
                    ok += 1
                else:
                    fail += 1
            elif ext == ".md":
                if not args.overwrite and path.with_suffix(".json").exists():
                    ok += 1
                    continue
                md_json = md_to_paragraph_json(path)
                rewrite_with_windows(
                    md_json,
                    groq_client=groq_client,
                    groq_model=groq_model,
                    groq_max_chars=args.groq_max_chars,
                )
                ok += 1
        except Exception as e:
            print(f"[FAIL] {path.name}: {e}")
            fail += 1

        if idx % 10 == 0 or idx == total:
            print(f"[progress] {idx}/{total} processed ({ok} ok, {fail} fail)")

    print(f"Done. {ok} ok, {fail} failed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
