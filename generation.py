from __future__ import annotations

"""
Robust PDF/JSON -> Markdown conversion utilities.

Designed for noisy real-world legal documents:
- Multilingual text (English/Hindi) with Unicode normalization
- Header/footer removal via cross-page repetition detection
- OCR fallback (pdf2image + pytesseract) when extraction is poor
- Markdown structuring (headings, lists, simple tables) + cleanup
- Safe JSON parsing with repair strategies and schema-aware rendering
"""

import argparse
import ast
import json
import logging
import os
import re
import shutil
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Base exception for conversion failures."""


class PdfPasswordError(GenerationError):
    """Raised when a PDF is encrypted and cannot be decrypted."""


@dataclass(frozen=True)
class GenerationConfig:
    """
    Conversion configuration.

    Notes:
    - OCR requires Poppler (for `pdf2image`) and a local Tesseract install.
    - Default OCR language pack: English + Hindi ("eng+hin").
    """

    ocr_mode: Literal["auto", "never", "always", "hybrid"] = "auto"
    ocr_langs: str = "eng+hin"
    ocr_dpi: int = 400
    ocr_dpi_retry: int = 600
    ocr_workers: int = 1

    min_total_text_len_for_ocr: int = 300
    min_page_text_len_for_ocr: int = 40
    min_space_ratio: float = 0.02
    max_avg_word_len: float = 12.0

    max_pages: int | None = None

    header_footer_top_lines: int = 3
    header_footer_bottom_lines: int = 3
    header_footer_min_pages: int = 4
    header_footer_min_fraction: float = 0.6

    emit_page_breaks: bool = False
    page_break_marker: str = "\n\n---\n\n"

    json_max_depth: int = 12
    json_max_list_items: int = 2000
    json_max_str_preview: int = 8000

    write_error_report: bool = True


@dataclass
class GenerationResult:
    input_path: Path
    output_path: Path
    ok: bool = False
    used_ocr: bool = False
    pages_processed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)


_DIRECTIONAL_MARKS_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _read_text_best_effort(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _remove_control_chars(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")  # NBSP
    text = unicodedata.normalize("NFKC", text)
    text = _DIRECTIONAL_MARKS_RE.sub("", text)
    text = _remove_control_chars(text)
    return text


def _fix_common_ocr_spacing(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([\u0900-\u097F])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u0900-\u097F])", r"\1 \2", text)
    return text


def _fix_hyphenation(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)


def _normalize_newlines_and_spaces(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _insert_structure_breaks(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"(?m)^\s*(SECTION|Section|Sec\.?)\s*([0-9A-Za-z\(\)\.]+)\s*",
        r"\n\nSection \2\n",
        text,
    )
    text = re.sub(
        r"(?m)^\s*(CHAPTER|Chapter)\s*([IVXLC0-9]+)\s*",
        r"\n\nChapter \2\n",
        text,
    )
    text = re.sub(r"(?m)^\s*धारा\s*([0-9A-Za-z\(\)\.]+)\s*", r"\n\nधारा \1\n", text)
    text = re.sub(r"(?m)^\s*अध्याय\s*([IVXLC0-9]+)\s*", r"\n\nअध्याय \1\n", text)
    return text


def normalize_extracted_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace("\u00c2", "")
    text = text.replace("�", "")
    text = _fix_hyphenation(text)
    text = _fix_common_ocr_spacing(text)
    text = _insert_structure_breaks(text)
    text = _normalize_newlines_and_spaces(text)
    return text


def _is_text_poor(
    text: str,
    *,
    min_len: int,
    min_space_ratio: float,
    max_avg_word_len: float,
) -> bool:
    raw = (text or "").strip()
    if len(raw) < min_len:
        return True

    space_ratio = raw.count(" ") / max(1, len(raw))
    if space_ratio < min_space_ratio:
        return True

    words = raw.split()
    if not words:
        return True

    words_limited = words[:50_000]
    letters = 0
    for w in words_limited:
        letters += sum(1 for ch in w if ch.isalpha())
    avg_word_len = letters / max(1, len(words_limited))
    if avg_word_len > max_avg_word_len:
        return True

    return False


def _line_signature(line: str) -> str:
    line = normalize_unicode(line).casefold()
    line = re.sub(r"\d+", "#", line)
    line = re.sub(r"\b(page|pg)\b", "page", line)
    line = re.sub(r"[^\w\s#]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < 4:
        return ""
    return line


def _first_last_nonempty_lines(text: str, top_n: int, bottom_n: int) -> tuple[list[str], list[str]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return [], []
    return lines[:top_n], (lines[-bottom_n:] if bottom_n > 0 else [])


def detect_repeating_headers_footers(
    page_texts: Iterable[str],
    *,
    top_n: int,
    bottom_n: int,
    min_pages: int,
    min_fraction: float,
    page_count: int | None = None,
) -> tuple[set[str], set[str]]:
    if page_count is None:
        page_texts_list = list(page_texts)
        page_count = len(page_texts_list)
        iter_texts: Iterable[str] = page_texts_list
    else:
        iter_texts = page_texts

    if page_count < min_pages:
        return set(), set()

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()

    for text in iter_texts:
        top_lines, bottom_lines = _first_last_nonempty_lines(text, top_n, bottom_n)
        for ln in top_lines:
            sig = _line_signature(ln)
            if sig:
                header_counts[sig] += 1
        for ln in bottom_lines:
            sig = _line_signature(ln)
            if sig:
                footer_counts[sig] += 1

    threshold = max(min_pages, int(page_count * min_fraction))
    headers = {sig for sig, c in header_counts.items() if c >= threshold}
    footers = {sig for sig, c in footer_counts.items() if c >= threshold}
    return headers, footers


_PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*(?:page|pg)\s*[\.:]?\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*$",
    re.IGNORECASE,
)


def strip_headers_footers(
    text: str,
    header_sigs: set[str],
    footer_sigs: set[str],
    *,
    top_n: int,
    bottom_n: int,
) -> str:
    if not text.strip():
        return ""

    lines = text.splitlines()

    top_idxs: list[int] = []
    for idx, ln in enumerate(lines):
        if ln.strip():
            top_idxs.append(idx)
        if len(top_idxs) >= top_n:
            break

    bottom_idxs: list[int] = []
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            bottom_idxs.append(idx)
        if len(bottom_idxs) >= bottom_n:
            break
    bottom_idxs_set = set(bottom_idxs)

    to_drop: set[int] = set()
    for idx in top_idxs:
        sig = _line_signature(lines[idx])
        if sig and sig in header_sigs:
            to_drop.add(idx)
        elif _PAGE_NUMBER_LINE_RE.match(lines[idx].strip()):
            to_drop.add(idx)

    for idx in bottom_idxs_set:
        sig = _line_signature(lines[idx])
        if sig and sig in footer_sigs:
            to_drop.add(idx)
        elif _PAGE_NUMBER_LINE_RE.match(lines[idx].strip()):
            to_drop.add(idx)

    kept = [ln for i, ln in enumerate(lines) if i not in to_drop]
    return "\n".join(kept).strip()


_BULLET_RE = re.compile(r"^\s*([•\-\*])\s+(.*)$")
_NUM_LIST_RE = re.compile(r"^\s*(\d{1,4})[.)]\s+(.*)$")
_ALPHA_LIST_RE = re.compile(r"^\s*\(?([a-zA-Z])\)\s+(.*)$")
_ROMAN_LIST_RE = re.compile(r"^\s*\(?([ivxlcdm]+)\)\s+(.*)$", re.IGNORECASE)


def _heading_line_to_md(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None

    m = re.match(r"^(Section|Sec\.?)\s+([0-9A-Za-z\(\)\.]+)\s*(.*)$", s, re.IGNORECASE)
    if m:
        sec = m.group(2).strip().rstrip(".")
        rest = m.group(3).strip(" -:\t")
        return f"## {sec}{(': ' + rest) if rest else ''}".strip()

    m = re.match(r"^(Chapter)\s+([IVXLC0-9]+)\s*(.*)$", s, re.IGNORECASE)
    if m:
        chap = m.group(2).strip()
        rest = m.group(3).strip(" -:\t")
        return f"## Chapter {chap}{(': ' + rest) if rest else ''}".strip()

    m = re.match(r"^(धारा)\s+([0-9A-Za-z\(\)\.]+)\s*(.*)$", s)
    if m:
        sec = m.group(2).strip().rstrip(".")
        rest = m.group(3).strip(" -:\t")
        return f"## धारा {sec}{(': ' + rest) if rest else ''}".strip()
    m = re.match(r"^(अध्याय)\s+([IVXLC0-9]+)\s*(.*)$", s)
    if m:
        chap = m.group(2).strip()
        rest = m.group(3).strip(" -:\t")
        return f"## अध्याय {chap}{(': ' + rest) if rest else ''}".strip()

    m = re.match(r"^(\d{1,4}[A-Z]?)\.\s+(.+)$", s)
    if m and len(s) <= 160:
        return f"## {m.group(1)}. {m.group(2).strip()}"
    m = re.match(r"^(\d{1,4}[A-Z]?)\s+([A-Za-z][A-Za-z0-9 ,:;()\"'\\-/]{3,})$", s)
    if m and len(s) <= 120:
        return f"## {m.group(1)}. {m.group(2).strip()}"

    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        if len(s) <= 70 and upper_ratio >= 0.75 and not s.endswith("."):
            return f"## {s.title()}"

    return None


def _split_table_cells(line: str) -> list[str]:
    if "|" in line:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return [c for c in cells if c]
    return [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]


def _table_block_to_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    n = len(header)
    sep = ["---"] * n
    body = rows[1:] if len(rows) > 1 else []

    def fmt_row(r: list[str]) -> str:
        r2 = (r + [""] * n)[:n]
        return "| " + " | ".join(r2) + " |"

    out = [fmt_row(header), fmt_row(sep)]
    out.extend(fmt_row(r) for r in body)
    return "\n".join(out)


def text_to_markdown(text: str, *, already_normalized: bool = False) -> str:
    if not already_normalized:
        text = normalize_extracted_text(text)
    else:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    lines = [ln.rstrip() for ln in text.splitlines()]
    out_lines: list[str] = []
    paragraph: list[str] = []
    pending_list: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        joined = " ".join(s.strip() for s in paragraph if s.strip())
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            out_lines.append(joined)
        paragraph = []

    def flush_list_item() -> None:
        nonlocal pending_list
        if pending_list is None:
            return
        out_lines.append(pending_list.strip())
        pending_list = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush_list_item()
            flush_paragraph()
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            i += 1
            continue

        heading = _heading_line_to_md(line)
        if heading:
            flush_list_item()
            flush_paragraph()
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(heading)
            out_lines.append("")
            i += 1
            continue

        cells = _split_table_cells(line)
        if len(cells) >= 3:
            j = i + 1
            rows = [cells]
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                nxt_cells = _split_table_cells(nxt)
                if len(nxt_cells) != len(cells):
                    break
                rows.append(nxt_cells)
                j += 1
                if len(rows) >= 50:
                    break
            if len(rows) >= 2:
                flush_list_item()
                flush_paragraph()
                out_lines.append(_table_block_to_md(rows))
                out_lines.append("")
                i = j
                continue

        m = _BULLET_RE.match(line)
        if m:
            flush_paragraph()
            flush_list_item()
            pending_list = f"- {m.group(2).strip()}"
            i += 1
            continue
        m = _NUM_LIST_RE.match(line)
        if m:
            flush_paragraph()
            flush_list_item()
            pending_list = f"{m.group(1)}. {m.group(2).strip()}"
            i += 1
            continue
        m = _ALPHA_LIST_RE.match(line)
        if m:
            flush_paragraph()
            flush_list_item()
            pending_list = f"- ({m.group(1)}) {m.group(2).strip()}"
            i += 1
            continue
        m = _ROMAN_LIST_RE.match(line)
        if m:
            flush_paragraph()
            flush_list_item()
            pending_list = f"- ({m.group(1).lower()}) {m.group(2).strip()}"
            i += 1
            continue

        if pending_list is not None:
            pending_list = re.sub(r"\s+", " ", f"{pending_list} {line}").strip()
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_list_item()
    flush_paragraph()

    md = "\n".join(out_lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def _infer_title_from_text(text: str) -> str | None:
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _PAGE_NUMBER_LINE_RE.match(ln):
            continue
        if len(ln) < 5 or len(ln) > 140:
            continue
        if re.search(r"\b(?:copyright|all rights reserved)\b", ln, re.IGNORECASE):
            continue
        return re.sub(r"\s+", " ", ln)
    return None


def _make_spool_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        name = f"mdgen_spool_{os.getpid()}_{os.urandom(6).hex()}"
        candidate = parent / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise GenerationError("Could not create spool directory")


def _spool_page_path(spool_dir: Path, page_num_1based: int) -> Path:
    return spool_dir / f"page_{page_num_1based:06d}.txt"


def _spool_write_page(spool_dir: Path, page_num_1based: int, text: str) -> None:
    path = _spool_page_path(spool_dir, page_num_1based)
    path.write_text(text, encoding="utf-8", errors="ignore", newline="\n")


def _spool_read_page(spool_dir: Path, page_num_1based: int) -> str:
    path = _spool_page_path(spool_dir, page_num_1based)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def _iter_spooled_pages(spool_dir: Path, page_count: int) -> Iterable[str]:
    for p in range(1, page_count + 1):
        yield _spool_read_page(spool_dir, p)


def _pdf_spool_pages_pypdf(
    pdf_path: Path,
    spool_dir: Path,
    *,
    max_pages: int | None,
    config: GenerationConfig,
) -> tuple[int, str, list[bool]]:
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise GenerationError(f"Missing dependency pypdf: {e}") from e

    try:
        with pdf_path.open("rb") as f:
            reader = PdfReader(f)
            if getattr(reader, "is_encrypted", False):
                try:
                    ok = reader.decrypt("")  # type: ignore[attr-defined]
                    if not ok:
                        raise PdfPasswordError(f"PDF is encrypted and requires a password: {pdf_path.name}")
                except PdfPasswordError:
                    raise
                except Exception as e:
                    raise PdfPasswordError(
                        f"PDF is encrypted and could not be decrypted: {pdf_path.name} ({e})"
                    ) from e

            page_count = len(reader.pages)
            if max_pages is not None:
                page_count = min(page_count, max_pages)

            preview_parts: list[str] = []
            page_poor: list[bool] = []

            for i in range(page_count):
                try:
                    t = reader.pages[i].extract_text() or ""
                except Exception as e:
                    logger.warning("PDF text extraction failed on page %s (%s): %s", i + 1, pdf_path.name, e)
                    t = ""
                norm = normalize_extracted_text(t)
                _spool_write_page(spool_dir, i + 1, norm)
                if len(preview_parts) < 5:
                    preview_parts.append(norm)
                if config.ocr_mode == "hybrid":
                    page_poor.append(
                        _is_text_poor(
                            norm,
                            min_len=config.min_page_text_len_for_ocr,
                            min_space_ratio=config.min_space_ratio,
                            max_avg_word_len=config.max_avg_word_len,
                        )
                    )

            preview_text = "\n".join(preview_parts)
            return page_count, preview_text, page_poor
    except PdfPasswordError:
        raise
    except Exception as e:
        raise GenerationError(f"Failed to read PDF {pdf_path.name}: {e}") from e


def _pdf_page_count_via_pdf2image(pdf_path: Path) -> int:
    try:
        from pdf2image import pdfinfo_from_path
    except Exception as e:  # pragma: no cover
        raise GenerationError(f"Missing dependency pdf2image: {e}") from e

    try:
        info = pdfinfo_from_path(str(pdf_path))
        pages = info.get("Pages") or info.get("pages")
        return int(pages) if pages is not None else 0
    except Exception as e:
        raise GenerationError(f"Could not determine page count for OCR fallback: {e}") from e


def _ocr_pdf_page(pdf_path: Path, page_num_1based: int, *, config: GenerationConfig) -> str:
    try:
        from pdf2image import convert_from_path
    except Exception as e:  # pragma: no cover
        raise GenerationError(f"Missing dependency pdf2image: {e}") from e

    try:
        import pytesseract
    except Exception as e:  # pragma: no cover
        raise GenerationError(f"Missing dependency pytesseract: {e}") from e

    last_err: Exception | None = None
    for dpi in (config.ocr_dpi, config.ocr_dpi_retry):
        if dpi <= 0:
            continue
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=page_num_1based,
                last_page=page_num_1based,
            )
            if not images:
                return ""
            txt = pytesseract.image_to_string(images[0], lang=config.ocr_langs)
            txt = normalize_extracted_text(txt)
            if txt.strip():
                return txt
        except Exception as e:
            last_err = e
            continue

    raise GenerationError(f"OCR failed on page {page_num_1based} of {pdf_path.name}: {last_err}")


def convert_pdf_to_markdown(pdf_path: Path, output_path: Path, config: GenerationConfig) -> GenerationResult:
    result = GenerationResult(input_path=pdf_path, output_path=output_path)

    if not pdf_path.exists():
        result.add_error(f"Input PDF does not exist: {pdf_path}")
        return result
    if pdf_path.stat().st_size == 0:
        result.add_error(f"Input PDF is empty: {pdf_path}")
        if config.write_error_report:
            _atomic_write_text(output_path, f"# {pdf_path.stem}\n\n(Empty PDF file.)\n")
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)

    spool_dir = _make_spool_dir(output_path.parent)

    try:
        extraction_failed: Exception | None = None
        try:
            page_count, preview_text, hybrid_page_poor = _pdf_spool_pages_pypdf(
                pdf_path,
                spool_dir,
                max_pages=config.max_pages,
                config=config,
            )
        except PdfPasswordError:
            raise
        except GenerationError as e:
            extraction_failed = e
            if config.ocr_mode == "never":
                raise
            page_count = _pdf_page_count_via_pdf2image(pdf_path)
            if page_count <= 0:
                raise GenerationError(f"{e}. OCR fallback could not determine page count.") from e
            if config.max_pages is not None:
                page_count = min(page_count, config.max_pages)
            preview_text = ""
            hybrid_page_poor = []

        result.pages_processed = page_count

        doc_poor = _is_text_poor(
            preview_text,
            min_len=config.min_total_text_len_for_ocr,
            min_space_ratio=config.min_space_ratio,
            max_avg_word_len=config.max_avg_word_len,
        )

        pages_to_ocr: set[int] = set()
        if extraction_failed is not None:
            pages_to_ocr = set(range(1, page_count + 1))
        elif config.ocr_mode == "always":
            pages_to_ocr = set(range(1, page_count + 1))
        elif config.ocr_mode == "auto" and doc_poor:
            pages_to_ocr = set(range(1, page_count + 1))
        elif config.ocr_mode == "hybrid":
            pages_to_ocr = {i for i, poor in enumerate(hybrid_page_poor, start=1) if poor}
        elif config.ocr_mode == "never":
            pages_to_ocr = set()

        if pages_to_ocr:
            result.used_ocr = True

            def do_ocr(page_num: int) -> tuple[int, str]:
                return page_num, _ocr_pdf_page(pdf_path, page_num, config=config)

            if config.ocr_workers > 1 and len(pages_to_ocr) > 1:
                with ThreadPoolExecutor(max_workers=config.ocr_workers) as ex:
                    futs = [ex.submit(do_ocr, p) for p in sorted(pages_to_ocr)]
                    for fut in as_completed(futs):
                        try:
                            p, txt = fut.result()
                            if txt.strip():
                                _spool_write_page(spool_dir, p, txt)
                        except Exception as e:
                            result.add_warning(str(e))
            else:
                for p in sorted(pages_to_ocr):
                    try:
                        txt = _ocr_pdf_page(pdf_path, p, config=config)
                        if txt.strip():
                            _spool_write_page(spool_dir, p, txt)
                    except Exception as e:
                        result.add_warning(str(e))

        header_sigs, footer_sigs = detect_repeating_headers_footers(
            _iter_spooled_pages(spool_dir, page_count),
            top_n=config.header_footer_top_lines,
            bottom_n=config.header_footer_bottom_lines,
            min_pages=config.header_footer_min_pages,
            min_fraction=config.header_footer_min_fraction,
            page_count=page_count,
        )

        first_page_text = _spool_read_page(spool_dir, 1) if page_count else ""
        first_page_preview = strip_headers_footers(
            first_page_text,
            header_sigs,
            footer_sigs,
            top_n=config.header_footer_top_lines,
            bottom_n=config.header_footer_bottom_lines,
        )
        title = _infer_title_from_text(first_page_preview) or pdf_path.stem.replace("_", " ").strip()
        title = re.sub(r"\s+", " ", title).strip() or "Document"

        tmp_out = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        wrote_any = False
        try:
            with tmp_out.open("w", encoding="utf-8", newline="\n") as out:
                out.write(f"# {title}\n\n")
                for idx in range(1, page_count + 1):
                    page_text = _spool_read_page(spool_dir, idx)
                    cleaned = strip_headers_footers(
                        page_text,
                        header_sigs,
                        footer_sigs,
                        top_n=config.header_footer_top_lines,
                        bottom_n=config.header_footer_bottom_lines,
                    )
                    md = text_to_markdown(cleaned, already_normalized=True)
                    if md.strip():
                        wrote_any = True
                        out.write(md.strip())
                        out.write("\n\n")
                    if config.emit_page_breaks and idx != page_count and md.strip():
                        out.write(config.page_break_marker.strip())
                        out.write("\n\n")
                if not wrote_any:
                    out.write("(No extractable text.)\n")
            os.replace(tmp_out, output_path)
        finally:
            try:
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass

        if wrote_any:
            result.ok = True
        else:
            result.add_error("No extractable text (text extraction and OCR produced no content).")
        return result
    except PdfPasswordError as e:
        result.add_error(str(e))
    except Exception as e:
        result.add_error(f"PDF conversion failed for {pdf_path.name}: {e}")
        logger.exception("PDF conversion failed for %s", pdf_path)
    finally:
        try:
            shutil.rmtree(spool_dir, ignore_errors=True)
        except Exception:
            pass

    if config.write_error_report:
        msg = "\n".join(f"- {m}" for m in (result.errors or ["Unknown error"]))
        _atomic_write_text(output_path, f"# {pdf_path.stem}\n\n## Conversion failed\n\n{msg}\n")
    return result


def _balanced_json_substring(text: str) -> str | None:
    s = text.lstrip()
    if not s:
        return None
    start_candidates = [i for i, ch in enumerate(s) if ch in "[{"]
    if not start_candidates:
        return None
    start = start_candidates[0]
    s2 = s[start:]

    stack: list[str] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s2):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch in "[{":
            stack.append(ch)
            continue
        if ch in "]}":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch, ch) not in (("[", "]"), ("{", "}")):
                return None
            if not stack:
                return s2[: i + 1]
    return None


def _json_repair_pass(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def safe_parse_json(text: str) -> tuple[Any | None, list[str]]:
    errs: list[str] = []
    raw = text.strip()
    if not raw:
        return None, ["Empty JSON input"]

    attempts: list[str] = [raw, _json_repair_pass(raw)]
    balanced = _balanced_json_substring(raw)
    if balanced:
        attempts.append(balanced)
        attempts.append(_json_repair_pass(balanced))

    for cand in attempts:
        try:
            return json.loads(cand), errs
        except json.JSONDecodeError as e:
            errs.append(f"json: {e}")
        except Exception as e:
            errs.append(f"json: {e}")

    # JSONL (line-delimited) fallback
    objs = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            objs.append(json.loads(ln))
        except Exception:
            objs = []
            break
    if objs:
        return objs, errs + ["Parsed as JSONL"]

    # Python-literal fallback (last resort)
    py = raw
    py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
    py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
    py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(py), errs + ["Parsed using ast.literal_eval fallback"]
    except Exception as e:
        errs.append(f"literal_eval: {e}")

    return None, errs


def _is_section_list(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    for item in obj[:50]:
        if not isinstance(item, dict):
            return False
        if "section" not in item and "Section" not in item:
            return False
    return True


def _md_escape_heading(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t.replace("\n", " ")


def _render_scalar(value: Any, *, config: GenerationConfig) -> list[str]:
    if value is None:
        return ["(null)"]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)):
        return [str(value)]
    s = normalize_extracted_text(str(value))
    if len(s) > config.json_max_str_preview:
        s = s[: config.json_max_str_preview].rstrip() + "…"
    return [s] if s else [""]


def json_to_markdown(obj: Any, *, title: str, config: GenerationConfig) -> str:
    lines: list[str] = [f"# {title}", ""]

    if _is_section_list(obj):
        for item in obj:
            section = item.get("section") if isinstance(item, dict) else None
            title2 = None
            desc = None
            if isinstance(item, dict):
                title2 = item.get("title") or item.get("section_title")
                desc = item.get("description") or item.get("section_desc")
                if section is None:
                    section = item.get("Section")

            if section is None:
                continue
            lines.append(f"## {section}")
            if title2:
                lines.append(str(title2).strip())
            if desc:
                lines.extend(_render_scalar(desc, config=config))
            lines.append("")
            lines.append("---")
            lines.append("")

        md = "\n".join(lines)
        md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
        return md

    def render(value: Any, depth: int, heading_level: int) -> None:
        if depth > config.json_max_depth:
            lines.append("> (Max depth reached; output truncated.)")
            lines.append("")
            return

        if isinstance(value, dict):
            for k, v in value.items():
                hk = _md_escape_heading(k)
                level = min(6, heading_level)
                lines.append(f"{'#' * level} {hk}")
                lines.append("")
                render(v, depth + 1, heading_level + 1)
            return

        if isinstance(value, list):
            if not value:
                lines.append("(empty list)")
                lines.append("")
                return
            if len(value) > config.json_max_list_items:
                value = value[: config.json_max_list_items]
                lines.append(f"> (List truncated to {config.json_max_list_items} items.)")
                lines.append("")

            if all(not isinstance(v, (dict, list)) for v in value[:50]):
                for v in value:
                    scalar = _render_scalar(v, config=config)
                    bullet = scalar[0] if scalar else ""
                    lines.append(f"- {bullet}".rstrip())
                lines.append("")
                return

            for idx, v in enumerate(value, start=1):
                lines.append(f"{idx}.")
                if isinstance(v, (dict, list)):
                    render(v, depth + 1, min(heading_level + 1, 6))
                else:
                    scalar = _render_scalar(v, config=config)
                    if scalar and scalar[0]:
                        lines.append(f"   {scalar[0]}")
                        lines.append("")
            lines.append("")
            return

        lines.extend(_render_scalar(value, config=config))
        lines.append("")

    render(obj, depth=0, heading_level=2)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def convert_json_to_markdown(json_path: Path, output_path: Path, config: GenerationConfig) -> GenerationResult:
    result = GenerationResult(input_path=json_path, output_path=output_path)
    if not json_path.exists():
        result.add_error(f"Input JSON does not exist: {json_path}")
        return result
    if json_path.stat().st_size == 0:
        result.add_error(f"Input JSON is empty: {json_path}")
        if config.write_error_report:
            _atomic_write_text(output_path, f"# {json_path.stem}\n\n(Empty JSON file.)\n")
        return result

    try:
        text = _read_text_best_effort(json_path)
        obj, errs = safe_parse_json(text)
        for e in errs[:10]:
            result.add_warning(f"{json_path.name}: {e}")

        title = json_path.stem.replace("_", " ").strip() or "Document"
        if obj is None:
            result.add_error(f"Could not parse JSON: {json_path.name}")
            md = f"# {title}\n\n## JSON parse failed\n\n```text\n{text[:50000]}\n```\n"
            _atomic_write_text(output_path, md)
            return result

        md = json_to_markdown(obj, title=title, config=config)
        _atomic_write_text(output_path, md)
        result.ok = True
        return result
    except Exception as e:
        result.add_error(f"JSON conversion failed for {json_path.name}: {e}")
        logger.exception("JSON conversion failed for %s", json_path)
        if config.write_error_report:
            _atomic_write_text(output_path, f"# {json_path.stem}\n\n## Conversion failed\n\n- {e}\n")
        return result


def convert_file_to_markdown(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: GenerationConfig | None = None,
    overwrite: bool = True,
) -> GenerationResult:
    cfg = config or GenerationConfig()
    src = Path(input_path)

    out_dir = Path(output_dir) if output_dir is not None else src.parent
    out_path = out_dir / f"{src.stem}.md"
    if out_path.exists() and not overwrite:
        res = GenerationResult(input_path=src, output_path=out_path, ok=True)
        res.add_warning(f"Skipping existing output (overwrite disabled): {out_path}")
        return res

    ext = src.suffix.lower()
    if ext == ".pdf":
        return convert_pdf_to_markdown(src, out_path, cfg)
    if ext == ".json":
        return convert_json_to_markdown(src, out_path, cfg)

    res = GenerationResult(input_path=src, output_path=out_path)
    res.add_error(f"Unsupported input type: {src.suffix}")
    if cfg.write_error_report:
        _atomic_write_text(out_path, f"# {src.stem}\n\n(Unsupported input type: {src.suffix})\n")
    return res


def _discover_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for child in it:
                if child.is_file() and child.suffix.lower() in {".pdf", ".json"}:
                    out.append(child)
        elif p.is_file():
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert PDF/JSON files into clean Markdown.")
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("--output-dir", default=None, help="Directory to write .md outputs")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing .md outputs")
    parser.add_argument("--ocr-mode", choices=["auto", "never", "always", "hybrid"], default="auto")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    cfg = GenerationConfig(ocr_mode=args.ocr_mode)
    inputs = [Path(s) for s in args.inputs]
    files = _discover_inputs(inputs, recursive=args.recursive)
    if not files:
        logger.error("No input files found.")
        return 2

    ok = 0
    for f in files:
        try:
            res = convert_file_to_markdown(
                f,
                output_dir=args.output_dir,
                config=cfg,
                overwrite=not args.no_overwrite,
            )
            if res.ok:
                ok += 1
                logger.info("Wrote: %s", res.output_path)
            else:
                logger.error("Failed: %s", f)
        except Exception as e:
            logger.exception("Unhandled error for %s: %s", f, e)

    logger.info("Done. %s/%s succeeded.", ok, len(files))
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
