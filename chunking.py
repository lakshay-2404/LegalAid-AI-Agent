"""
chunking.py — Text normalisation, legal structure extraction, and document
chunking/loading for the LegalAid RAG pipeline.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------

MIN_TEXT_LEN_FOR_OCR = 300
OCR_DPI = 400
OCR_DPI_RETRY = 600
OCR_RETRY_MIN_TEXT_LEN = 100
PDF_QUALITY_SAMPLE_PAGES = 6

# Maximum characters we send to the embedding model (prevents context overflow).
MAX_EMBED_CHARS = int(os.environ.get("INGEST_MAX_CHARS", "4000"))   # raised from 6000 default

# Lightweight statute/section reference finder for queries
_QUERY_SECTION_RE = re.compile(r"\b(?:section|sec|s)\.?\s*([0-9]{1,4}[A-Z]?)\b", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(
    r"(?:^|[\n\r]| {2,})([0-9]{1,4}[A-Z]?)(?:\s*\(([a-z0-9]+)\))?\.\s+([^\n]{3,180})",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Text splitters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleRecursiveTextSplitter:
    chunk_size: int
    chunk_overlap: int
    separators: list[str]

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        return [t for t in self._recursive_split(text, self.separators) if t]

    def _recursive_split(self, text: str, seps: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not seps:
            return self._split_by_length(text)
        sep = seps[0]
        if sep and sep in text:
            parts = text.split(sep)
            splits = [p + sep for p in parts[:-1]] + [parts[-1]]
            merged = self._merge_splits(splits)
            if len(seps) > 1:
                out: list[str] = []
                for chunk in merged:
                    if len(chunk) > self.chunk_size:
                        out.extend(self._recursive_split(chunk, seps[1:]))
                    else:
                        out.append(chunk)
                return out
            return merged
        return self._recursive_split(text, seps[1:])

    def _split_by_length(self, text: str) -> list[str]:
        if self.chunk_size <= 0:
            return [text]
        out: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            out.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return out

    def _merge_splits(self, splits: list[str]) -> list[str]:
        if not splits:
            return []
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in splits:
            piece_len = len(piece)
            if current and current_len + piece_len > self.chunk_size:
                chunk = "".join(current)
                chunks.append(chunk)
                if self.chunk_overlap > 0:
                    overlap = chunk[-self.chunk_overlap:]
                    current = [overlap] if overlap else []
                    current_len = len(overlap)
                else:
                    current = []
                    current_len = 0
            current.append(piece)
            current_len += piece_len
        if current:
            chunks.append("".join(current))
        return chunks


section_splitter = SimpleRecursiveTextSplitter(
    chunk_size=1600, chunk_overlap=150,
    separators=["\n\nSection", "\n\n", "\n", ". ", " "],
)

qa_splitter = SimpleRecursiveTextSplitter(
    chunk_size=1200, chunk_overlap=150,
    separators=["\n\nQuestion:", "\n\nAnswer:", "\n\n", "\n", ". "],
)

md_splitter = SimpleRecursiveTextSplitter(
    chunk_size=1400, chunk_overlap=150,
    separators=["\n# ", "\n## ", "\n\nSection", "\n\n", "\n", ". ", " "],
)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00c2", "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"(SECTION|Section|Sec\.?)[\s]*([0-9A-Za-z\(\)\.]+)", r"\n\nSection \2\n", text)
    text = re.sub(r"(CHAPTER|Chapter)[\s]*([IVXLC0-9]+)", r"\n\nChapter \2\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    return text.replace("`", "")

# ---------------------------------------------------------------------------
# Legal structure extraction
# ---------------------------------------------------------------------------

_ACT_PATTERNS: list[tuple[str, str]] = [
    (r"\bBNS\b|Bharatiya\s+Nyaya\s+Sanhita", "Bharatiya Nyaya Sanhita"),
    (r"\bIPC\b|Indian\s+Penal\s+Code", "Indian Penal Code"),
    (r"\bCPC\b|Code\s+of\s+Civil\s+Procedure", "Code of Civil Procedure"),
    (r"\bCrPC\b|Code\s+of\s+Criminal\s+Procedure", "Code of Criminal Procedure"),
    (r"\bBNSS\b|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita", "Bharatiya Nagarik Suraksha Sanhita"),
    (r"\bBSA\b|Bharatiya\s+Sakshya\s+Adhiniyam", "Bharatiya Sakshya Adhiniyam"),
    (r"Indian\s+Contract\s+Act", "Indian Contract Act"),
    (r"Arbitration\s+and\s+Conciliation\s+Act", "Arbitration and Conciliation Act"),
    (r"Consumer\s+Protection\s+Act", "Consumer Protection Act"),
    (r"Specific\s+Relief\s+Act", "Specific Relief Act"),
    (r"Transfer\s+of\s+Property\s+Act", "Transfer of Property Act"),
    (r"Motor\s+Vehicles\s+Act", "Motor Vehicles Act"),
    (r"Indian\s+Succession\s+Act", "Indian Succession Act"),
    (r"Hindu\s+Marriage\s+Act", "Hindu Marriage Act"),
    (r"\bDivorce\s+Act\b|Indian\s+Divorce\s+Act", "Indian Divorce Act"),
    (r"Hindu\s+Minority\s+and\s+Guardianship\s+Act", "Hindu Minority and Guardianship Act"),
    (r"Hindu\s+Adoptions\s+and\s+Maintenance\s+Act", "Hindu Adoptions and Maintenance Act"),
    (r"Delimitation\s+Act", "Delimitation Act"),
    (r"Limitation\s+Act", "Limitation Act"),
    (r"Constitution\s+of\s+India", "Constitution of India"),
    (r"\bRTI\b|Right\s+to\s+Information\s+Act", "Right to Information Act"),
    (r"Protection\s+of\s+Women\s+from\s+Domestic\s+Violence\s+Act", "Protection of Women from Domestic Violence Act"),
]


def _match_act_name(text: str) -> str | None:
    for pattern, act_name in _ACT_PATTERNS:
        if re.search(pattern, text or "", re.IGNORECASE):
            return act_name
    return None


def _clean_section_title(value: str) -> str:
    title = re.sub(r"\s+", " ", (value or "").strip(" .:-"))
    if not title:
        return ""
    title = re.split(r"\s{2,}(?=[A-Z\"(])", title, maxsplit=1)[0]
    return title[:180].strip()


def extract_legal_structure(text: str) -> dict:
    """Extract act, chapter, section, subsection, clause from a text snippet."""
    meta: dict = {}
    act_name = _match_act_name(text)
    if act_name:
        meta["act"] = act_name

    chapter = re.search(r"Chapter\s+([IVXLC0-9]+)", text, re.IGNORECASE)
    if chapter:
        meta["chapter"] = chapter.group(1)

    heading = _SECTION_HEADING_RE.search(text or "")
    if heading:
        meta["section"] = heading.group(1).upper()
        if heading.group(2):
            meta["subsection"] = heading.group(2)
        title = _clean_section_title(heading.group(3) or "")
        if title:
            meta["section_title"] = title
    else:
        section = re.search(r"\bSection\s+([0-9]+(?:[A-Z])?)", text, re.IGNORECASE)
        if section:
            meta["section"] = section.group(1).upper()

        subsection = re.search(r"\bSection\s+(?:[0-9]+(?:[A-Z])?)?\s*\(([a-z0-9]+)\)", text, re.IGNORECASE)
        if subsection:
            meta["subsection"] = subsection.group(1)

    clause = re.search(r"Clause\s+\(([a-z]+)\)", text, re.IGNORECASE)
    if clause:
        meta["clause"] = clause.group(1)

    if meta.get("act") and meta.get("section"):
        citation = f"{meta['section']}, {meta['act']}"
        if meta.get("subsection"):
            citation = f"{meta['section']}({meta['subsection']}), {meta['act']}"
        meta["citation"] = citation

    return meta


def enrich_legal_metadata(text: str, metadata: dict | None = None) -> dict:
    enriched = dict(metadata or {})
    inferred = extract_legal_structure(text or "")
    source_hint = str(enriched.get("source") or enriched.get("source_path") or "")
    source_act = _match_act_name(source_hint.replace("_", " ").replace("-", " "))
    statute_like = str(enriched.get("doc_type") or "").lower() in {"json", "md", "pdf", "txt"}

    if source_act and statute_like:
        enriched["act"] = source_act
        for key in ("section", "subsection", "section_title"):
            if inferred.get(key):
                enriched[key] = inferred[key]

    for key in ("act", "chapter", "section", "subsection", "clause", "section_title"):
        if not enriched.get(key) and inferred.get(key):
            enriched[key] = inferred[key]

    if enriched.get("act") and enriched.get("section"):
        citation = f"{enriched['section']}, {enriched['act']}"
        if enriched.get("subsection"):
            citation = f"{enriched['section']}({enriched['subsection']}), {enriched['act']}"
        enriched["citation"] = citation

    return enriched


def detect_statute_query(text: str) -> tuple[str | None, list[str]]:
    """Return (act, [sections]) detected in a query string."""
    meta = extract_legal_structure(text)
    act = (meta.get("act") or "").strip() or None
    sections: list[str] = []
    seen: set[str] = set()
    for m in _QUERY_SECTION_RE.finditer(text or ""):
        sec = m.group(1).upper()
        if sec not in seen:
            sections.append(sec)
            seen.add(sec)
    if meta.get("section"):
        sec = str(meta["section"]).upper()
        if sec not in seen:
            sections.insert(0, sec)
    return act, sections

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def safe_rel_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.name


def infer_act_from_path(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    canonical = _match_act_name(name)
    if canonical:
        return canonical
    return re.sub(r"\s+", " ", name)

# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _get_ocr_deps():
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path  # type: ignore
    except Exception as e:
        print(f"OCR unavailable (missing pdf2image): {e}")
        return None, None, None
    try:
        import pytesseract  # type: ignore
    except Exception as e:
        print(f"OCR unavailable (missing pytesseract): {e}")
        return None, None, None
    return convert_from_path, pdfinfo_from_path, pytesseract


def _iter_ocr_page_texts(pdf_path: Path, dpi: int = OCR_DPI):
    convert_from_path, pdfinfo_from_path, pytesseract = _get_ocr_deps()
    if not all((convert_from_path, pdfinfo_from_path, pytesseract)):
        return
    try:
        total_pages = int(pdfinfo_from_path(str(pdf_path)).get("Pages") or 0)
    except Exception as e:
        print(f"OCR unavailable (could not inspect {pdf_path.name}): {e}")
        return
    for page_num in range(1, total_pages + 1):
        images: list = []
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num)
            if not images:
                continue
            text = pytesseract.image_to_string(images[0], lang="eng+hin")
            if text.strip():
                yield text
        except Exception as e:
            print(f"OCR error on page {page_num} of {pdf_path.name}: {e}")
        finally:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass


def is_text_extraction_poor(text: str) -> bool:
    raw = text.strip()
    if len(raw) < MIN_TEXT_LEN_FOR_OCR:
        return True
    words = [w for w in raw.split() if w]
    if not words:
        return True
    space_ratio = raw.count(" ") / max(1, len(raw))
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    return space_ratio < 0.02 or avg_word_len > 12

# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def _iter_pdf_page_contents(path: Path):
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        print(f"PDF text extraction unavailable (missing pypdf): {e}")
        return
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                yield text
    except Exception as e:
        print(f"PDF text extraction failed for {path.name}: {e}")
        return


def _sample_pdf_text_quality(path: Path, max_pages: int = PDF_QUALITY_SAMPLE_PAGES) -> str:
    parts: list[str] = []
    for idx, page_text in enumerate(_iter_pdf_page_contents(path)):
        parts.append(normalize_pdf_text(page_text))
        if idx + 1 >= max_pages:
            break
    return "\n".join(p for p in parts if p).strip()


def _iter_pdf_chunks_from_page_texts(page_texts):
    buffer = ""
    for page_text in page_texts:
        normalized = normalize_pdf_text(page_text)
        if not normalized:
            continue
        combined = f"{buffer}\n\n{normalized}".strip() if buffer else normalized
        chunks = section_splitter.split_text(combined)
        if not chunks:
            continue
        for chunk in chunks[:-1]:
            chunk = chunk.strip()
            if len(chunk) >= 50:
                yield chunk
        buffer = chunks[-1].strip()
    if buffer:
        for chunk in section_splitter.split_text(buffer):
            chunk = chunk.strip()
            if len(chunk) >= 50:
                yield chunk


def iter_pdf_docs(path: Path, base_dir: Path, ocr_dpi: int = OCR_DPI):
    """Streaming PDF → Document generator."""
    try:
        sample_text = _sample_pdf_text_quality(path)
        using_ocr = is_text_extraction_poor(sample_text)
        if using_ocr:
            print(f"Using OCR for {path.name} (extracted text quality is low)...")
            chunks_iter = _iter_pdf_chunks_from_page_texts(_iter_ocr_page_texts(path, dpi=ocr_dpi))
        else:
            chunks_iter = _iter_pdf_chunks_from_page_texts(_iter_pdf_page_contents(path))

        act_hint = infer_act_from_path(path)
        source_path = safe_rel_path(path, base_dir)
        pending: list[str] = []
        total_chars = 0
        can_emit = False

        for chunk in chunks_iter:
            chunk = (chunk or "").strip()
            if len(chunk) < 50:
                continue
            total_chars += len(chunk)
            if not can_emit:
                pending.append(chunk)
                if total_chars < MIN_TEXT_LEN_FOR_OCR:
                    continue
                can_emit = True

            current = pending if pending else [chunk]
            pending = []
            for c in current:
                meta = extract_legal_structure(c)
                if not meta.get("act") and act_hint:
                    meta["act"] = act_hint
                yield Document(page_content=c, metadata={"doc_type": "pdf", "source": path.name, "source_path": source_path, **meta})

        if total_chars < MIN_TEXT_LEN_FOR_OCR:
            if using_ocr and total_chars < OCR_RETRY_MIN_TEXT_LEN and ocr_dpi < OCR_DPI_RETRY:
                print(f"Retrying OCR on {path.name} with DPI={OCR_DPI_RETRY}")
                yield from iter_pdf_docs(path, base_dir, ocr_dpi=OCR_DPI_RETRY)
                return
            print(f"Skipping {path.name} — insufficient content")
    except Exception as e:
        print(f"Error loading {path.name}: {e}")

# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def iter_json_docs_from_path(path: Path, base_dir: Path):
    """Streaming JSON → Document generator supporting multiple schemas."""
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"Warning: Could not load {path.name}: {e}")
        return

    source_path = safe_rel_path(path, base_dir)
    act_hint = infer_act_from_path(path)

    # --- Paragraph schema ("{paragraphs: [{text, page, ...}], ...}") -----------
    if isinstance(data, dict) and isinstance(data.get("paragraphs"), list):
        case_id = data.get("case_id") or path.stem
        paras: list[dict] = [p for p in data["paragraphs"] if isinstance(p, dict) and str(p.get("text") or "").strip()]

        # Deduplicate nested windows — keep widest span.
        deduped: list[dict] = []
        for para in paras:
            start, end = para.get("paragraph_start"), para.get("paragraph_end")
            if start is None or end is None:
                deduped.append(para)
                continue
            if not any(
                k.get("paragraph_start") is not None and k["paragraph_start"] <= start and end <= k["paragraph_end"]
                for k in deduped
                if k.get("paragraph_start") is not None
            ):
                deduped.append(para)
        paras = deduped

        MIN_CHARS, MAX_CHARS, OVERLAP = 800, 2000, 2
        i = 0
        while i < len(paras):
            start = i
            length = 0
            while i < len(paras) and length < MIN_CHARS:
                length += len(str(paras[i].get("text", "")))
                i += 1
            end = max(i, start + 1)
            while end - start > 1 and length > MAX_CHARS:
                end -= 1
                length -= len(str(paras[end].get("text", "")))
            window = paras[start:end]
            combined = "\n\n".join(str(p.get("text", "")).strip() for p in window if str(p.get("text", "")).strip())
            if len(combined) < 50:
                i = end
                continue
            pages = sorted({p.get("page") for p in window if p.get("page") is not None})
            para_nums = [p.get("paragraph_number") for p in window if p.get("paragraph_number") is not None]
            meta = extract_legal_structure(combined)
            meta.update({
                "doc_type": "json",
                "doc_subtype": "statute" if meta.get("act") else None,
                "source": path.name, "source_path": source_path, "case_id": case_id,
                "paragraph_start": para_nums[0] if para_nums else None,
                "paragraph_end": para_nums[-1] if para_nums else None,
            })
            if pages:
                meta["pages"] = pages
            if not meta.get("act") and act_hint:
                meta["act"] = act_hint
            if meta.get("act") and not meta.get("doc_subtype"):
                meta["doc_subtype"] = "statute"
            yield Document(page_content=combined, metadata=meta)
            i = max(start + 1, end - OVERLAP)
        return

    # --- List schemas ---------------------------------------------------------
    if not isinstance(data, list):
        return

    for item in data:
        if not isinstance(item, dict):
            continue

        # Q&A schema
        if "question" in item and "answer" in item:
            question, answer = item.get("question"), item.get("answer")
            if not question or not answer:
                continue
            content = f"Question:\n{question}\n\nAnswer:\n{answer}"
            meta = extract_legal_structure(content)
            for chunk in qa_splitter.split_text(content.strip()):
                if len(chunk.strip()) < 50:
                    continue
                yield Document(
                    page_content=chunk,
                    metadata={"doc_type": "qa", "source": path.name, "source_path": source_path,
                               "qa_type": "case_law", "is_question": chunk.strip().lower().startswith("question:"), **meta},
                )
            continue

        # CSV-in-JSON (hma.json) schema
        if len(item) == 1:
            key, value = next(iter(item.items()))
            if "chapter,section,section_title,section_desc" in key:
                if not isinstance(value, str) or not value.strip():
                    continue
                row = next(csv.reader([value]), [])
                while len(row) < 4:
                    row.append("")
                chapter, section, section_title, section_desc = row[:4]
                header = f"Section {section.strip()}" if section.strip() else ""
                if header and section_title.strip():
                    header = f"{header}. {section_title.strip()}"
                elif section_title.strip():
                    header = section_title.strip()
                content = "\n".join(p for p in [header, section_desc.strip()] if p).strip()
                if len(content) < 50:
                    continue
                meta = extract_legal_structure(content)
                if chapter.strip():
                    meta["chapter"] = chapter.strip()
                if section.strip():
                    meta["section"] = section.strip()
                if section_title.strip():
                    meta["section_title"] = section_title.strip()
                if not meta.get("act") and act_hint:
                    meta["act"] = act_hint
                yield Document(page_content=content, metadata={"doc_type": "json", "source": path.name, "source_path": source_path, **meta})
                continue

        # Standard section schema
        if any(k in item for k in ("section", "title", "description", "section_title", "section_desc")):
            section = item.get("section")
            title = item.get("title") or item.get("section_title")
            desc = item.get("description") or item.get("section_desc")
            header = ""
            if section is not None and str(section).strip():
                header = f"Section {str(section).strip()}"
                if title:
                    header = f"{header}. {str(title).strip()}"
            elif title:
                header = str(title).strip()
            content = "\n".join(p for p in [header, str(desc) if desc else ""] if p).strip()
            if len(content) < 50:
                continue
            meta = extract_legal_structure(content)
            if section is not None and str(section).strip():
                meta["section"] = str(section).strip()
            if title:
                meta["section_title"] = str(title).strip()
            if not meta.get("act") and act_hint:
                meta["act"] = act_hint
            yield Document(page_content=content, metadata={"doc_type": "json", "source": path.name, "source_path": source_path, **meta})
            continue

        # Fallback: join string-like values
        content = "\n".join(str(v) for v in item.values() if isinstance(v, (str, int, float))).strip()
        if len(content) < 50:
            continue
        meta = extract_legal_structure(content)
        if not meta.get("act") and act_hint:
            meta["act"] = act_hint
        yield Document(page_content=content, metadata={"doc_type": "json", "source": path.name, "source_path": source_path, **meta})

# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

def iter_md_docs(path: Path, base_dir: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {path.name}: {e}")
        return

    act_hint = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            act_hint = line.replace("#", "").strip()
            break
    if not act_hint:
        act_hint = infer_act_from_path(path)

    cleaned = normalize_pdf_text(strip_markdown(text))
    source_path = safe_rel_path(path, base_dir)

    for chunk in md_splitter.split_text(cleaned):
        chunk = re.sub(r"^\s{0,3}#{1,6}\s*", "", chunk, flags=re.MULTILINE).strip()
        chunk = re.sub(r"(?m)^\s*(\d{1,4}[A-Z]?)\s*$", r"Section \1", chunk)
        chunk = re.sub(r"(?m)^\s*(\d{1,4}[A-Z]?)\.\s*", r"Section \1. ", chunk)
        if len(chunk) < 50:
            continue
        meta = extract_legal_structure(chunk)
        if not meta.get("act") and act_hint:
            meta["act"] = act_hint
        if meta.get("act"):
            meta.setdefault("doc_subtype", "statute")
        yield Document(page_content=chunk, metadata={"doc_type": "md", "source": path.name, "source_path": source_path, **meta})

# ---------------------------------------------------------------------------
# TXT loader
# ---------------------------------------------------------------------------

def iter_txt_docs(path: Path, base_dir: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {path.name}: {e}")
        return
    cleaned = normalize_pdf_text(text)
    source_path = safe_rel_path(path, base_dir)
    for chunk in md_splitter.split_text(cleaned):
        chunk = chunk.strip()
        if len(chunk) < 50:
            continue
        meta = extract_legal_structure(chunk)
        yield Document(page_content=chunk, metadata={"doc_type": "txt", "source": path.name, "source_path": source_path, **meta})
