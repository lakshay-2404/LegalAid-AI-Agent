from pathlib import Path
import csv
import hashlib
import json
import re
import threading
import time
from typing import Dict, List

import pytesseract
from pdf2image import convert_from_path
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chrome_langchain_db"
DB_DIR.mkdir(exist_ok=True)
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

JSON_FILES = [
    BASE_DIR / "IndicLegalQA Dataset_10K.json",
    BASE_DIR / "IndicLegalQA Dataset_10K_Revised.json",
]

COLLECTION_NAME = "indian_legal_rag"

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".json"}
DISCOVER_DIRS = [PDF_DIR]
MANIFEST_PATH = DB_DIR / "ingest_manifest.json"

INGEST_SCHEMA_VERSION = 2

# Embedding batches that are too large can cause huge JSON responses from Ollama and MemoryError.
# Keep this conservative for reliability (especially on Windows).
BATCH_SIZE = 32
MIN_TEXT_LEN_FOR_OCR = 300
OCR_DPI = 400
OCR_DPI_RETRY = 600
# Set to 0 to include all docs (best for recall/precision of lexical matching).
BM25_MAX_DOCS = 0
VECTOR_QUERY_RETRIES = 2
VECTOR_QUERY_RETRY_SLEEP_S = 1.0

_ingest_lock = threading.Lock()
_bm25_lock = threading.Lock()
_did_ingest = False


def _is_hnsw_load_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "hnsw" in msg
        and ("error loading" in msg or "segment reader" in msg or "error constructing" in msg)
    )


def _repair_chroma_db() -> bool:
    """
    Best-effort repair for corrupted Chroma HNSW indexes.
    Strategy: move the persist directory aside and rebuild from sources.

    Returns True if we moved the directory, False otherwise.
    """
    global DB_DIR, MANIFEST_PATH
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / f"{DB_DIR.name}_corrupt_{ts}"
    try:
        if DB_DIR.exists():
            DB_DIR.rename(backup_dir)
        DB_DIR.mkdir(exist_ok=True)
        print(f"Repaired Chroma DB by moving corrupted directory to: {backup_dir}")
        return True
    except Exception as e:
        # If we can't rename (common on Windows due to locked files), fall back to creating
        # a fresh DB directory next to the existing one so the app can keep working.
        print(f"Warning: failed to move corrupted Chroma DB directory: {e}")
        new_dir = BASE_DIR / f"{DB_DIR.name}_rebuild_{ts}"
        try:
            new_dir.mkdir(exist_ok=True)
            DB_DIR = new_dir
            MANIFEST_PATH = DB_DIR / "ingest_manifest.json"
            print(f"Using a new Chroma DB directory instead: {DB_DIR}")
            return True
        except Exception as e2:
            print(f"Warning: failed to create replacement Chroma DB directory: {e2}")
            print(f"Manual fix: close the app, then rename/delete: {DB_DIR}")
            return False

# =====================================================
# EMBEDDINGS
# =====================================================

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_pdf_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00c2", "")

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

    text = re.sub(
        r"(SECTION|Section|Sec\.?)[\s]*([0-9A-Za-z\(\)\.]+)",
        r"\n\nSection \2\n",
        text,
    )
    text = re.sub(
        r"(CHAPTER|Chapter)[\s]*([IVXLC0-9]+)",
        r"\n\nChapter \2\n",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_legal_structure(text: str) -> dict:
    """Enhanced extraction with hierarchical structure and Indian legal acts."""
    meta = {}

    # Act detection: use word boundaries for acronyms to avoid false matches
    # (e.g., "RTI" would otherwise match the "rti" in "parties").
    act_patterns: list[tuple[str, str]] = [
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

    for pattern, act_name in act_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            meta["act"] = act_name
            break

    # Chapter
    chapter = re.search(r"Chapter\s+([IVXLC0-9]+)", text, re.IGNORECASE)
    if chapter:
        meta["chapter"] = chapter.group(1)

    # Main section
    section = re.search(r"Section\s+([0-9]+(?:[A-Z])?)", text)
    if section:
        meta["section"] = section.group(1)

    # Subsection (e.g., 302(a), 304(1))
    subsection = re.search(r"Section\s+(?:[0-9]+)?\s*\(([a-z0-9]+)\)", text)
    if subsection:
        meta["subsection"] = subsection.group(1)

    # Clause
    clause = re.search(r"Clause\s+\(([a-z]+)\)", text)
    if clause:
        meta["clause"] = clause.group(1)

    # Full citation string
    if meta.get("act") and meta.get("section"):
        citation = f"{meta['section']}, {meta['act']}"
        if meta.get("subsection"):
            citation = f"{meta['section']}({meta['subsection']}), {meta['act']}"
        meta["citation"] = citation

    return meta


# =====================================================
# OCR
# =====================================================

def ocr_pdf(pdf_path: Path, dpi: int = OCR_DPI) -> str:
    """Enhanced OCR with better DPI, multi-language support, and retry logic."""
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
        text = []

        for i, img in enumerate(images):
            try:
                t = pytesseract.image_to_string(img, lang="eng+hin")
                if t.strip():
                    text.append(t)
            except Exception as e:
                print(f"OCR error on page {i + 1} of {pdf_path.name}: {e}")
                continue

        result = normalize_pdf_text("\n".join(text))
        if not result or len(result) < 100:
            if dpi < OCR_DPI_RETRY:
                print(f"Retrying OCR on {pdf_path.name} with DPI={OCR_DPI_RETRY}")
                return ocr_pdf(pdf_path, dpi=OCR_DPI_RETRY)
        return result
    except Exception as e:
        print(f"OCR failed for {pdf_path.name}: {e}")
        return ""


def is_text_extraction_poor(text: str) -> bool:
    raw = text.strip()
    if len(raw) < MIN_TEXT_LEN_FOR_OCR:
        return True

    space_ratio = raw.count(" ") / max(1, len(raw))
    words = [w for w in raw.split() if w]
    if not words:
        return True
    letters = sum(len(w) for w in words)
    avg_word_len = letters / max(1, len(words))

    if space_ratio < 0.02 or avg_word_len > 12:
        return True
    return False


# =====================================================
# CHUNKING - ENHANCED FOR LEGAL STRUCTURE
# =====================================================

# Section-aware splitter: prioritizes legal structure over character count
section_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Smaller for precise section matching
    chunk_overlap=200,  # Higher overlap for context between sections
    separators=[
        "\n\nSection",  # Main section breaks
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence breaks
        " ",
    ],
)

# Fallback splitter for Q&A documents
qa_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Slightly larger for Q&A coherence
    chunk_overlap=100,
    separators=[
        "\n\nQuestion:",
        "\n\nAnswer:",
        "\n\n",
        "\n",
        ". ",
    ],
)

# Markdown splitter for statutes in .md
md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    separators=[
        "\n# ",
        "\n## ",
        "\n\nSection",
        "\n\n",
        "\n",
        ". ",
        " ",
    ],
)

# =====================================================
# HELPERS
# =====================================================


def infer_act_from_path(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = text.replace("`", "")
    return text


def safe_rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return path.name


def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "files" not in data:
            return {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
        return data
    except Exception:
        return {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}


def save_manifest(manifest: Dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# =====================================================
# LOAD JSON (Q&A and Statutes)
# =====================================================

def load_json_docs_from_path(path: Path) -> List[Document]:
    # NOTE: Kept for backward compatibility. The ingestion pipeline uses the
    # streaming generator `iter_json_docs_from_path` to reduce peak memory.
    docs = []
    for d in iter_json_docs_from_path(path):
        docs.append(d)
    return docs


def iter_json_docs_from_path(path: Path):
    docs = []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"Warning: Could not load {path.name}: {e}")
        return

    if not isinstance(data, list):
        return

    source_path = safe_rel_path(path)
    act_hint = infer_act_from_path(path)

    for item in data:
        if not isinstance(item, dict):
            continue

        # Q&A schema
        if "question" in item and "answer" in item:
            question = item.get("question")
            answer = item.get("answer")

            if not question or not answer:
                continue

            content = f"""Question:\n{question}\n\nAnswer:\n{answer}"""
            meta = extract_legal_structure(content)

            for chunk in qa_splitter.split_text(content.strip()):
                if len(chunk.strip()) < 50:
                    continue
                is_question = chunk.strip().lower().startswith("question:")
                yield Document(
                    page_content=chunk,
                    metadata={
                        "doc_type": "qa",
                        "source": path.name,
                        "source_path": source_path,
                        "qa_type": "case_law",
                        "is_question": is_question,
                        **meta,
                    },
                )
                # docs.append(
                #     Document(
                #         page_content=chunk,
                #         metadata={
                #             "doc_type": "qa",
                #             "source": path.name,
                #             "source_path": source_path,
                #             "qa_type": "case_law",
                #             "is_question": is_question,
                #             **meta,
                #         },
                #     )
                # )
            continue

        # CSV-in-JSON schema (hma.json)
        if len(item) == 1:
            key, value = next(iter(item.items()))
            if "chapter,section,section_title,section_desc" in key:
                if not isinstance(value, str) or not value.strip():
                    continue
                row = next(csv.reader([value]), [])
                if not row:
                    continue
                while len(row) < 4:
                    row.append("")
                chapter, section, section_title, section_desc = row[:4]

                content_parts = []
                header = ""
                if section.strip():
                    header = f"Section {section.strip()}"
                    if section_title.strip():
                        header = f"{header}. {section_title.strip()}"
                elif section_title.strip():
                    header = section_title.strip()
                if header:
                    content_parts.append(header)
                if section_desc.strip():
                    content_parts.append(section_desc.strip())
                content = "\n".join(content_parts).strip()

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

                yield Document(
                    page_content=content,
                    metadata={
                        "doc_type": "json",
                        "source": path.name,
                        "source_path": source_path,
                        **meta,
                    },
                )
                continue

        # Standard section schema
        if any(k in item for k in ("section", "title", "description", "section_title", "section_desc")):
            section = item.get("section")
            title = item.get("title") or item.get("section_title")
            desc = item.get("description") or item.get("section_desc")

            content_parts = []
            header = ""
            if section is not None and str(section).strip():
                header = f"Section {str(section).strip()}"
                if title:
                    header = f"{header}. {str(title).strip()}"
            elif title:
                header = str(title).strip()
            if header:
                content_parts.append(header)
            if desc:
                content_parts.append(str(desc))
            content = "\n".join(content_parts).strip()

            if len(content) < 50:
                continue

            meta = extract_legal_structure(content)
            if section is not None and str(section).strip():
                meta["section"] = str(section).strip()
            if title:
                meta["section_title"] = str(title).strip()
            if not meta.get("act") and act_hint:
                meta["act"] = act_hint

            yield Document(
                page_content=content,
                metadata={
                    "doc_type": "json",
                    "source": path.name,
                    "source_path": source_path,
                    **meta,
                },
            )
            continue

        # Fallback: combine string-like values
        values = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
        content = "\n".join(values).strip()
        if len(content) < 50:
            continue
        meta = extract_legal_structure(content)
        if not meta.get("act") and act_hint:
            meta["act"] = act_hint

        yield Document(
            page_content=content,
            metadata={
                "doc_type": "json",
                "source": path.name,
                "source_path": source_path,
                **meta,
            },
        )

    return


def load_json_docs() -> List[Document]:
    """Load Q&A documents from root JSON files."""
    docs = []
    for file in JSON_FILES:
        if not file.exists():
            continue
        docs.extend(load_json_docs_from_path(file))
    return docs


# =====================================================
# LOAD PDF (TEXT + OCR)
# =====================================================

def load_pdf_docs(path: Path) -> List[Document]:
    # NOTE: Kept for backward compatibility. The ingestion pipeline uses the
    # streaming generator `iter_pdf_docs` to reduce peak memory.
    return list(iter_pdf_docs(path))


def iter_pdf_docs(path: Path):
    """Streaming PDF loader to reduce peak memory usage during ingestion."""
    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        raw_text_parts = []
        for page in pages:
            if page.page_content:
                raw_text_parts.append(page.page_content)
        normalized = normalize_pdf_text("\n".join(raw_text_parts))

        if is_text_extraction_poor(normalized):
            print(f"Using OCR for {path.name} (extracted text quality is low)...")
            text = ocr_pdf(path)
        else:
            text = normalized

        if len(text.strip()) < MIN_TEXT_LEN_FOR_OCR:
            print(f"Skipping {path.name} - insufficient content after OCR")
            return

        act_hint = infer_act_from_path(path)
        source_path = safe_rel_path(path)

        for chunk in section_splitter.split_text(text):
            if len(chunk.strip()) < 50:
                continue

            meta = extract_legal_structure(chunk)
            if not meta.get("act") and act_hint:
                meta["act"] = act_hint

            yield Document(
                page_content=chunk,
                metadata={
                    "doc_type": "pdf",
                    "source": path.name,
                    "source_path": source_path,
                    **meta,
                },
            )
    except Exception as e:
        print(f"Error loading {path.name}: {e}")
        return


# =====================================================
# LOAD MD
# =====================================================

def load_md_docs(path: Path) -> List[Document]:
    # NOTE: Kept for backward compatibility. The ingestion pipeline uses the
    # streaming generator `iter_md_docs` to reduce peak memory.
    return list(iter_md_docs(path))


def iter_md_docs(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {path.name}: {e}")
        return

    cleaned = strip_markdown(text)
    cleaned = normalize_pdf_text(cleaned)

    act_hint = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            act_hint = line.replace("#", "").strip()
            break
    if not act_hint:
        act_hint = infer_act_from_path(path)

    source_path = safe_rel_path(path)

    for chunk in md_splitter.split_text(cleaned):
        chunk_clean = re.sub(r"^\s{0,3}#{1,6}\s*", "", chunk, flags=re.MULTILINE).strip()
        # Many statute markdowns use headings like "## 57" and lines like "57. Title".
        # Convert these to "Section 57" so extraction and section-aware splitting work better.
        chunk_clean = re.sub(r"(?m)^\s*(\d{1,4}[A-Z]?)\s*$", r"Section \1", chunk_clean)
        chunk_clean = re.sub(r"(?m)^\s*(\d{1,4}[A-Z]?)\.\s*", r"Section \1. ", chunk_clean)
        if len(chunk_clean) < 50:
            continue
        meta = extract_legal_structure(chunk_clean)
        if not meta.get("act") and act_hint:
            meta["act"] = act_hint
        yield Document(
            page_content=chunk_clean,
            metadata={
                "doc_type": "md",
                "source": path.name,
                "source_path": source_path,
                **meta,
            },
        )
    return


# =====================================================
# SEPARATE Q&A INDEX (for specialized retrieval)
# =====================================================

class QAIndexer:
    """Dedicated Q&A retrieval for better question matching."""

    def __init__(self):
        self.qa_docs = []

    def load(self):
        """Load Q&A documents."""
        self.qa_docs = [d for d in load_json_docs() if d.metadata.get("doc_type") == "qa"]
        print(f"Loaded {len(self.qa_docs)} Q&A documents")

    def retrieve_similar_questions(self, query: str, k: int = 5) -> List[Document]:
        """Find Q&A pairs with similar questions."""
        if not self.qa_docs:
            return []

        from difflib import SequenceMatcher

        question_docs = [d for d in self.qa_docs if d.metadata.get("is_question")]
        if not question_docs:
            return []

        scores = []
        for doc in question_docs:
            similarity = SequenceMatcher(None, query.lower(), doc.page_content.lower()).ratio()
            scores.append((similarity, doc))

        return [doc for sim, doc in sorted(scores, reverse=True)[:k] if sim > 0.3]


# =====================================================
# VECTOR STORE
# =====================================================

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# If the on-disk HNSW index is corrupted, Chroma queries can hard-fail.
# Detect and attempt a one-time repair by moving the DB aside and rebuilding.
try:
    _ = vector_store._collection.count()
except Exception as e:
    if _is_hnsw_load_error(e):
        if _repair_chroma_db():
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(DB_DIR),
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
    else:
        # Non-HNSW errors should surface so we don't hide real problems.
        raise


# =====================================================
# INGEST (INCREMENTAL)
# =====================================================

def discover_source_files() -> Dict[str, Path]:
    files = {}

    for file in JSON_FILES:
        if file.exists():
            files[safe_rel_path(file)] = file

    for directory in DISCOVER_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files[safe_rel_path(path)] = path

    return files


def delete_doc_ids(doc_ids: List[str]) -> None:
    if not doc_ids:
        return
    try:
        vector_store._collection.delete(ids=doc_ids)
    except Exception as e:
        print(f"Warning: could not delete old docs: {e}")


def load_docs_for_path(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_docs(path)
    if ext == ".md":
        return load_md_docs(path)
    if ext == ".json":
        return load_json_docs_from_path(path)
    return []


def iter_docs_for_path(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        yield from iter_pdf_docs(path)
    elif ext == ".md":
        yield from iter_md_docs(path)
    elif ext == ".json":
        yield from iter_json_docs_from_path(path)
    else:
        return


def ingest_documents() -> None:
    global _did_ingest
    if _did_ingest:
        return

    # Streamlit reruns can create concurrent calls in edge cases; guard with a lock.
    with _ingest_lock:
        if _did_ingest:
            return
    manifest = load_manifest()
    manifest_files = manifest.get("files", {})
    # If the DB directory was recreated/wiped but the manifest is still present,
    # the incremental logic would "skip" everything and leave an empty collection.
    # Detect that and force a full re-ingest.
    try:
        if vector_store._collection.count() == 0 and manifest_files:
            print("Manifest exists but collection is empty; forcing full re-ingest.")
            manifest_files = {}
            manifest = {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
    except Exception:
        # If count fails, we will attempt ingestion anyway.
        pass
    if manifest.get("schema_version") != INGEST_SCHEMA_VERSION:
        # If ingestion logic/metadata changed, selectively refresh formats that depend on
        # parsing/metadata changes. Avoid forcing a full rebuild (PDF OCR can be slow).
        refresh_exts = {".md"}
        for rel, entry in list(manifest_files.items()):
            if Path(rel).suffix.lower() in refresh_exts:
                delete_doc_ids(entry.get("doc_ids", []))
                manifest_files.pop(rel, None)
        manifest["schema_version"] = INGEST_SCHEMA_VERSION

    current_files = discover_source_files()
    current_keys = set(current_files.keys())
    manifest_keys = set(manifest_files.keys())

    removed = manifest_keys - current_keys
    if removed:
        for rel in removed:
            delete_doc_ids(manifest_files.get(rel, {}).get("doc_ids", []))
            manifest_files.pop(rel, None)

    for rel, path in current_files.items():
        file_hash = compute_file_hash(path)
        existing = manifest_files.get(rel)
        if existing and existing.get("hash") == file_hash:
            continue

        if existing:
            delete_doc_ids(existing.get("doc_ids", []))

        # Streaming ingestion to reduce peak memory usage.
        # Old approach (kept for reference):
        # docs = load_docs_for_path(path)
        # if not docs:
        #     manifest_files.pop(rel, None)
        #     continue

        doc_ids = []
        docs_batch = []
        ids_batch = []
        chunk_index = 0

        for doc in iter_docs_for_path(path):
            doc_id = f"{rel}:{chunk_index}:{file_hash[:8]}"
            doc.metadata["doc_id"] = doc_id
            doc.metadata.setdefault("source_path", rel)
            doc.metadata["char_count"] = len(doc.page_content)

            doc_ids.append(doc_id)
            docs_batch.append(doc)
            ids_batch.append(doc_id)
            chunk_index += 1

            if len(docs_batch) >= BATCH_SIZE:
                vector_store.add_documents(docs_batch, ids=ids_batch)
                docs_batch = []
                ids_batch = []

        if docs_batch:
            vector_store.add_documents(docs_batch, ids=ids_batch)

        if not doc_ids:
            manifest_files.pop(rel, None)
            continue

        manifest_files[rel] = {"hash": file_hash, "doc_ids": doc_ids}

    manifest["schema_version"] = INGEST_SCHEMA_VERSION
    manifest["files"] = manifest_files
    save_manifest(manifest)
    _did_ingest = True


# Ingestion behavior:
# - Auto-ingest on first run (empty DB or missing manifest)
# - After that, avoid re-ingesting on every import (Streamlit reruns) unless user triggers rebuild
def ensure_ingested(force: bool = False) -> None:
    """
    Ensure the vector store is ready.
    - On first run (no manifest OR empty collection), ingest automatically.
    - If force=True, ingest (incremental) regardless.
    """
    if force:
        ingest_documents()
        return

    if not MANIFEST_PATH.exists():
        ingest_documents()
        return

    try:
        if vector_store._collection.count() == 0:
            ingest_documents()
            return
    except Exception:
        # If count fails (e.g., corrupted HNSW), ingestion can't proceed safely; repair path
        # will have already attempted a rebuild directory above.
        ingest_documents()
        return


def rebuild_index() -> None:
    """
    Full rebuild entrypoint (for Streamlit button).
    Clears the current collection + manifest and re-ingests all sources.
    """
    global _did_ingest, _bm25, _all_docs, vector_store
    with _ingest_lock:
        _did_ingest = False
        _bm25 = None
        _all_docs = []

        # Best-effort: delete collection contents.
        try:
            vector_store._collection.delete(where={})
        except Exception:
            # Fallback: drop and recreate collection via client if supported.
            try:
                vector_store._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        # Reset manifest so ingest happens as "first run".
        try:
            if MANIFEST_PATH.exists():
                MANIFEST_PATH.unlink()
        except Exception as e:
            print(f"Warning: failed to remove manifest: {e}")

        # Recreate vector store handle to ensure a clean collection.
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

        ingest_documents()


print("Initializing vector store...")
ensure_ingested()


def index_status() -> dict:
    """
    Lightweight health/status info for the UI.
    """
    status = {
        "db_dir": str(DB_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_exists": MANIFEST_PATH.exists(),
        "collection_name": COLLECTION_NAME,
    }
    try:
        status["count"] = vector_store._collection.count()
        status["count_error"] = None
    except Exception as e:
        status["count"] = None
        status["count_error"] = str(e)
    return status


# =====================================================
# BM25 INDEX (LAZY)
# =====================================================

_bm25 = None
_all_docs: List[Document] = []


def ensure_bm25() -> None:
    global _bm25, _all_docs
    if _bm25 is not None:
        return
    with _bm25_lock:
        if _bm25 is not None:
            return

        try:
            # Some Chroma versions do not accept "ids" as an include item; ids are returned separately.
            data = vector_store._collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"Warning: Could not load documents for BM25: {e}")
            _bm25 = None
            _all_docs = []
            return

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        ids = data.get("ids") or []

        docs = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if not doc:
                continue
            meta = meta or {}
            # Preserve stable IDs if present (improves dedup across pipelines).
            if doc_id and "doc_id" not in meta:
                meta["doc_id"] = doc_id
            docs.append(Document(page_content=doc, metadata=meta))

        if BM25_MAX_DOCS and len(docs) > BM25_MAX_DOCS:
            docs = docs[:BM25_MAX_DOCS]

        _all_docs = docs
        if not _all_docs:
            _bm25 = None
            return

        _bm25 = BM25Okapi([doc.page_content.split() for doc in _all_docs])


# =====================================================
# HYBRID RETRIEVAL
# =====================================================

def _doc_key(doc: Document) -> str:
    return doc.metadata.get("doc_id") or str(id(doc))


def hybrid_retrieve(query: str, k: int = 10, max_distance: float = 0.5) -> List[Document]:
    """
    Hybrid retrieval combining vector similarity and BM25 keyword matching.
    Returns merged, deduplicated results ranked by combined score.
    60% vector weight + 40% BM25 weight = better semantic + keyword coverage.
    """
    vector_hits = []
    # Chroma can transiently fail if the HNSW index is being compacted or is partially corrupted.
    # Retry a couple times, then fall back to BM25-only if needed.
    for attempt in range(VECTOR_QUERY_RETRIES + 1):
        try:
            vector_hits = vector_store.similarity_search_with_score(query, k=20)
            break
        except Exception as e:
            if attempt >= VECTOR_QUERY_RETRIES:
                print(f"Warning: vector search failed; falling back to BM25-only. Error: {e}")
                vector_hits = []
                break
            time.sleep(VECTOR_QUERY_RETRY_SLEEP_S)
    vector_docs = {}
    vector_scores = {}
    
    for d, dist in vector_hits:
        if dist <= max_distance:
            doc_id = _doc_key(d)
            vector_docs[doc_id] = d
            vector_scores[doc_id] = max(0.0, 1.0 - dist)

    # If the distance threshold was too strict (common for short "what is X" queries),
    # keep a few best vector hits anyway and let downstream relevance filtering decide.
    if not vector_docs and vector_hits:
        for d, dist in vector_hits[:5]:
            doc_id = _doc_key(d)
            vector_docs[doc_id] = d
            vector_scores[doc_id] = max(0.0, 1.0 - dist)

    ensure_bm25()

    bm25_docs = {}
    bm25_scores = {}
    if _bm25 is not None and _all_docs:
        scores = _bm25.get_scores(query.split())
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        max_score = max(scores_list) if scores_list and max(scores_list) > 0 else 1.0

        for d, s in sorted(zip(_all_docs, scores_list), key=lambda x: x[1], reverse=True)[:20]:
            if s > 0:
                doc_id = _doc_key(d)
                bm25_docs[doc_id] = d
                bm25_scores[doc_id] = s / max_score

    merged_ids = set(vector_docs.keys()) | set(bm25_docs.keys())
    merged = {}
    scores_combined = {}

    for doc_id in merged_ids:
        v_score = vector_scores.get(doc_id, 0.0)
        b_score = bm25_scores.get(doc_id, 0.0)
        combined = (v_score * 0.6) + (b_score * 0.4)

        merged[doc_id] = vector_docs.get(doc_id) or bm25_docs.get(doc_id)
        scores_combined[doc_id] = combined

    sorted_docs = sorted(merged.items(), key=lambda x: scores_combined[x[0]], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]


def filter_by_case_name(query: str, docs: List[Document]) -> List[Document]:
    query_lower = query.lower()

    def has_case_anchor(doc: Document) -> bool:
        text = doc.page_content.lower()
        return any(token in text for token in query_lower.split() if len(token) > 3)

    anchored = [d for d in docs if has_case_anchor(d)]
    return anchored if anchored else docs


def rerank(docs: List[Document], query: str | None = None) -> List[Document]:
    """
    Legal authority-aware reranking.
    Higher score = higher priority.

    Scoring:
    - Statutory Acts: +100
    - Sections cited: +50
    - Q&A Case law: +80
    - Primary documents (pdf/md/json): +20
    """

    query_terms = []
    if query:
        query_terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2]

    def score(doc: Document) -> int:
        s = 0

        if doc.metadata.get("doc_type") == "qa":
            s += 80  # Case-law authority

        if doc.metadata.get("act"):
            s += 100  # Statutory authority (highest)

        if doc.metadata.get("section"):
            s += 50  # Section specificity

        if doc.metadata.get("doc_type") in {"pdf", "md", "json"}:
            s += 20  # Primary source

        # Relevance bonus so authority doesn't completely override query match.
        if query_terms:
            text = doc.page_content.lower()
            matched = sum(1 for t in query_terms if t in text)
            rel = matched / max(1, len(query_terms))
            s += int(rel * 100)

        return s

    return sorted(docs, key=score, reverse=True)
