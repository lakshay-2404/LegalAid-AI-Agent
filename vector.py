from pathlib import Path
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from vector_store import MilvusConfig, MilvusVectorStore, infer_embedding_dim

logger = logging.getLogger(__name__)

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
BM25_DB_PATH = DB_DIR / "bm25.sqlite"
EMBED_DIM_PATH = DB_DIR / "embed_dim.json"

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
        try:
            from pdf2image import convert_from_path
        except Exception as e:
            print(f"OCR unavailable (missing pdf2image): {e}")
            return ""

        try:
            import pytesseract
        except Exception as e:
            print(f"OCR unavailable (missing pytesseract): {e}")
            return ""

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

vector_store: Optional[MilvusVectorStore] = None
_vector_store_init_error: Optional[str] = None


def _load_cached_embed_dim() -> Optional[int]:
    try:
        if EMBED_DIM_PATH.exists():
            payload = json.loads(EMBED_DIM_PATH.read_text(encoding="utf-8"))
            dim = int(payload.get("dim"))
            return dim if dim > 0 else None
    except Exception:
        return None
    return None


def _cache_embed_dim(dim: int) -> None:
    try:
        EMBED_DIM_PATH.write_text(json.dumps({"dim": dim}, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_embedding_dim() -> int:
    cached = _load_cached_embed_dim()
    if cached:
        return cached

    env = os.environ.get("EMBEDDING_DIM")
    if env:
        try:
            dim = int(env)
            if dim > 0:
                _cache_embed_dim(dim)
                return dim
        except Exception:
            pass

    # Last resort: probe the embedding server.
    try:
        dim = infer_embedding_dim(embeddings.embed_query)
        _cache_embed_dim(dim)
        return dim
    except Exception as e:
        raise RuntimeError(
            "Could not infer embedding dimension. Start Ollama (for the embedding probe) "
            "or set EMBEDDING_DIM, then retry."
        ) from e


def get_vector_store() -> MilvusVectorStore:
    global vector_store, _vector_store_init_error
    if vector_store is not None:
        return vector_store
    if _vector_store_init_error:
        raise RuntimeError(_vector_store_init_error)

    dim = get_embedding_dim()
    cfg = MilvusConfig.from_env(dim=dim)
    try:
        vector_store = MilvusVectorStore(
            config=cfg,
            embed_documents=embeddings.embed_documents,
            embed_query=embeddings.embed_query,
        )
        return vector_store
    except Exception as e:
        _vector_store_init_error = f"Failed to initialize Milvus vector store: {e}"
        raise


_graph_store = None
_graph_store_init_error: Optional[str] = None


def _graph_enabled() -> bool:
    return os.environ.get("ENABLE_GRAPH", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_graph_store():
    """
    Lazy Neo4j initialization. Returns None if disabled or unavailable.
    """
    global _graph_store, _graph_store_init_error
    if not _graph_enabled():
        return None
    if _graph_store is not None:
        return _graph_store
    if _graph_store_init_error:
        return None
    try:
        from graph_store import GraphStore, Neo4jConfig

        gs = GraphStore(Neo4jConfig.from_env())
        gs.ensure_schema()
        _graph_store = gs
        return _graph_store
    except Exception as e:
        _graph_store_init_error = str(e)
        logger.warning("Neo4j init failed; graph layer disabled: %s", e)
        return None


def _graph_upsert_docs(docs: List[Document]) -> None:
    gs = get_graph_store()
    if not gs:
        return
    try:
        gs.upsert_chunks(docs)
    except Exception as e:
        logger.warning("Neo4j upsert failed: %s", e)


def _graph_delete_doc_ids(doc_ids: List[str]) -> None:
    gs = get_graph_store()
    if not gs:
        return
    try:
        gs.delete_chunks(doc_ids)
    except Exception as e:
        logger.warning("Neo4j delete failed: %s", e)


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
        get_vector_store().delete_by_ids(doc_ids)
        _graph_delete_doc_ids(doc_ids)
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

    vs = get_vector_store()
    _bm25_db_init()

    # Streamlit reruns can create concurrent calls in edge cases; guard with a lock.
    with _ingest_lock:
        if _did_ingest:
            return
    manifest = load_manifest()
    manifest_files = manifest.get("files", {})
    changed = False
    # If the DB directory was recreated/wiped but the manifest is still present,
    # the incremental logic would "skip" everything and leave an empty collection.
    # Detect that and force a full re-ingest.
    try:
        if vs.count() == 0 and manifest_files:
            print("Manifest exists but collection is empty; forcing full re-ingest.")
            manifest_files = {}
            manifest = {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
            changed = True
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
            old_ids = manifest_files.get(rel, {}).get("doc_ids", [])
            delete_doc_ids(old_ids)
            _bm25_db_delete_ids(old_ids)
            manifest_files.pop(rel, None)
            changed = True

    for rel, path in current_files.items():
        file_hash = compute_file_hash(path)
        existing = manifest_files.get(rel)
        if existing and existing.get("hash") == file_hash:
            continue

        if existing:
            old_ids = existing.get("doc_ids", [])
            delete_doc_ids(old_ids)
            _bm25_db_delete_ids(old_ids)
            changed = True

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
                try:
                    vs.add_documents(docs_batch, ids=ids_batch)
                    _bm25_db_upsert_docs(docs_batch, ids_batch)
                    _graph_upsert_docs(docs_batch)
                except Exception as e:
                    print(f"Warning: failed to ingest batch for {rel}: {e}")
                docs_batch = []
                ids_batch = []

        if docs_batch:
            try:
                vs.add_documents(docs_batch, ids=ids_batch)
                _bm25_db_upsert_docs(docs_batch, ids_batch)
                _graph_upsert_docs(docs_batch)
            except Exception as e:
                print(f"Warning: failed to ingest final batch for {rel}: {e}")

        if not doc_ids:
            manifest_files.pop(rel, None)
            continue

        manifest_files[rel] = {"hash": file_hash, "doc_ids": doc_ids}
        changed = True

    manifest["schema_version"] = INGEST_SCHEMA_VERSION
    manifest["files"] = manifest_files
    save_manifest(manifest)
    _did_ingest = True

    if changed and os.environ.get("AUTO_DOCS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            from documentation_generator import regenerate_docs_if_needed

            regenerate_docs_if_needed(force=False)
        except Exception as e:
            logger.warning("Docs regeneration failed: %s", e)


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
        if get_vector_store().count() == 0:
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

        # Best-effort: drop the collection and recreate it on next use.
        try:
            from pymilvus import utility  # type: ignore[import-untyped]

            coll = os.environ.get("MILVUS_COLLECTION", COLLECTION_NAME)
            if utility.has_collection(coll):
                utility.drop_collection(coll)
        except Exception as e:
            print(f"Warning: failed to drop Milvus collection: {e}")

        # Reset manifest so ingest happens as "first run".
        try:
            if MANIFEST_PATH.exists():
                MANIFEST_PATH.unlink()
        except Exception as e:
            print(f"Warning: failed to remove manifest: {e}")

        # Reset BM25 local corpus.
        try:
            if BM25_DB_PATH.exists():
                BM25_DB_PATH.unlink()
        except Exception as e:
            print(f"Warning: failed to reset BM25 db: {e}")

        # Force re-init of vector store on next call.
        vector_store = None

        ingest_documents()


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
        status["count"] = get_vector_store().count()
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


def _bm25_db_connect() -> sqlite3.Connection:
    BM25_DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(BM25_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _bm25_db_init() -> None:
    try:
        with _bm25_db_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corpus_doc_id ON corpus(doc_id);")
    except Exception as e:
        logger.warning("Failed to init BM25 db: %s", e)


def _bm25_db_delete_ids(doc_ids: List[str]) -> None:
    if not doc_ids:
        return
    try:
        with _bm25_db_connect() as conn:
            for i in range(0, len(doc_ids), 900):
                batch = doc_ids[i : i + 900]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM corpus WHERE doc_id IN ({placeholders});", batch)
    except Exception as e:
        logger.warning("Failed to delete BM25 rows: %s", e)


def _bm25_db_upsert_docs(docs: List[Document], ids: List[str]) -> None:
    if not docs:
        return
    try:
        rows: list[tuple[str, str, str]] = []
        for doc, doc_id in zip(docs, ids):
            meta = doc.metadata or {}
            meta.setdefault("doc_id", doc_id)
            rows.append((doc_id, doc.page_content or "", json.dumps(meta, ensure_ascii=False)))
        with _bm25_db_connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO corpus (doc_id, text, metadata_json) VALUES (?, ?, ?);", rows)
    except Exception as e:
        logger.warning("Failed to upsert BM25 rows: %s", e)


def _bm25_db_load_docs() -> List[Document]:
    try:
        with _bm25_db_connect() as conn:
            if BM25_MAX_DOCS and BM25_MAX_DOCS > 0:
                cur = conn.execute("SELECT doc_id, text, metadata_json FROM corpus LIMIT ?;", (BM25_MAX_DOCS,))
            else:
                cur = conn.execute("SELECT doc_id, text, metadata_json FROM corpus;")
            out: List[Document] = []
            for doc_id, text, meta_json in cur.fetchall():
                meta = {}
                try:
                    meta = json.loads(meta_json or "{}")
                except Exception:
                    meta = {}
                meta.setdefault("doc_id", doc_id)
                out.append(Document(page_content=text or "", metadata=meta))
            return out
    except Exception as e:
        logger.warning("Failed to load BM25 corpus: %s", e)
        return []


def ensure_bm25() -> None:
    global _bm25, _all_docs
    if _bm25 is not None:
        return
    with _bm25_lock:
        if _bm25 is not None:
            return

        _bm25_db_init()
        _all_docs = _bm25_db_load_docs()
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
    ensure_ingested()
    vector_hits: list[tuple[Document, float]] = []
    # Milvus calls can transiently fail during load/index transitions.
    # Retry a couple times, then fall back to BM25-only if needed.
    for attempt in range(VECTOR_QUERY_RETRIES + 1):
        try:
            vector_hits = get_vector_store().similarity_search_with_score(query, k=20)
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
