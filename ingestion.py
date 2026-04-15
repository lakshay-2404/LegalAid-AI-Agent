"""
ingestion.py — Incremental ingestion into Milvus + BM25 SQLite for LegalAid RAG.

Manages:
- Source file discovery and manifest tracking
- Embedding + Milvus upsert (streaming, batched)
- BM25 SQLite corpus population
- Graph (Neo4j) upsert when ENABLE_GRAPH=1
- ensure_ingested() — public entry point called on startup
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

import embeddings as emb
from chunking import (
    MAX_EMBED_CHARS,
    enrich_legal_metadata,
    iter_json_docs_from_path,
    iter_md_docs,
    iter_pdf_docs,
    iter_txt_docs,
)
from vector_store import MilvusConfig, MilvusVectorStore, infer_embedding_dim

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_DIR = Path(os.environ.get("DB_DIR", str(BASE_DIR / "chrome_langchain_db")))
DB_DIR.mkdir(exist_ok=True)
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

SC_ZIP_PATH = Path(os.environ.get("SC_JUDGMENTS_ZIP", Path.home() / "Downloads" / "legal-dataset-sc-judgments-india-19502024.zip"))
SC_DATA_DIR = PDF_DIR / "sc_judgments"

JSON_FILES: list[Path] = []   # QA datasets excluded by default for leaner ingestion

COLLECTION_NAME = "indian_legal_rag"
SUPPORTED_EXTENSIONS = {".json", ".md"}
DISCOVER_DIRS = [PDF_DIR]
MANIFEST_PATH = DB_DIR / "ingest_manifest.json"
BM25_DB_PATH = DB_DIR / "bm25.sqlite"
BM25_PICKLE_PATH = DB_DIR / "bm25.pkl"
EMBED_DIM_PATH = DB_DIR / "embed_dim.json"

PREFERRED_SOURCE_PRIORITY = {".json": 3, ".md": 2}
INGEST_SCHEMA_VERSION = 2

# Ingestion tuning (env-overridable, defaults raised for performance)
BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "64"))          # raised from 32
INGEST_INSERT_BATCH = int(os.environ.get("INGEST_INSERT_BATCH", "256"))    # raised from 128
INGEST_FLUSH_STRATEGY = os.environ.get("INGEST_FLUSH_STRATEGY", "batch").strip().lower()
INGEST_DOCS_PER_FLUSH = int(os.environ.get("INGEST_DOCS_PER_FLUSH", "256"))  # raised from 32
INGEST_MANIFEST_SAVE_INTERVAL = int(os.environ.get("INGEST_MANIFEST_SAVE_INTERVAL", "25"))  # raised from 10
BM25_MAX_DOCS = 10_000

_ingest_lock = threading.Lock()
_did_ingest = False
_did_graph_backfill = False

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

_vector_store: Optional[MilvusVectorStore] = None


def _load_cached_embed_dim() -> Optional[int]:
    try:
        if EMBED_DIM_PATH.exists():
            dim = int(json.loads(EMBED_DIM_PATH.read_text(encoding="utf-8")).get("dim", 0))
            return dim if dim > 0 else None
    except Exception:
        return None


def _cache_embed_dim(dim: int) -> None:
    try:
        EMBED_DIM_PATH.write_text(json.dumps({"dim": dim}, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_embedding_dim() -> int:
    override = emb.get_embedding_dim_override()
    if override:
        return override
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
    try:
        dim = infer_embedding_dim(emb.embed_query)
        _cache_embed_dim(dim)
        return dim
    except Exception as e:
        raise RuntimeError(
            "Could not infer embedding dimension. Start Ollama or set EMBEDDING_DIM."
        ) from e


def get_vector_store() -> MilvusVectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    dim = get_embedding_dim()
    cfg = MilvusConfig.from_env(dim=dim)
    _vector_store = MilvusVectorStore(
        config=cfg,
        embed_documents=emb.embed_documents_parallel,
        embed_query=emb.embed_query,
    )
    return _vector_store


# ---------------------------------------------------------------------------
# Graph store (Neo4j, optional)
# ---------------------------------------------------------------------------

_graph_store = None
_graph_store_init_error: Optional[str] = None


def _graph_enabled() -> bool:
    return os.environ.get("ENABLE_GRAPH", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_graph_store():
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


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _file_identity_mode() -> str:
    mode = (os.environ.get("INGEST_FILE_IDENTITY_MODE") or "stat").strip().lower()
    return "sha256" if mode in {"sha256", "hash", "content"} else "stat"


def compute_file_fingerprint(path: Path) -> str:
    if _file_identity_mode() == "sha256":
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return f"sha256:{sha.hexdigest()}"
    stat = path.stat()
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    return f"stat:{stat.st_size}:{mtime_ns}"


def _file_identity_token(fingerprint: str) -> str:
    if not fingerprint:
        return ""
    if fingerprint.startswith("sha256:"):
        return fingerprint.split(":", 1)[1]
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _entry_matches_fingerprint(entry: Optional[Dict], fingerprint: str) -> bool:
    if not entry or not fingerprint:
        return False
    return str(entry.get("fingerprint") or "") == fingerprint


def _normalize_rel_path_str(value) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    raw = re.sub(r"^[.]/+", "", raw) if raw else raw
    return re.sub(r"/{2,}", "/", raw) if raw else raw


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "files" in data:
            return data
    except Exception:
        pass
    return {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}


def save_manifest(manifest: Dict) -> None:
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _backfill_manifest_fingerprints(manifest_files: Dict, current_files: Dict[str, Path]) -> bool:
    changed = False
    for rel, entry in manifest_files.items():
        if not isinstance(entry, dict) or entry.get("fingerprint"):
            continue
        path = current_files.get(rel)
        if path and path.exists():
            entry["fingerprint"] = compute_file_fingerprint(path)
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _prefer_rich_source(existing: Optional[Path], candidate: Path) -> Path:
    if existing is None:
        return candidate
    if PREFERRED_SOURCE_PRIORITY.get(candidate.suffix.lower(), 0) > PREFERRED_SOURCE_PRIORITY.get(existing.suffix.lower(), 0):
        return candidate
    return existing


def _ensure_sc_dataset() -> None:
    try:
        if SC_DATA_DIR.exists() and any(SC_DATA_DIR.iterdir()):
            return
        if not SC_ZIP_PATH.exists():
            return
        SC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(SC_ZIP_PATH, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".pdf", ".json", ".md", ".txt"))]
            for member in members:
                zf.extract(member, SC_DATA_DIR)
        print(f"Extracted Supreme Court dataset to {SC_DATA_DIR}")
    except Exception as e:
        print(f"Warning: could not unpack SC dataset: {e}")


def discover_source_files() -> Dict[str, Path]:
    _ensure_sc_dataset()
    preferred_by_stem: Dict[tuple, Path] = {}
    for file in JSON_FILES:
        if file.exists():
            key = (file.parent, file.stem)
            preferred_by_stem[key] = _prefer_rich_source(preferred_by_stem.get(key), file)
    for directory in DISCOVER_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                key = (path.parent, path.stem)
                preferred_by_stem[key] = _prefer_rich_source(preferred_by_stem.get(key), path)

    from chunking import safe_rel_path
    return {safe_rel_path(p, BASE_DIR): p for p in preferred_by_stem.values()}


# ---------------------------------------------------------------------------
# BM25 SQLite
# ---------------------------------------------------------------------------

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
                "CREATE TABLE IF NOT EXISTS corpus "
                "(doc_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL);"
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
                batch = doc_ids[i: i + 900]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM corpus WHERE doc_id IN ({placeholders});", batch)
    except Exception as e:
        logger.warning("Failed to delete BM25 rows: %s", e)


def _bm25_db_upsert_docs(docs: List[Document], ids: List[str], conn: Optional[sqlite3.Connection] = None) -> None:
    """Upsert docs into BM25 SQLite. Pass a shared `conn` to reuse a transaction."""
    if not docs:
        return
    rows = []
    for doc, doc_id in zip(docs, ids):
        meta = dict(doc.metadata or {})
        meta.setdefault("doc_id", doc_id)
        rows.append((doc_id, doc.page_content or "", json.dumps(meta, ensure_ascii=False)))
    try:
        if conn is not None:
            conn.executemany("INSERT OR REPLACE INTO corpus (doc_id, text, metadata_json) VALUES (?, ?, ?);", rows)
        else:
            with _bm25_db_connect() as c:
                c.executemany("INSERT OR REPLACE INTO corpus (doc_id, text, metadata_json) VALUES (?, ?, ?);", rows)
    except Exception as e:
        logger.warning("Failed to upsert BM25 rows: %s", e)


def _bm25_db_load_docs() -> List[Document]:
    try:
        with _bm25_db_connect() as conn:
            if BM25_MAX_DOCS > 0:
                cur = conn.execute("SELECT doc_id, text, metadata_json FROM corpus LIMIT ?;", (BM25_MAX_DOCS,))
            else:
                cur = conn.execute("SELECT doc_id, text, metadata_json FROM corpus;")
            out: List[Document] = []
            for doc_id, text, meta_json in cur.fetchall():
                try:
                    meta = json.loads(meta_json or "{}")
                except Exception:
                    meta = {}
                meta.setdefault("doc_id", doc_id)
                meta = enrich_legal_metadata(text or "", meta)
                out.append(Document(page_content=text or "", metadata=meta))
            return out
    except Exception as e:
        logger.warning("Failed to load BM25 corpus: %s", e)
        return []


def rebuild_manifest_from_bm25() -> Dict:
    if not BM25_DB_PATH.exists():
        return {}
    try:
        with _bm25_db_connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, COALESCE(json_extract(metadata_json, '$.source_path'), '') FROM corpus"
            ).fetchall()
    except Exception as e:
        logger.warning("Could not rebuild manifest from BM25 corpus: %s", e)
        return {}

    if not rows:
        return {}

    import re
    def _norm(v) -> str:
        raw = str(v or "").strip().replace("\\", "/")
        raw = re.sub(r"^[.]/+", "", raw)
        return re.sub(r"/{2,}", "/", raw)

    doc_ids_by_source: Dict[str, List[str]] = {}
    for doc_id, source_path in rows:
        rel = _norm(source_path) or _norm(str(doc_id or "").rsplit(":", 2)[0] if str(doc_id or "").count(":") >= 2 else "")
        if rel:
            doc_ids_by_source.setdefault(rel, []).append(str(doc_id))

    if not doc_ids_by_source:
        return {}

    current_files = discover_source_files()
    manifest_files: Dict = {}
    for rel, doc_ids in doc_ids_by_source.items():
        path = current_files.get(rel)
        fingerprint = compute_file_fingerprint(path) if path and path.exists() else ""
        manifest_files[rel] = {
            "hash": _file_identity_token(fingerprint),
            "fingerprint": fingerprint,
            "doc_ids": doc_ids,
        }

    manifest = {
        "schema_version": INGEST_SCHEMA_VERSION,
        "files": manifest_files,
        "bootstrap_manifest": False,
        "rebuilt_from_bm25": True,
    }
    save_manifest(manifest)
    logger.info("Rebuilt ingestion manifest from BM25 for %s source files.", len(manifest_files))
    return manifest


# ---------------------------------------------------------------------------
# Delete helpers
# ---------------------------------------------------------------------------

def delete_doc_ids(doc_ids: List[str]) -> None:
    if not doc_ids:
        return
    try:
        get_vector_store().delete_by_ids(doc_ids)
        _graph_delete_doc_ids(doc_ids)
    except Exception as e:
        print(f"Warning: could not delete old docs: {e}")


# ---------------------------------------------------------------------------
# Document iterator dispatcher
# ---------------------------------------------------------------------------

def iter_docs_for_path(path: Path):
    ext = path.suffix.lower()
    if ext == ".json":
        yield from iter_json_docs_from_path(path, BASE_DIR)
    elif ext == ".md":
        yield from iter_md_docs(path, BASE_DIR)
    elif ext == ".txt":
        yield from iter_txt_docs(path, BASE_DIR)
    elif ext == ".pdf":
        yield from iter_pdf_docs(path, BASE_DIR)


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def ingest_documents() -> None:
    global _did_ingest
    if _did_ingest:
        return

    vs = get_vector_store()
    _bm25_db_init()
    embed_dim = get_embedding_dim()
    probe_dim = infer_embedding_dim(emb.embed_query)
    if probe_dim != embed_dim:
        raise RuntimeError(
            "Embedding dimension mismatch before ingestion: "
            f"configured={embed_dim}, provider={probe_dim}. "
            "Align ST_EMBED_MODEL/EMBEDDING_DIM and rebuild the index."
        )
    target_mem_gb = float(os.environ.get("INGEST_MAX_MEM_GB", "6"))
    files_per_flush = int(os.environ.get("INGEST_FILES_PER_FLUSH", "100"))

    # Accumulated-embedding flush buffers (used when INGEST_FLUSH_STRATEGY != "batch")
    acc_ids: List[str] = []
    acc_embeddings: List[List[float]] = []
    acc_docs: List[Document] = []
    acc_file_count = 0

    def flush_accumulated(*, force: bool = False) -> None:
        nonlocal acc_ids, acc_embeddings, acc_docs, acc_file_count
        if not acc_ids:
            return
        est_gb = (len(acc_embeddings) * embed_dim * 4) / 1e9
        if not force and acc_file_count < files_per_flush and est_gb < target_mem_gb:
            return
        for start in range(0, len(acc_ids), INGEST_INSERT_BATCH):
            sl = slice(start, start + INGEST_INSERT_BATCH)
            vs.upsert_embeddings(
                ids=acc_ids[sl],
                embeddings=acc_embeddings[sl],
                documents=[d.page_content for d in acc_docs[sl]],
                metadatas=[d.metadata or {} for d in acc_docs[sl]],
            )
            # Single shared connection for the whole batch → one transaction
            with _bm25_db_connect() as conn:
                _bm25_db_upsert_docs(acc_docs[sl], acc_ids[sl], conn=conn)
            _graph_upsert_docs(acc_docs[sl])
        acc_ids.clear()
        acc_embeddings.clear()
        acc_docs.clear()
        acc_file_count = 0

    with _ingest_lock:
        if _did_ingest:
            return

        manifest = load_manifest()
        manifest_files = manifest.get("files", {})
        changed = False

        # Detect wiped DB with stale manifest — force full re-ingest.
        try:
            if vs.count() == 0 and manifest_files:
                print("Manifest exists but collection is empty; forcing full re-ingest.")
                manifest_files = {}
                manifest = {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
                changed = True
        except Exception:
            pass

        # Schema version bump — selectively refresh MD files.
        if manifest.get("schema_version") != INGEST_SCHEMA_VERSION:
            for rel, entry in list(manifest_files.items()):
                if Path(rel).suffix.lower() in {".md"}:
                    delete_doc_ids(entry.get("doc_ids", []))
                    manifest_files.pop(rel, None)
            manifest["schema_version"] = INGEST_SCHEMA_VERSION

        current_files = discover_source_files()
        if _backfill_manifest_fingerprints(manifest_files, current_files):
            changed = True

        # Remove deleted files.
        for rel in set(manifest_files.keys()) - set(current_files.keys()):
            old_ids = manifest_files.pop(rel, {}).get("doc_ids", [])
            delete_doc_ids(old_ids)
            _bm25_db_delete_ids(old_ids)
            changed = True

        total_files = len(current_files)
        start_index = int(os.environ.get("INGEST_START_INDEX", "0"))
        if start_index > 0:
            print(f"INGEST_START_INDEX={start_index} → skipping first {start_index} files.")
        processed_since_save = 0
        flush_per_file = INGEST_FLUSH_STRATEGY in {"file", "once", "all"}
        flush_threshold = INGEST_DOCS_PER_FLUSH if INGEST_FLUSH_STRATEGY == "batch" else BATCH_SIZE

        for i, (rel, path) in enumerate(current_files.items()):
            if i < start_index:
                continue
            if i % 10 == 0:
                print(f"Checking file {i}/{total_files}: {rel} ...")

            file_fingerprint = compute_file_fingerprint(path)
            existing = manifest_files.get(rel)
            if _entry_matches_fingerprint(existing, file_fingerprint):
                continue

            file_hash = _file_identity_token(file_fingerprint)
            if existing:
                delete_doc_ids(existing.get("doc_ids", []))
                _bm25_db_delete_ids(existing.get("doc_ids", []))
                changed = True

            doc_ids: List[str] = []
            inserted_ids: List[str] = []
            docs_batch: List[Document] = []
            ids_batch: List[str] = []
            texts_for_file: List[str] = []
            metas_for_file: List[dict] = []
            chunk_index = 0
            file_failed = False

            for doc in iter_docs_for_path(path):
                doc_id = f"{rel}:{chunk_index}:{file_hash[:8]}"
                if len(doc.page_content) > MAX_EMBED_CHARS:
                    doc.page_content = doc.page_content[:MAX_EMBED_CHARS]
                doc.metadata["doc_id"] = doc_id
                doc.metadata.setdefault("source_path", rel)
                doc.metadata["char_count"] = len(doc.page_content)

                doc_ids.append(doc_id)
                docs_batch.append(doc)
                ids_batch.append(doc_id)
                texts_for_file.append(doc.page_content)
                metas_for_file.append(doc.metadata or {})
                chunk_index += 1

                # Batch flush (batch strategy)
                if not flush_per_file and len(docs_batch) >= flush_threshold:
                    try:
                        embeddings_slice = emb.embed_documents_resilient([d.page_content for d in docs_batch])
                        vs.upsert_embeddings(
                            ids=ids_batch,
                            embeddings=embeddings_slice,
                            documents=[d.page_content for d in docs_batch],
                            metadatas=[d.metadata or {} for d in docs_batch],
                        )
                        with _bm25_db_connect() as conn:
                            _bm25_db_upsert_docs(docs_batch, ids_batch, conn=conn)
                        _graph_upsert_docs(docs_batch)
                        inserted_ids.extend(ids_batch)
                    except Exception as e:
                        file_failed = True
                        print(f"Warning: failed to ingest batch for {rel}: {e}")
                    docs_batch = []
                    ids_batch = []

            # Final remainder batch (batch strategy)
            if docs_batch and not flush_per_file:
                try:
                    embeddings_slice = emb.embed_documents_resilient([d.page_content for d in docs_batch])
                    vs.upsert_embeddings(
                        ids=ids_batch,
                        embeddings=embeddings_slice,
                        documents=[d.page_content for d in docs_batch],
                        metadatas=[d.metadata or {} for d in docs_batch],
                    )
                    with _bm25_db_connect() as conn:
                        _bm25_db_upsert_docs(docs_batch, ids_batch, conn=conn)
                    _graph_upsert_docs(docs_batch)
                    inserted_ids.extend(ids_batch)
                except Exception as e:
                    file_failed = True
                    print(f"Warning: failed to ingest final batch for {rel}: {e}")

            # Per-file accumulation (file/once/all strategy)
            if flush_per_file and doc_ids:
                try:
                    embeddings_all = emb.embed_documents_resilient(texts_for_file)
                    acc_ids.extend(doc_ids)
                    acc_embeddings.extend(embeddings_all)
                    acc_docs.extend(Document(page_content=texts_for_file[j], metadata=metas_for_file[j]) for j in range(len(doc_ids)))
                    acc_file_count += 1
                    flush_accumulated(force=False)
                except Exception as e:
                    file_failed = True
                    print(f"Warning: failed to ingest per-file batch for {rel}: {e}")

            if not doc_ids:
                manifest_files.pop(rel, None)
                continue

            if file_failed:
                if inserted_ids:
                    delete_doc_ids(inserted_ids)
                    _bm25_db_delete_ids(inserted_ids)
                manifest_files.pop(rel, None)
                changed = True
                continue

            manifest_files[rel] = {"hash": file_hash, "fingerprint": file_fingerprint, "doc_ids": doc_ids}
            changed = True
            processed_since_save += 1
            if processed_since_save >= INGEST_MANIFEST_SAVE_INTERVAL:
                manifest["files"] = manifest_files
                manifest["schema_version"] = INGEST_SCHEMA_VERSION
                save_manifest(manifest)
                processed_since_save = 0

        manifest["schema_version"] = INGEST_SCHEMA_VERSION
        manifest["files"] = manifest_files
        save_manifest(manifest)
        _did_ingest = True

    flush_accumulated(force=True)

    if changed and os.environ.get("AUTO_DOCS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            from documentation_generator import regenerate_docs_if_needed
            regenerate_docs_if_needed(force=False)
        except Exception as e:
            logger.warning("Docs regeneration failed: %s", e)


# ---------------------------------------------------------------------------
# Neo4j backfill (background, on newly-enabled graph)
# ---------------------------------------------------------------------------

def _backfill_graph_worker() -> None:
    try:
        gs = get_graph_store()
        if gs is None or gs.chunk_count() > 0:
            return
        _bm25_db_init()
        corpus_docs = _bm25_db_load_docs()
        if not corpus_docs:
            return
        structured = [d for d in corpus_docs if (d.metadata.get("act") or "").strip() and (d.metadata.get("section") or "").strip()]
        if not structured:
            return
        print(f"Neo4j backfill: {len(structured)} structured chunks ...")
        for i in range(0, len(structured), BATCH_SIZE):
            try:
                gs.upsert_chunks(structured[i: i + BATCH_SIZE])
            except Exception as e:
                logger.warning("Neo4j backfill batch failed: %s", e)
        print(f"Neo4j backfill complete ({gs.chunk_count()} graph nodes).")
    except Exception as e:
        logger.warning("Neo4j backfill skipped: %s", e)


def _maybe_backfill_graph() -> None:
    global _did_graph_backfill
    if _did_graph_backfill or not _graph_enabled():
        return
    _did_graph_backfill = True
    import threading
    threading.Thread(target=_backfill_graph_worker, daemon=True, name="neo4j-backfill").start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_ingested(force: bool = False) -> None:
    """Ensure vector store is populated. Call once at startup."""
    global _did_ingest

    manifest = load_manifest() if MANIFEST_PATH.exists() else {"schema_version": INGEST_SCHEMA_VERSION, "files": {}}
    if not manifest.get("files") or manifest.get("bootstrap_manifest"):
        rebuilt = rebuild_manifest_from_bm25()
        if rebuilt:
            manifest = rebuilt
    manifest_bootstrap = manifest.get("bootstrap_manifest") or not manifest.get("files")

    def _bootstrap_manifest(count_hint=None) -> None:
        if MANIFEST_PATH.exists():
            return
        MANIFEST_PATH.parent.mkdir(exist_ok=True)
        save_manifest({"schema_version": INGEST_SCHEMA_VERSION, "files": {}, "bootstrap_manifest": True, "count_hint": count_hint})

    if _did_ingest and not force and not manifest_bootstrap:
        _maybe_backfill_graph()
        return

    try:
        existing_count = get_vector_store().count()
    except Exception:
        existing_count = 0

    if existing_count > 0 and not force and not manifest_bootstrap:
        _did_ingest = True
        _bootstrap_manifest(existing_count)
        _maybe_backfill_graph()
        return

    if force or manifest_bootstrap or not MANIFEST_PATH.exists():
        ingest_documents()
        _maybe_backfill_graph()
        return

    try:
        if get_vector_store().count() == 0:
            ingest_documents()
            _maybe_backfill_graph()
            return
        _did_ingest = True
        _maybe_backfill_graph()
    except Exception:
        ingest_documents()
        _maybe_backfill_graph()


def rebuild_index() -> None:
    """Full rebuild: drop collection + manifest + BM25, then re-ingest."""
    global _did_ingest, _vector_store
    with _ingest_lock:
        _did_ingest = False
        try:
            from pymilvus import utility  # type: ignore
            coll = os.environ.get("MILVUS_COLLECTION", COLLECTION_NAME)
            if utility.has_collection(coll):
                utility.drop_collection(coll)
        except Exception as e:
            print(f"Warning: failed to drop Milvus collection: {e}")
        for target in (MANIFEST_PATH, BM25_DB_PATH, BM25_PICKLE_PATH):
            try:
                if target.exists():
                    target.unlink()
            except Exception as e:
                print(f"Warning: failed to remove {target.name}: {e}")
        _vector_store = None
        ingest_documents()


def index_status() -> dict:
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


# ---------------------------------------------------------------------------
# Missing import
# ---------------------------------------------------------------------------
import re
