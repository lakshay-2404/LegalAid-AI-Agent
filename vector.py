"""
vector.py — Thin façade that re-exports all public symbols from the
new focused modules (embeddings, chunking, ingestion, retrieval).

Kept for backward compatibility: all existing code that does
    import vector
    vector.hybrid_retrieve(...)
continues to work unchanged.
"""
from __future__ import annotations

# -- Embeddings ---------------------------------------------------------------
from embeddings import (
    DOC_INSTRUCTION,
    QUERY_INSTRUCTION,
    BATCH_SIZE,
    INGEST_WORKERS,
    embed_query,
    embed_documents_parallel as _embed_documents_parallel,
    embed_documents_resilient as _embed_documents_resilient,
    get_embedding_dim_override,
    get_cross_encoder as get_default_cross_encoder,
    cross_encoder_score as default_cross_encoder_fn,
)

# -- Chunking -----------------------------------------------------------------
from chunking import (
    MAX_EMBED_CHARS,
    MIN_TEXT_LEN_FOR_OCR,
    OCR_DPI,
    normalize_pdf_text,
    strip_markdown,
    extract_legal_structure,
    detect_statute_query,
    safe_rel_path as _safe_rel_path,
    infer_act_from_path,
    iter_json_docs_from_path as _iter_json_docs_from_path,
    iter_md_docs as _iter_md_docs,
    iter_pdf_docs as _iter_pdf_docs,
    iter_txt_docs as _iter_txt_docs,
    section_splitter,
    qa_splitter,
    md_splitter,
)

# -- Ingestion ----------------------------------------------------------------
from ingestion import (
    BASE_DIR,
    DB_DIR,
    PDF_DIR,
    COLLECTION_NAME,
    MANIFEST_PATH,
    BM25_DB_PATH,
    SUPPORTED_EXTENSIONS,
    INGEST_SCHEMA_VERSION,
    INGEST_INSERT_BATCH,
    INGEST_DOCS_PER_FLUSH,
    INGEST_MANIFEST_SAVE_INTERVAL,
    BM25_MAX_DOCS,
    compute_file_fingerprint,
    discover_source_files,
    load_manifest,
    save_manifest,
    rebuild_manifest_from_bm25,
    delete_doc_ids,
    iter_docs_for_path,
    ingest_documents,
    ensure_ingested,
    rebuild_index,
    index_status,
    get_vector_store,
    get_embedding_dim,
    get_graph_store,
    _graph_enabled,
    _bm25_db_init,
    _bm25_db_load_docs,
    _bm25_db_upsert_docs,
    _bm25_db_delete_ids,
)

# -- Retrieval ----------------------------------------------------------------
from retrieval import (
    ensure_bm25,
    hybrid_retrieve,
    filter_by_case_name,
    rerank,
    build_citation_prompt,
    _mmr,
    _all_docs,
    _bm25,
    _doc_lookup,
)

# -- vector_store (re-export for code that does `from vector import MilvusVectorStore`) --
from vector_store import MilvusConfig, MilvusVectorStore, infer_embedding_dim


# Convenience wrappers that forward to the new module's base_dir-aware versions.
# Kept so existing call-sites (e.g. rag_core.py) that call vector.iter_md_docs(path)
# still work without modification.
from pathlib import Path

def iter_json_docs_from_path(path: Path):
    yield from _iter_json_docs_from_path(path, BASE_DIR)

def load_json_docs_from_path(path: Path):
    return list(iter_json_docs_from_path(path))

def iter_md_docs(path: Path):
    yield from _iter_md_docs(path, BASE_DIR)

def load_md_docs(path: Path):
    return list(iter_md_docs(path))

def iter_pdf_docs(path: Path):
    yield from _iter_pdf_docs(path, BASE_DIR)

def load_pdf_docs(path: Path):
    return list(iter_pdf_docs(path))

def iter_txt_docs(path: Path):
    yield from _iter_txt_docs(path, BASE_DIR)

def load_txt_docs(path: Path):
    return list(iter_txt_docs(path))

def safe_rel_path(path: Path) -> str:
    return _safe_rel_path(path, BASE_DIR)
