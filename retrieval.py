"""
retrieval.py — Hybrid BM25 + vector retrieval, MMR, reranking, and citation
prompt assembly for the LegalAid RAG pipeline.
"""
from __future__ import annotations

import heapq
import logging
import re
import time
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document

import embeddings as emb
from chunking import detect_statute_query
from ingestion import (
    BM25_DB_PATH,
    BM25_MAX_DOCS,
    BM25_PICKLE_PATH,
    _bm25_db_connect,
    _bm25_db_init,
    ensure_ingested,
    get_graph_store,
    _graph_enabled,
)

logger = logging.getLogger(__name__)

VECTOR_QUERY_RETRIES = 2
VECTOR_QUERY_RETRY_SLEEP_S = 1.0

# ---------------------------------------------------------------------------
# BM25 in-memory index (loaded lazily)
# ---------------------------------------------------------------------------

import threading

_bm25 = None
_all_docs: List[Document] = []
_doc_lookup: dict[str, Document] = {}
_bm25_lock = threading.Lock()


def ensure_bm25() -> None:
    global _bm25, _all_docs, _doc_lookup
    if _bm25 is not None:
        return
    with _bm25_lock:
        if _bm25 is not None:
            return

        _bm25_db_init()
        from ingestion import _bm25_db_load_docs
        _all_docs = _bm25_db_load_docs()
        _doc_lookup.clear()

        if not _all_docs:
            _bm25 = None
            return

        # Try cached pickle first.
        try:
            if BM25_PICKLE_PATH.exists():
                import pickle
                with open(BM25_PICKLE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if hasattr(cached, "corpus_size") and cached.corpus_size == len(_all_docs):
                    _bm25 = cached
                    return
        except Exception:
            pass

        from rank_bm25 import BM25Okapi  # type: ignore
        _bm25 = BM25Okapi([doc.page_content.split() for doc in _all_docs])

        try:
            import pickle
            BM25_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(BM25_PICKLE_PATH, "wb") as f:
                pickle.dump(_bm25, f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Graph expansion helpers
# ---------------------------------------------------------------------------

def _build_doc_lookup() -> None:
    global _doc_lookup
    if _doc_lookup:
        return
    ensure_bm25()
    if not _all_docs:
        return
    _doc_lookup = {str(d.metadata.get("doc_id")): d for d in _all_docs if d.metadata.get("doc_id")}


def _graph_expand_docs(
    base_docs: List[Document],
    act_hint: Optional[str],
    section_hints: List[str],
    limit: int = 60,
) -> List[Document]:
    gs = get_graph_store()
    if not gs:
        return base_docs

    acts: set[str] = set()
    act_sections: dict[str, set[str]] = {}
    for d in base_docs[:20]:
        meta = d.metadata or {}
        act = str(meta.get("act") or "").strip()
        sec = str(meta.get("section") or "").strip()
        if act:
            acts.add(act)
            if sec:
                act_sections.setdefault(act, set()).add(sec)
    if act_hint:
        acts.add(act_hint)
        if section_hints:
            act_sections.setdefault(act_hint, set()).update(section_hints)

    _build_doc_lookup()
    if not _doc_lookup:
        return base_docs

    expanded: list[Document] = list(base_docs)
    seen_ids = {str(d.metadata.get("doc_id")) for d in base_docs if d.metadata.get("doc_id")}

    for act in acts:
        sections = sorted(act_sections.get(act, set()) or section_hints or [])
        try:
            cand_ids = gs.candidate_doc_ids(act=act, sections=sections, limit=limit)
        except Exception as e:
            logger.warning("Graph expansion failed for act %s: %s", act, e)
            continue
        for doc_id in cand_ids:
            if doc_id in seen_ids:
                continue
            doc = _doc_lookup.get(doc_id)
            if doc:
                expanded.append(doc)
                seen_ids.add(doc_id)
            if len(expanded) >= limit + len(base_docs):
                break
    return expanded


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------

def _mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    docs: List[Document],
    *,
    lambda_mult: float = 0.7,
    top_k: int = 10,
    section_penalty: float = 0.2,
    case_penalty: float = 0.1,
) -> List[Document]:
    if len(docs) <= top_k:
        return docs

    sim_to_query = doc_vecs @ query_vec
    sim_to_query = sim_to_query / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9)

    def key_section(idx: int):
        m = docs[idx].metadata or {}
        return (m.get("source_path"), m.get("act"), m.get("section"))

    def key_case(idx: int):
        m = docs[idx].metadata or {}
        return m.get("case_id") or m.get("case_name")

    selected: list[int] = [int(np.argmax(sim_to_query))]
    candidates = set(range(len(docs))) - set(selected)

    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
    doc_vecs_norm = doc_vecs / doc_norms

    while len(selected) < min(top_k, len(docs)):
        best_score, best_idx = -1e9, None
        for j in candidates:
            max_sim_sel = max(float(doc_vecs_norm[j] @ doc_vecs_norm[i]) for i in selected)
            penalty = (section_penalty if any(key_section(j) == key_section(i) and key_section(j) != (None, None, None) for i in selected) else 0.0)
            penalty += (case_penalty if any(key_case(j) and key_case(j) == key_case(i) for i in selected) else 0.0)
            score = lambda_mult * sim_to_query[j] - (1 - lambda_mult) * max_sim_sel - penalty
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [docs[i] for i in selected]


# ---------------------------------------------------------------------------
# Key retrieval functions
# ---------------------------------------------------------------------------

def _doc_key(doc: Document) -> str:
    return doc.metadata.get("doc_id") or str(id(doc))


def hybrid_retrieve(
    query: str,
    k: int = 10,
    max_distance: float = 0.58,
    *,
    use_mmr: bool = False,
    use_cross_encoder: bool = False,
    cross_encoder_fn=None,
    mmr_lambda: float = 0.7,
    mmr_section_penalty: float = 0.2,
    mmr_case_penalty: float = 0.1,
    fetch_k: int = 40,
    statute_boost: bool = True,
    graph_expand: bool = True,
    statute_hint: tuple[str | None, list[str]] | None = None,
) -> List[Document]:
    """
    Hybrid retrieval: 60% vector + 40% BM25.
    Returns merged, deduplicated, ranked results.
    """
    from ingestion import get_vector_store
    ensure_ingested()

    vector_hits: list[tuple[Document, float]] = []
    for attempt in range(VECTOR_QUERY_RETRIES + 1):
        try:
            k_val = min(128, max(fetch_k, k))
            vector_hits = get_vector_store().similarity_search_with_score(query, k=k_val)
            break
        except Exception as e:
            if attempt >= VECTOR_QUERY_RETRIES:
                print(f"Warning: vector search failed; falling back to BM25-only. Error: {e}")
                break
            time.sleep(VECTOR_QUERY_RETRY_SLEEP_S)

    vector_docs: dict[str, Document] = {}
    vector_scores: dict[str, float] = {}
    for d, dist in vector_hits:
        if dist <= max_distance:
            doc_id = _doc_key(d)
            vector_docs[doc_id] = d
            vector_scores[doc_id] = max(0.0, 1.0 - dist)

    # Relax threshold if no results — keep top-5 regardless.
    if not vector_docs and vector_hits:
        for d, dist in vector_hits[:5]:
            doc_id = _doc_key(d)
            vector_docs[doc_id] = d
            vector_scores[doc_id] = max(0.0, 1.0 - dist)

    ensure_bm25()
    act_hint, section_hints = statute_hint or detect_statute_query(query)

    bm25_docs: dict[str, Document] = {}
    bm25_scores: dict[str, float] = {}
    if _bm25 is not None and _all_docs:
        scores = _bm25.get_scores(query.split())
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        max_score = max(scores_list) if scores_list and max(scores_list) > 0 else 1.0
        top_n = min(max(fetch_k, k, 20), len(scores_list))
        for idx in heapq.nlargest(top_n, range(len(scores_list)), key=scores_list.__getitem__):
            d = _all_docs[idx]
            s = scores_list[idx]
            if s > 0:
                doc_id = _doc_key(d)
                bm25_docs[doc_id] = d
                bm25_scores[doc_id] = s / max_score

    merged_ids = set(vector_docs) | set(bm25_docs)
    scores_combined: dict[str, float] = {}
    merged: dict[str, Document] = {}

    for doc_id in merged_ids:
        combined = (vector_scores.get(doc_id, 0.0) * 0.6) + (bm25_scores.get(doc_id, 0.0) * 0.4)
        doc = vector_docs.get(doc_id) or bm25_docs[doc_id]
        if statute_boost:
            meta = doc.metadata or {}
            act = str(meta.get("act") or "").lower()
            sec = str(meta.get("section") or "").upper()
            if act_hint and act_hint.lower() in act:
                combined += 0.2
            if section_hints and sec and sec in {s.upper() for s in section_hints}:
                combined += 0.1
            if meta.get("doc_type") == "statute":
                combined += 0.05
        merged[doc_id] = doc
        scores_combined[doc_id] = combined

    ordered_docs = [doc for _, doc in sorted(merged.items(), key=lambda x: scores_combined[x[0]], reverse=True)]

    if use_mmr and ordered_docs:
        try:
            query_vec = np.asarray(emb.embed_query(query), dtype=float)
            cand_docs = ordered_docs[: max(fetch_k, k, 30)]
            doc_vecs = np.asarray(emb.embed_documents_parallel([d.page_content for d in cand_docs]), dtype=float)
            ordered_docs = _mmr(
                query_vec, doc_vecs, cand_docs,
                lambda_mult=mmr_lambda, top_k=k,
                section_penalty=mmr_section_penalty, case_penalty=mmr_case_penalty,
            )
        except Exception as e:
            print(f"Warning: MMR rerank skipped: {e}")
            ordered_docs = ordered_docs[:k]

    if use_cross_encoder and ordered_docs:
        fn = cross_encoder_fn or emb.cross_encoder_score
        try:
            cand = ordered_docs[: max(fetch_k, k, 30)]
            scores = fn(query, cand)
            paired = sorted(zip(cand[: len(scores)], scores), key=lambda x: x[1], reverse=True)
            ordered_docs = [d for d, _ in paired][:k]
        except Exception as e:
            print(f"Warning: cross-encoder rerank skipped: {e}")
            ordered_docs = ordered_docs[:k]

    if graph_expand and _graph_enabled():
        try:
            ordered_docs = _graph_expand_docs(ordered_docs[: max(fetch_k, k, 30)], act_hint, section_hints, limit=max(fetch_k, k, 80))
        except Exception as e:
            logger.warning("Graph expand skipped: %s", e)

    return ordered_docs[:k]


def filter_by_case_name(query: str, docs: List[Document]) -> List[Document]:
    query_lower = query.lower()
    anchored = [d for d in docs if any(t in d.page_content.lower() for t in query_lower.split() if len(t) > 3)]
    return anchored if anchored else docs


def rerank(docs: List[Document], query: Optional[str] = None) -> List[Document]:
    """Legal authority-aware reranking."""
    query_terms: list[str] = []
    if query:
        query_terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2]

    def score(doc: Document) -> int:
        s = 0
        meta = doc.metadata or {}
        if meta.get("doc_type") == "qa":
            s += 80
        if meta.get("act"):
            s += 100
        if meta.get("section"):
            s += 50
        if meta.get("doc_type") in {"pdf", "md", "json"}:
            s += 20
        if query_terms:
            text = doc.page_content.lower()
            matched = sum(1 for t in query_terms if t in text)
            s += int(matched / max(1, len(query_terms)) * 100)
        return s

    return sorted(docs, key=score, reverse=True)


def build_citation_prompt(query: str, docs: List[Document], max_chunks: int = 15) -> str:
    """Assemble a citation-grounded prompt from ranked docs."""
    lines = [
        "You are an expert Indian Legal AI Assistant (NyayGram).",
        "Your primary directive is to provide accurate, factual, and highly professional legal analysis.",
        "RULES:",
        "1. STRICTLY base your entire answer ONLY on the provided Context Sources below.",
        "2. If the Context Sources do not contain the answer, you MUST state that you lack sufficient information.",
        "3. NEVER hallucinate or invent laws, section numbers, or cases.",
        "4. Always cite specific Acts, Sections, and Case Law explicitly inline when you use them.",
        "5. Structure your response clearly using markdown headings, bullet points, and bold text for legal terms.",
        "\n==================== CONTEXT SOURCES ===================="
    ]
    
    for i, d in enumerate(docs[:max_chunks], start=1):
        meta = d.metadata or {}
        cite_parts = [str(meta["act"])] if meta.get("act") else []
        if meta.get("section"):
            cite_parts.append(f"Section {meta['section']}")
        if meta.get("citation"):
            cite_parts.append(str(meta["citation"]))
        
        cite_str = " | ".join(cite_parts) if cite_parts else (meta.get("source_path") or f"Source {i}")
        
        lines.append(f"\n--- [Source {i}: {cite_str}] ---")
        lines.append(d.page_content.strip())

    lines.append("\n=========================================================")
    lines.append(f"\nUSER QUESTION: {query.strip()}")
    lines.append("EXPERT LEGAL ANSWER:")
    return "\n".join(lines)
