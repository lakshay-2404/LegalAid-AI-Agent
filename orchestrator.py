from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.documents import Document

import vector
from graph_store import GraphStore, Neo4jConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    enable_graph: bool = False
    graph_candidate_limit: int = 1500
    graph_cite_hops: int = 1
    max_doc_id_filter: int = 2000

    @staticmethod
    def from_env() -> "OrchestratorConfig":
        enable_graph = os.environ.get("ENABLE_GRAPH", "0").strip() in {"1", "true", "yes", "on"}
        return OrchestratorConfig(enable_graph=enable_graph)


class HybridLegalRAGOrchestrator:
    """
    Query orchestration for Vector+Graph hybrid retrieval.

    Retrieval flow (query-time):
    1) Entity extraction (act + section refs)
    2) Neo4j traversal to produce candidate `doc_id` set
    3) Milvus similarity search restricted to candidates (when small enough)
    4) Fallback to normal vector search when graph yields no candidates

    Reranking + confidence scoring stays in `rag_core.py` to preserve existing logic.
    """

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self._cfg = cfg
        self._graph: Optional[GraphStore] = None
        self._graph_init_error: Optional[str] = None

    def _get_graph(self) -> Optional[GraphStore]:
        if not self._cfg.enable_graph:
            return None
        if self._graph is not None:
            return self._graph
        if self._graph_init_error:
            return None
        try:
            gs = GraphStore(Neo4jConfig.from_env())
            gs.ensure_schema()
            self._graph = gs
            return gs
        except Exception as e:
            self._graph_init_error = str(e)
            logger.warning("Neo4j disabled (init failed): %s", e)
            return None

    def extract_query_entities(self, query: str) -> dict[str, Any]:
        # Reuse the existing statute metadata extractor to keep Act naming consistent.
        meta = vector.extract_legal_structure(query or "")
        act = meta.get("act")

        # Reuse the section ref extractor from rag_core-like logic but keep local to avoid cycles.
        sec_refs = []
        try:
            import re

            for m in re.finditer(r"\b(?:section|sec|s)\.?\s*([0-9]{1,4}[a-z]?)\b", (query or "").lower()):
                sec_refs.append(m.group(1))
        except Exception:
            sec_refs = []

        # Normalize (13b -> 13B)
        norm: list[str] = []
        seen: set[str] = set()
        for r in sec_refs:
            r = (r or "").strip()
            if not r:
                continue
            import re

            m = re.match(r"^([0-9]{1,4})([a-z])?$", r, flags=re.IGNORECASE)
            if not m:
                continue
            sec = m.group(1)
            suf = (m.group(2) or "").upper()
            s = f"{sec}{suf}"
            if s not in seen:
                norm.append(s)
                seen.add(s)

        return {"act": act, "sections": norm}

    def graph_candidates(self, *, act: str | None, sections: list[str] | None) -> list[str]:
        gs = self._get_graph()
        if not gs:
            return []
        try:
            return gs.candidate_doc_ids(
                act=act,
                sections=sections,
                limit=self._cfg.graph_candidate_limit,
                cite_hops=self._cfg.graph_cite_hops,
            )
        except Exception as e:
            logger.warning("Neo4j traversal failed: %s", e)
            return []

    def graph_vector_retrieve(self, query: str, *, k: int = 20) -> list[Document]:
        vector.ensure_ingested()
        vs = vector.get_vector_store()

        ent = self.extract_query_entities(query)
        act = ent.get("act")
        sections = ent.get("sections") or []

        candidate_ids = self.graph_candidates(act=act, sections=sections)
        if not candidate_ids:
            # Graph not available or no candidates: normal vector retrieval.
            try:
                return [d for d, _ in vs.similarity_search_with_score(query, k=k)]
            except Exception:
                return []

        # Restrict vector retrieval to graph-derived candidates (when small enough).
        filt: dict[str, Any] = {}
        if act:
            filt["act"] = act
        if len(candidate_ids) <= self._cfg.max_doc_id_filter:
            filt = {"$and": [filt, {"doc_id": {"$in": candidate_ids}}]} if filt else {"doc_id": {"$in": candidate_ids}}
        else:
            logger.info("Graph candidate set too large (%s); skipping doc_id filter", len(candidate_ids))

        try:
            hits = vs.similarity_search_with_score(query, k=k, filter=filt or None)
            docs = [d for d, _dist in hits]
        except Exception:
            docs = []

        # Backfill if strict filter returns too few.
        if len(docs) < max(5, k // 2):
            try:
                backfill = [d for d, _ in vs.similarity_search_with_score(query, k=k)]
            except Exception:
                backfill = []
            seen = {vector._doc_key(d) for d in docs}
            for d in backfill:
                if vector._doc_key(d) in seen:
                    continue
                docs.append(d)
                if len(docs) >= k:
                    break
        return docs[:k]


_orch: Optional[HybridLegalRAGOrchestrator] = None


def get_orchestrator() -> HybridLegalRAGOrchestrator:
    global _orch
    if _orch is None:
        _orch = HybridLegalRAGOrchestrator(OrchestratorConfig.from_env())
    return _orch

