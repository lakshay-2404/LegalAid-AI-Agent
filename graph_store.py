from __future__ import annotations

import base64
import json
import logging
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


_SECTION_REF_RE = re.compile(r"\b(?:section|sec|s)\.?\s*([0-9]{1,4}[a-z]?)\b", re.IGNORECASE)


def extract_section_refs(text: str) -> list[str]:
    """
    Extract normalized section references from free text.
    Returns values like ["27", "13B"].
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _SECTION_REF_RE.finditer(text or ""):
        raw = (m.group(1) or "").strip()
        m2 = re.match(r"^([0-9]{1,4})([a-z])?$", raw, flags=re.IGNORECASE)
        if not m2:
            continue
        sec = m2.group(1)
        suf = (m2.group(2) or "").upper()
        norm = f"{sec}{suf}"
        if norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    query_uri: str
    user: str
    password: str
    database: str = ""

    @staticmethod
    def from_env() -> "Neo4jConfig":
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        query_uri = os.environ.get("NEO4J_QUERY_URI", "").strip()
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4j_password")
        database = os.environ.get("NEO4J_DATABASE", "").strip()
        return Neo4jConfig(uri=uri, query_uri=query_uri, user=user, password=password, database=database)


class GraphStore:
    """
    Neo4j-backed legal knowledge graph.

    Graph model (superset of requested nodes to support chunk-level retrieval):
    - (:Act {name})
    - (:Section {key, act, section, subsection, clause, citation})
    - (:Chunk {doc_id, source, source_path, doc_type})
    - Relationships:
      - (Act)-[:HAS_SECTION]->(Section)
      - (Section)-[:HAS_CHUNK]->(Chunk)
      - (Section)-[:CITES]->(Section)  (heuristic, same-act)
      - (Section)-[:AMENDED_BY]->(Amendment) (optional extension)
    """

    def __init__(self, cfg: Neo4jConfig) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Missing dependency neo4j: {e}") from e

        self._cfg = cfg
        self._query_uri = (cfg.query_uri or "").strip()
        self._driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.user, cfg.password),
            max_connection_pool_size=50,
            connection_timeout=5.0,  # Prevent hanging
        )

    def _session(self):
        """Open a session, routing to the configured database (required for Aura)."""
        kwargs = {}
        if self._cfg.database:
            kwargs["database"] = self._cfg.database
        return self._driver.session(**kwargs)

    def _flatten_cypher(self, statement: str) -> str:
        """Query API requires single-line Cypher; normalize whitespace."""
        return " ".join((statement or "").split())

    def _run_query_api(self, statement: str, parameters: dict | None = None) -> dict:
        if not self._query_uri:
            raise RuntimeError("NEO4J_QUERY_URI is not set for query API access.")
        payload = {
            "statement": self._flatten_cypher(statement),
            "parameters": parameters or {},
        }
        auth_bytes = f"{self._cfg.user}:{self._cfg.password}".encode("utf-8")
        auth_header = base64.b64encode(auth_bytes).decode("ascii")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {auth_header}",
        }
        req = urllib_request.Request(
            self._query_uri,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
        except urllib_error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"Query API HTTP error {e.code}: {detail or e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"Query API request failed: {e}") from e
        try:
            data = json.loads(body or "{}")
        except Exception as e:
            raise RuntimeError(f"Query API returned invalid JSON: {e}") from e
        if data.get("errors"):
            raise RuntimeError(f"Query API error response: {data.get('errors')}")
        return data

    def _extract_field_values(self, data: dict, field: str) -> list[str]:
        if not data:
            return []
        block = data.get("data")
        if not isinstance(block, dict):
            return []
        fields = block.get("fields") or []
        values = block.get("values") or []
        if field not in fields:
            return []
        idx = fields.index(field)
        out: list[str] = []
        for row in values:
            if isinstance(row, list) and idx < len(row):
                val = row[idx]
                if val is not None:
                    out.append(str(val))
        return out

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def ensure_schema(self) -> None:
        stmts = [
            "CREATE CONSTRAINT act_name IF NOT EXISTS FOR (a:Act) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT section_key IF NOT EXISTS FOR (s:Section) REQUIRE s.key IS UNIQUE",
            "CREATE CONSTRAINT chunk_doc_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.doc_id IS UNIQUE",
            "CREATE CONSTRAINT case_name IF NOT EXISTS FOR (c:Case) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT amendment_id IF NOT EXISTS FOR (a:Amendment) REQUIRE a.id IS UNIQUE",
        ]
        # Composite index for fast (act, section) lookups used by the retrieval queries.
        index_stmts = [
            "CREATE INDEX section_act_section IF NOT EXISTS FOR (s:Section) ON (s.act, s.section)",
        ]
        with self._session() as sess:
            for s in stmts + index_stmts:
                try:
                    sess.run(s)
                except Exception as e:
                    logger.warning("Neo4j schema statement failed: %s (%s)", s, e)

    @staticmethod
    def _section_key(meta: dict[str, Any]) -> Optional[str]:
        act = str(meta.get("act") or "").strip()
        section = str(meta.get("section") or "").strip()
        if not act or not section:
            return None
        subsection = str(meta.get("subsection") or "").strip()
        clause = str(meta.get("clause") or "").strip()
        return "|".join([act, section, subsection, clause])

    def upsert_chunks(self, docs: list[Document]) -> None:
        if not docs:
            return
        rows: list[dict[str, Any]] = []
        for d in docs:
            meta = d.metadata or {}
            doc_id = str(meta.get("doc_id") or "").strip()
            if not doc_id:
                continue

            act = str(meta.get("act") or "").strip()
            section = str(meta.get("section") or "").strip()
            subsection = str(meta.get("subsection") or "").strip()
            clause = str(meta.get("clause") or "").strip()
            citation = str(meta.get("citation") or "").strip()
            source = str(meta.get("source") or "").strip()
            source_path = str(meta.get("source_path") or "").strip()
            doc_type = str(meta.get("doc_type") or "").strip()

            section_key = self._section_key(meta)
            # Skip chunks that have no act/section — they can't form graph
            # relationships and would just be orphan Chunk nodes.
            if not section_key:
                continue

            refs = extract_section_refs(d.page_content or "")
            rows.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "source_path": source_path,
                    "doc_type": doc_type,
                    "act": act,
                    "section": section,
                    "subsection": subsection,
                    "clause": clause,
                    "citation": citation,
                    "section_key": section_key,
                    "refs": refs,
                }
            )

        if not rows:
            return

        q = """
        UNWIND $rows AS r
        MERGE (c:Chunk {doc_id: r.doc_id})
        SET c.source = r.source,
            c.source_path = r.source_path,
            c.doc_type = r.doc_type

        WITH c, r
        MERGE (a:Act {name: r.act})
        MERGE (s:Section {key: r.section_key})
        SET s.act = r.act,
            s.section = r.section,
            s.subsection = r.subsection,
            s.clause = r.clause,
            s.citation = r.citation
        MERGE (a)-[:HAS_SECTION]->(s)
        MERGE (s)-[:HAS_CHUNK]->(c)

        WITH s, r
        UNWIND r.refs AS ref
        WITH s, r, ref
        WHERE ref IS NOT NULL AND ref <> "" AND r.act <> "" AND r.section <> "" AND ref <> r.section
        MERGE (s2:Section {key: r.act + "|" + ref + "||"})
        SET s2.act = r.act, s2.section = ref
        MERGE (s)-[:CITES]->(s2)
        """

        with self._session() as sess:
            sess.execute_write(lambda tx: tx.run(q, rows=rows))

    def delete_chunks(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        q = """
        UNWIND $ids AS id
        MATCH (c:Chunk {doc_id: id})
        DETACH DELETE c
        """
        with self._session() as sess:
            sess.execute_write(lambda tx: tx.run(q, ids=[str(i) for i in doc_ids if i]))

    def chunk_count(self) -> int:
        q = """
        MATCH (c:Chunk)
        RETURN count(c) AS chunk_count
        """
        if self._query_uri:
            data = self._run_query_api(q, {})
            vals = self._extract_field_values(data, "chunk_count")
            return int(vals[0]) if vals else 0
        with self._session() as sess:
            rec = sess.run(q).single()
        return int((rec or {}).get("chunk_count") or 0)

    def candidate_doc_ids(
        self,
        *,
        act: str | None = None,
        sections: list[str] | None = None,
        limit: int = 1500,
        cite_hops: int = 1,
    ) -> list[str]:
        act = (act or "").strip()
        sections = [s.strip() for s in (sections or []) if s and str(s).strip()]

        if not act:
            return []

        if sections:
            q = """
            MATCH (a:Act {name: $act})-[:HAS_SECTION]->(s:Section)
            WHERE s.section IN $sections
            OPTIONAL MATCH (s)-[:CITES]->(s2:Section)
            WITH collect(DISTINCT s) + collect(DISTINCT s2) AS allSections
            UNWIND allSections AS sec
            MATCH (sec)-[:HAS_CHUNK]->(c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id
            LIMIT $limit
            """
            params = {"act": act, "sections": sections, "limit": int(limit)}
        else:
            q = """
            MATCH (a:Act {name: $act})-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
            RETURN DISTINCT c.doc_id AS doc_id
            LIMIT $limit
            """
            params = {"act": act, "limit": int(limit)}

        if self._query_uri:
            data = self._run_query_api(q, params)
            return self._extract_field_values(data, "doc_id")
        with self._session() as sess:
            res = sess.run(q, **params)
            return [str(r["doc_id"]) for r in res if r.get("doc_id")]
