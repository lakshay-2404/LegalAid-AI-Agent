from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


MilvusMetric = Literal["COSINE", "IP", "L2"]
MilvusIndexType = Literal["HNSW", "IVF_FLAT", "IVF_PQ", "AUTOINDEX"]


def _escape_milvus_str(value: str) -> str:
    # Milvus expr strings use double quotes for string literals.
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _dict_to_expr(where: dict[str, Any] | None) -> str | None:
    """
    Convert a Chroma-style where/filter dict to a Milvus expr.

    Supports:
    - {"field": "value"} (equality)
    - {"$and": [ {...}, {...} ]}
    - {"$or":  [ {...}, {...} ]}
    - {"field": {"$in": [..]}}  (limited)
    """
    if not where:
        return None

    def one(d: dict[str, Any]) -> str:
        if not d:
            return ""
        if "$and" in d:
            parts = [one(x) for x in (d.get("$and") or []) if one(x)]
            return "(" + " && ".join(parts) + ")" if parts else ""
        if "$or" in d:
            parts = [one(x) for x in (d.get("$or") or []) if one(x)]
            return "(" + " || ".join(parts) + ")" if parts else ""

        if len(d) != 1:
            # Implicit AND for multi-field dicts
            parts = [one({k: v}) for k, v in d.items()]
            parts = [p for p in parts if p]
            return "(" + " && ".join(parts) + ")" if parts else ""

        field, value = next(iter(d.items()))
        if isinstance(value, dict):
            if "$in" in value:
                items = value.get("$in") or []
                items_escaped = [f'"{_escape_milvus_str(str(x))}"' for x in items]
                return f"{field} in [{', '.join(items_escaped)}]"
            raise ValueError(f"Unsupported where operator for field {field}: {value}")

        if value is None:
            # No nulls in expr; best effort (treat as empty string)
            return f'{field} == ""'
        return f'{field} == "{_escape_milvus_str(str(value))}"'

    expr = one(where)
    return expr or None


@dataclass(frozen=True)
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection: str = "indian_legal_rag"
    dim: int = 768

    metric: MilvusMetric = "COSINE"
    index_type: MilvusIndexType = "HNSW"
    index_params: dict[str, Any] | None = None
    search_params: dict[str, Any] | None = None

    text_max_length: int = 65535
    varchar_max_length: int = 512
    metadata_json_max_length: int = 65535

    @staticmethod
    def from_env(*, dim: int) -> "MilvusConfig":
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = int(os.environ.get("MILVUS_PORT", "19530"))
        collection = os.environ.get("MILVUS_COLLECTION", "indian_legal_rag")
        metric = os.environ.get("MILVUS_METRIC", "COSINE").upper()
        index_type = os.environ.get("MILVUS_INDEX_TYPE", "HNSW").upper()
        return MilvusConfig(host=host, port=port, collection=collection, dim=dim, metric=metric, index_type=index_type)


class MilvusVectorStore:
    """
    Minimal LangChain-compatible vector store adapter backed by Milvus (pymilvus).

    Goals:
    - Preserve stable ids (`doc_id`) as primary keys
    - Support metadata filtering (Chroma-like dict -> Milvus expr)
    - Provide similarity_search / similarity_search_with_score used by existing pipeline
    - Support explicit-embedding inserts for Chroma -> Milvus migration
    """

    def __init__(
        self,
        *,
        config: MilvusConfig,
        embed_documents: Callable[[list[str]], list[list[float]]],
        embed_query: Callable[[str], list[float]] | None = None,
        connect_alias: str = "default",
    ) -> None:
        self._config = config
        self._embed_documents = embed_documents
        self._embed_query = embed_query or (lambda q: embed_documents([q])[0])
        self._alias = connect_alias

        self._collection = None
        self._connect()
        self._ensure_collection()

    @property
    def collection_name(self) -> str:
        return self._config.collection

    def _connect(self) -> None:
        try:
            from pymilvus import connections
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Missing dependency pymilvus: {e}") from e
        connections.connect(alias=self._alias, host=self._config.host, port=str(self._config.port))

    def _ensure_collection(self) -> None:
        from pymilvus import (  # type: ignore[import-untyped]
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(self._config.collection, using=self._alias):
            self._collection = Collection(self._config.collection, using=self._alias)
            # Best-effort schema dim validation.
            try:
                for f in self._collection.schema.fields:
                    if f.name == "embedding":
                        existing_dim = int(getattr(f, "params", {}).get("dim") or getattr(f, "dim", 0) or 0)
                        if existing_dim and existing_dim != self._config.dim:
                            raise ValueError(
                                f"Milvus collection dim mismatch: existing={existing_dim} expected={self._config.dim}"
                            )
            except Exception:
                # If schema introspection changes, do not hard fail here.
                pass
        else:
            fields = [
                FieldSchema(
                    name="doc_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=self._config.varchar_max_length,
                ),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._config.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self._config.text_max_length),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=self._config.varchar_max_length),
                FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="act", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="subsection", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="clause", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="citation", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="char_count", dtype=DataType.INT64),
                FieldSchema(
                    name="metadata_json",
                    dtype=DataType.VARCHAR,
                    max_length=self._config.metadata_json_max_length,
                ),
            ]

            schema = CollectionSchema(fields=fields, description="Legal RAG chunks", enable_dynamic_field=False)
            self._collection = Collection(self._config.collection, schema=schema, using=self._alias)

        # Index setup (idempotent)
        index_params = self._config.index_params or self._default_index_params()
        try:
            existing = {idx.field_name for idx in (self._collection.indexes or [])}
        except Exception:
            existing = set()

        if "embedding" not in existing:
            logger.info("Creating Milvus index (%s) on %s...", self._config.index_type, self._config.collection)
            try:
                self._collection.create_index("embedding", index_params=index_params)
            except Exception as e:
                logger.warning("Milvus index creation failed; continuing without index: %s", e)

        try:
            self._collection.load()
        except Exception as e:
            logger.warning("Milvus collection load failed (will still attempt queries): %s", e)

    def _default_index_params(self) -> dict[str, Any]:
        if self._config.index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self._config.metric,
                "params": {"M": 16, "efConstruction": 200},
            }
        if self._config.index_type == "IVF_FLAT":
            return {"index_type": "IVF_FLAT", "metric_type": self._config.metric, "params": {"nlist": 2048}}
        if self._config.index_type == "IVF_PQ":
            return {"index_type": "IVF_PQ", "metric_type": self._config.metric, "params": {"nlist": 2048, "m": 64}}
        return {"index_type": "AUTOINDEX", "metric_type": self._config.metric, "params": {}}

    def _default_search_params(self) -> dict[str, Any]:
        if self._config.index_type == "HNSW":
            return {"metric_type": self._config.metric, "params": {"ef": 64}}
        if self._config.index_type.startswith("IVF"):
            return {"metric_type": self._config.metric, "params": {"nprobe": 16}}
        return {"metric_type": self._config.metric, "params": {}}

    def count(self) -> int:
        c = self._collection
        if c is None:
            return 0
        try:
            return int(c.num_entities)
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            self._collection.flush()  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Milvus flush failed: %s", e)

    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        expr_ids = ", ".join(f'"{_escape_milvus_str(i)}"' for i in ids)
        expr = f"doc_id in [{expr_ids}]"
        try:
            self._collection.delete(expr)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Milvus delete failed (%s ids): %s", len(ids), e)

    def upsert_embeddings(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if len(ids) != len(embeddings) or len(ids) != len(documents) or len(ids) != len(metadatas):
            raise ValueError("ids/embeddings/documents/metadatas length mismatch")

        # Delete existing to keep id stable and avoid duplicates across retries.
        self.delete_by_ids(ids)

        rows = self._rows_from_payload(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self._collection.insert(rows)  # type: ignore[union-attr]

    def add_documents(self, docs: list[Document], *, ids: list[str]) -> None:
        if not docs:
            return
        if len(docs) != len(ids):
            raise ValueError("docs/ids length mismatch")

        texts = [d.page_content for d in docs]
        metadatas = [d.metadata or {} for d in docs]

        embeddings = self._embed_documents(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding count mismatch")

        self.upsert_embeddings(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,  # noqa: A002 (keep name for compatibility)
    ) -> list[tuple[Document, float]]:
        expr = _dict_to_expr(filter)
        qv = self._embed_query(query)
        search_params = self._config.search_params or self._default_search_params()

        hits = self._collection.search(  # type: ignore[union-attr]
            data=[qv],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=[
                "doc_id",
                "text",
                "doc_type",
                "source",
                "source_path",
                "act",
                "chapter",
                "section",
                "subsection",
                "clause",
                "citation",
                "char_count",
                "metadata_json",
            ],
        )

        out: list[tuple[Document, float]] = []
        for hit in (hits[0] if hits else []):
            ent = hit.entity
            meta = self._metadata_from_entity(ent)
            doc = Document(page_content=str(getattr(ent, "text", "") or ""), metadata=meta)
            out.append((doc, float(hit.distance)))
        return out

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[Document]:
        return [d for d, _s in self.similarity_search_with_score(query, k=k, filter=filter)]

    def get_docs(self, where: dict[str, Any], *, limit: int | None = None) -> list[Document]:
        expr = _dict_to_expr(where)
        if not expr:
            return []
        fields = [
            "doc_id",
            "text",
            "doc_type",
            "source",
            "source_path",
            "act",
            "chapter",
            "section",
            "subsection",
            "clause",
            "citation",
            "char_count",
            "metadata_json",
        ]
        res = self._collection.query(  # type: ignore[union-attr]
            expr=expr,
            output_fields=fields,
            limit=int(limit) if limit is not None else 10_000,
        )
        out: list[Document] = []
        for row in res or []:
            meta = self._metadata_from_row(row)
            out.append(Document(page_content=str(row.get("text") or ""), metadata=meta))
        return out

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        # Guard: expr length can explode; cap.
        if len(ids) > 5000:
            ids = ids[:5000]
        expr_ids = ", ".join(f'"{_escape_milvus_str(i)}"' for i in ids)
        expr = f"doc_id in [{expr_ids}]"
        res = self._collection.query(  # type: ignore[union-attr]
            expr=expr,
            output_fields=[
                "doc_id",
                "text",
                "doc_type",
                "source",
                "source_path",
                "act",
                "chapter",
                "section",
                "subsection",
                "clause",
                "citation",
                "char_count",
                "metadata_json",
            ],
            limit=len(ids),
        )
        out: list[Document] = []
        for row in res or []:
            meta = self._metadata_from_row(row)
            out.append(Document(page_content=str(row.get("text") or ""), metadata=meta))
        return out

    def _rows_from_payload(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> list[list[Any]]:
        doc_type = []
        source = []
        source_path = []
        act = []
        chapter = []
        section = []
        subsection = []
        clause = []
        citation = []
        char_count = []
        metadata_json = []

        for meta, text in zip(metadatas, documents):
            m = meta or {}
            doc_type.append(str(m.get("doc_type") or ""))
            source.append(str(m.get("source") or ""))
            source_path.append(str(m.get("source_path") or ""))
            act.append(str(m.get("act") or ""))
            chapter.append(str(m.get("chapter") or ""))
            section.append(str(m.get("section") or ""))
            subsection.append(str(m.get("subsection") or ""))
            clause.append(str(m.get("clause") or ""))
            citation.append(str(m.get("citation") or ""))
            char_count.append(int(m.get("char_count") or len(text or "")))
            try:
                metadata_json.append(json.dumps(m, ensure_ascii=False))
            except Exception:
                metadata_json.append("{}")

        return [
            ids,
            embeddings,
            documents,
            doc_type,
            source,
            source_path,
            act,
            chapter,
            section,
            subsection,
            clause,
            citation,
            char_count,
            metadata_json,
        ]

    def _metadata_from_entity(self, ent: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for k in (
            "doc_id",
            "doc_type",
            "source",
            "source_path",
            "act",
            "chapter",
            "section",
            "subsection",
            "clause",
            "citation",
            "char_count",
        ):
            try:
                v = getattr(ent, k, None)
            except Exception:
                v = None
            if v not in (None, ""):
                meta[k] = v
        meta_json = getattr(ent, "metadata_json", None)
        if meta_json:
            try:
                meta.update(json.loads(meta_json))
            except Exception:
                pass
        if "doc_id" not in meta:
            # If missing, keep milvus pk if present in dynamic shape.
            try:
                meta["doc_id"] = getattr(ent, "doc_id")
            except Exception:
                pass
        return meta

    def _metadata_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for k in (
            "doc_id",
            "doc_type",
            "source",
            "source_path",
            "act",
            "chapter",
            "section",
            "subsection",
            "clause",
            "citation",
            "char_count",
        ):
            v = row.get(k)
            if v not in (None, ""):
                meta[k] = v
        meta_json = row.get("metadata_json")
        if meta_json:
            try:
                meta.update(json.loads(meta_json))
            except Exception:
                pass
        return meta


def infer_embedding_dim(embed_query: Callable[[str], list[float]]) -> int:
    vec = embed_query("dimension probe")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("Embedding probe failed (empty vector)")
    return int(len(vec))
