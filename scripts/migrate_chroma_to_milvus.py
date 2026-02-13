from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings

from vector_store import MilvusConfig, MilvusVectorStore, infer_embedding_dim

logger = logging.getLogger(__name__)


def _load_chroma_collection(persist_dir: Path, collection_name: str):
    try:
        import chromadb
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency chromadb: {e}") from e

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


def _iter_chroma_batches(collection, *, batch_size: int) -> tuple[int, Any]:
    try:
        total = int(collection.count())
    except Exception:
        total = -1

    offset = 0
    while True:
        kwargs = {"include": ["documents", "metadatas", "embeddings"], "limit": batch_size, "offset": offset}
        try:
            data = collection.get(**kwargs)
        except TypeError:
            # Older chroma: may not support offset. Fall back to full get (not scalable).
            data = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids") or []
        if not ids:
            break
        yield offset, data
        if "offset" not in kwargs:
            break
        offset += len(ids)

    return total, None


def _safe_meta(meta: Any, doc_id: str) -> dict[str, Any]:
    m = meta if isinstance(meta, dict) else {}
    if doc_id and "doc_id" not in m:
        m["doc_id"] = doc_id
    return m


def migrate(
    *,
    chroma_persist_dir: Path,
    chroma_collection: str,
    milvus_host: str,
    milvus_port: int,
    milvus_collection: str,
    embedding_model: str,
    batch_size: int,
    progress_path: Path | None,
    resume_offset: int,
    insert_retries: int,
    insert_retry_sleep_s: float,
) -> None:
    embeddings = OllamaEmbeddings(model=embedding_model)
    dim = infer_embedding_dim(embeddings.embed_query)

    mv = MilvusVectorStore(
        config=MilvusConfig(host=milvus_host, port=milvus_port, collection=milvus_collection, dim=dim),
        embed_documents=embeddings.embed_documents,
        embed_query=embeddings.embed_query,
    )

    collection = _load_chroma_collection(chroma_persist_dir, chroma_collection)
    logger.info("Loaded Chroma collection: %s (persist=%s)", chroma_collection, chroma_persist_dir)

    migrated = 0
    failed: list[str] = []
    start = time.time()

    for offset, data in _iter_chroma_batches(collection, batch_size=batch_size):
        if offset < resume_offset:
            continue

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        vecs = data.get("embeddings") or []

        if not ids:
            continue

        # Some Chroma builds return embeddings=None unless explicitly enabled.
        if not vecs or len(vecs) != len(ids):
            logger.warning("Batch offset=%s missing embeddings; recomputing via embedder.", offset)
            texts = [d or "" for d in docs]
            vecs = embeddings.embed_documents(texts)

        # Validate dim
        for v in vecs[:3]:
            if v and len(v) != dim:
                raise RuntimeError(f"Embedding dim mismatch inside batch: got={len(v)} expected={dim}")

        ids2: list[str] = []
        vecs2: list[list[float]] = []
        docs2: list[str] = []
        metas2: list[dict[str, Any]] = []

        for doc_id, doc, meta, vec in zip(ids, docs, metas, vecs):
            if not doc_id:
                continue
            if not vec or not isinstance(vec, list):
                failed.append(str(doc_id))
                continue
            if len(vec) != dim:
                failed.append(str(doc_id))
                continue
            ids2.append(str(doc_id))
            docs2.append(str(doc or ""))
            metas2.append(_safe_meta(meta, str(doc_id)))
            vecs2.append([float(x) for x in vec])

        try:
            last_err: Exception | None = None
            for attempt in range(insert_retries + 1):
                try:
                    mv.upsert_embeddings(ids=ids2, embeddings=vecs2, documents=docs2, metadatas=metas2)
                    migrated += len(ids2)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt >= insert_retries:
                        break
                    time.sleep(max(0.0, insert_retry_sleep_s))
            if last_err is not None:
                raise last_err
        except Exception as e:
            logger.exception("Milvus insert failed at offset=%s after retries: %s", offset, e)
            failed.extend([str(i) for i in ids2])

        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": time.time(),
                "last_offset": offset,
                "migrated": migrated,
                "failed_count": len(failed),
            }
            progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if migrated and migrated % (batch_size * 5) == 0:
            elapsed = time.time() - start
            logger.info("Progress: migrated=%s elapsed=%.1fs rate=%.1f items/s", migrated, elapsed, migrated / max(1, elapsed))

    try:
        mv.flush()
    except Exception:
        pass

    logger.info("Done. Migrated=%s Failed=%s MilvusCount=%s", migrated, len(failed), mv.count())
    if failed:
        logger.warning("Some ids failed to migrate. First 20: %s", failed[:20])


def main() -> int:
    p = argparse.ArgumentParser(description="Migrate a persistent Chroma collection to Milvus (preserving vectors/ids).")
    p.add_argument("--chroma-persist-dir", default="chrome_langchain_db", help="Chroma persist directory")
    p.add_argument("--chroma-collection", default="indian_legal_rag", help="Chroma collection name")
    p.add_argument("--milvus-host", default="localhost")
    p.add_argument("--milvus-port", type=int, default=19530)
    p.add_argument("--milvus-collection", default="indian_legal_rag")
    p.add_argument("--embedding-model", default="nomic-embed-text")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--progress", default="chrome_langchain_db/migration_progress.json")
    p.add_argument("--resume-offset", type=int, default=0)
    p.add_argument("--insert-retries", type=int, default=2)
    p.add_argument("--insert-retry-sleep-s", type=float, default=1.0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    migrate(
        chroma_persist_dir=Path(args.chroma_persist_dir),
        chroma_collection=args.chroma_collection,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        milvus_collection=args.milvus_collection,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        progress_path=Path(args.progress) if args.progress else None,
        resume_offset=args.resume_offset,
        insert_retries=args.insert_retries,
        insert_retry_sleep_s=args.insert_retry_sleep_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
