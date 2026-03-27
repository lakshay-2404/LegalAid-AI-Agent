"""
ingestion_pipeline.py — CLI entry point for the ingestion pipeline.
Delegates to ingestion.ensure_ingested() via the public ingest() wrapper.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import ingestion


logger = logging.getLogger(__name__)


def ingest(*, force: bool = False, generate_docs: bool = True) -> None:
    """
    Production ingestion entrypoint.
    - Runs vector ingestion (Milvus + BM25 local corpus)
    - Optionally writes to Neo4j if ENABLE_GRAPH=1 and Neo4j is reachable
    - Regenerates docs/ARCHITECTURE.md when the system changes
    """
    ingestion.ensure_ingested(force=force)

    if generate_docs:
        try:
            from documentation_generator import regenerate_docs_if_needed
            regenerate_docs_if_needed(force=False)
        except Exception as e:
            logger.warning("Docs generation failed: %s", e)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest sources into Milvus (+ Neo4j optional).")
    ap.add_argument("--force", action="store_true", help="Force incremental ingestion run even if manifest is current")
    ap.add_argument("--no-docs", action="store_true", help="Disable docs regeneration after ingestion")
    ap.add_argument("--enable-graph", action="store_true", help="Enable Neo4j writes for this run (sets ENABLE_GRAPH=1)")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of parallel embedding workers (default: INGEST_WORKERS env, currently {ingestion.BATCH_SIZE})",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=f"Embedding batch size (default: INGEST_BATCH_SIZE env, currently {ingestion.BATCH_SIZE})",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    if args.enable_graph:
        os.environ["ENABLE_GRAPH"] = "1"
    if args.workers is not None:
        os.environ["INGEST_WORKERS"] = str(args.workers)
    if args.batch_size is not None:
        os.environ["INGEST_BATCH_SIZE"] = str(args.batch_size)

    ingest(force=args.force, generate_docs=not args.no_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
