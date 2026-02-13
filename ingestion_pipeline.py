from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import vector

from documentation_generator import regenerate_docs_if_needed

logger = logging.getLogger(__name__)


def ingest(*, force: bool = False, generate_docs: bool = True) -> None:
    """
    Production ingestion entrypoint.
    - Runs vector ingestion (Milvus + BM25 local corpus)
    - Optionally writes to Neo4j if ENABLE_GRAPH=1 and Neo4j is reachable
    - Regenerates docs/ARCHITECTURE.md when the system changes
    """
    vector.ensure_ingested(force=force)

    # Graph ingestion happens during vector ingestion when ENABLE_GRAPH=1.
    if generate_docs:
        try:
            regenerate_docs_if_needed(force=False)
        except Exception as e:
            logger.warning("Docs generation failed: %s", e)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest sources into Milvus (+ Neo4j optional).")
    ap.add_argument("--force", action="store_true", help="Force incremental ingestion run")
    ap.add_argument("--no-docs", action="store_true", help="Disable docs regeneration")
    ap.add_argument("--enable-graph", action="store_true", help="Enable Neo4j writes for this run")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.enable_graph:
        os.environ["ENABLE_GRAPH"] = "1"

    ingest(force=args.force, generate_docs=not args.no_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

