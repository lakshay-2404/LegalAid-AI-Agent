from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from langchain_ollama.llms import OllamaLLM

from rag_core import answer_query


MODEL_NAME = os.environ.get("DEFAULT_LLM_MODEL", "qwen2.5:3b")
CACHE_FILE = Path("query_cache.json")
CACHE_MAX_ENTRIES = 10_000
CACHE_TRIM_SIZE = 1_000
CACHE_TTL_DAYS = 7


def build_model() -> OllamaLLM:
    return OllamaLLM(
        model=MODEL_NAME,
        temperature=0.1,
        num_ctx=2048,
        num_thread=4,
        top_p=0.85,
    )


def load_cache() -> Dict[str, Dict[str, str]]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not load cache: {exc}")
        return {}


cache: Dict[str, Dict[str, str]] = load_cache()


def get_cached(query: str) -> Optional[str]:
    if not query or len(query) > 1000:
        return None

    entry = cache.get(query)
    if not entry:
        return None

    try:
        timestamp = datetime.fromisoformat(entry.get("timestamp", ""))
    except ValueError:
        cache.pop(query, None)
        return None

    if (datetime.now() - timestamp).days > CACHE_TTL_DAYS:
        cache.pop(query, None)
        return None

    return entry.get("answer")


def save_cached(query: str, answer: str) -> None:
    if len(cache) > CACHE_MAX_ENTRIES:
        oldest_keys = sorted(cache, key=lambda key: cache[key].get("timestamp", ""))[:CACHE_TRIM_SIZE]
        for key in oldest_keys:
            cache.pop(key, None)

    cache[query] = {"answer": answer, "timestamp": datetime.now().isoformat()}

    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not save cache: {exc}")


def main() -> None:
    model = build_model()

    print("\nLegalAid AI Assistant (type 'q' to quit)")
    print("=" * 60)

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue

        cached = get_cached(query)
        if cached:
            print("\nCached response")
            print(cached)
            continue

        started_at = time.time()
        try:
            answer, docs = answer_query(query, model)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        save_cached(query, answer)

        elapsed = time.time() - started_at
        print(f"\nAnswer ({elapsed:.2f}s)")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        if docs:
            print("\nSources:")
            for index, doc in enumerate(docs, start=1):
                print(f"{index}. {doc.metadata}")


if __name__ == "__main__":
    main()
