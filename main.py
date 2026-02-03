import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from langchain_ollama.llms import OllamaLLM
from rag_core import answer_query

# =========================
# CONFIG
# =========================

MODEL_NAME = "llama3.1:8b"
CACHE_FILE = Path("query_cache.json")

# =========================
# MODEL
# =========================

model = OllamaLLM(
    model=MODEL_NAME,
    temperature=0.1,  # Lower for more consistent legal answers
    num_ctx=4096,  # Larger context window for longer legal documents
    num_thread=4,
    top_p=0.85,  # More focused on likely tokens
)

# =========================
# CACHE
# =========================

cache: Dict[str, Dict] = {}

if CACHE_FILE.exists():
    try:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Warning: Could not load cache: {e}")
        cache = {}

def get_cached(query: str) -> Optional[str]:
    """Safely retrieve cached answer with validation."""
    # Sanitize cache key
    if not query or len(query) > 1000:
        return None
    
    entry = cache.get(query)
    if not entry:
        return None

    try:
        ts = datetime.fromisoformat(entry.get("timestamp", ""))
    except (ValueError, KeyError):
        # Invalid timestamp, remove entry
        cache.pop(query, None)
        return None
    
    if (datetime.now() - ts).days > 7:
        cache.pop(query, None)
        return None

    return entry.get("answer")

def save_cached(query: str, answer: str) -> None:
    """Safely save answer to cache with validation."""
    # Prevent cache from growing unbounded
    if len(cache) > 10000:
        # Remove oldest entries
        oldest_keys = sorted(
            cache.keys(),
            key=lambda k: cache[k].get("timestamp", "")
        )[:1000]
        for key in oldest_keys:
            cache.pop(key, None)
    
    cache[query] = {
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except (IOError, OSError) as e:
        print(f"⚠️ Warning: Could not save cache: {e}")

# =========================
# MAIN LOOP
# =========================

def main():
    print("\nLegalAID AI Assistant (type 'q' to quit)")
    print("=" * 60)

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue

        cached = get_cached(q)
        if cached:
            print("\n(Cached)")
            print(cached)
            continue

        start = time.time()

        try:
            answer, docs = answer_query(q, model)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

        save_cached(q, answer)

        elapsed = time.time() - start
        print(f"\n✅ Answer (took {elapsed:.2f}s)")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        if docs:
            print("\nSources:")
            for i, d in enumerate(docs, 1):
                print(f"{i}. {d.metadata}")

# =========================

if __name__ == "__main__":
    main()
