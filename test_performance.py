from __future__ import annotations

import os
import time
from typing import Optional

from langchain_ollama.llms import OllamaLLM

import vector


DEFAULT_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "qwen2.5:3b")
TEST_QUERIES = [
    "What are the legal rights of tenants?",
    "How do I file a consumer complaint in India?",
    "What is the process for divorce in India?",
]


def test_retriever(query: str, k: int = 5) -> Optional[float]:
    started_at = time.time()
    try:
        docs = vector.hybrid_retrieve(query, k=k)
    except Exception as exc:
        print(f"Retriever check failed: {exc}")
        return None

    duration = time.time() - started_at
    print(f"\nRetriever check (top {k})")
    print(f"Time: {duration:.2f}s")
    print(f"Documents: {len(docs)}")
    for index, doc in enumerate(docs[:3], start=1):
        metadata = doc.metadata or {}
        print(f"\nDocument {index}")
        if metadata.get("act"):
            print(f"  Act: {metadata['act']}")
        if metadata.get("section"):
            print(f"  Section: {metadata['section']}")
        preview = doc.page_content[:200]
        suffix = "..." if len(doc.page_content) > 200 else ""
        print(f"  Text: {preview}{suffix}")
    return duration


def test_model_response(prompt: str, model_name: str) -> Optional[float]:
    model = OllamaLLM(model=model_name)
    started_at = time.time()
    try:
        response = model.invoke(prompt)
    except Exception as exc:
        print(f"Model check failed: {exc}")
        return None

    duration = time.time() - started_at
    preview = response[:500]
    suffix = "..." if len(response) > 500 else ""
    print(f"\nModel check ({model_name})")
    print(f"Time: {duration:.2f}s")
    print(f"Response: {preview}{suffix}")
    return duration


def main() -> None:
    print("Starting performance checks")
    print(f"Using model: {DEFAULT_MODEL}")

    for index, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n{'=' * 50}")
        print(f"Test {index}: {query}")
        print("=" * 50)

        retrieval_time = test_retriever(query)
        healthcheck_time = test_model_response("Say hello in one sentence.", DEFAULT_MODEL)
        answer_time = test_model_response(f"Answer this legal question concisely: {query}", DEFAULT_MODEL)

        if None in {retrieval_time, healthcheck_time, answer_time}:
            continue

        total_time = retrieval_time + answer_time
        print(f"\nSummary for: {query}")
        print(f"- Retrieval: {retrieval_time:.2f}s")
        print(f"- Model healthcheck: {healthcheck_time:.2f}s")
        print(f"- End-to-end answer prompt: {answer_time:.2f}s")
        print(f"- Retrieval + answer total: {total_time:.2f}s")


if __name__ == "__main__":
    main()
