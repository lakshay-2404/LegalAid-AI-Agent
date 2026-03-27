"""
Cross-encoder reranker helper.

Uses a HuggingFace sentence-transformers style cross-encoder if available.
Falls back to a dummy scorer that returns zeros if the model or deps are missing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document


@lru_cache(maxsize=1)
def _load_model(model_name: str = "BAAI/bge-reranker-large"):
    try:
        from sentence_transformers import CrossEncoder
        
        return CrossEncoder(model_name)
    except Exception:
        return None


def cross_encoder_rerank(query: str, docs: List[Document], model_name: str = "BAAI/bge-reranker-large") -> List[float]:
    model = _load_model(model_name)
    if model is None:
        return [0.0] * len(docs)
    pairs = [[query, d.page_content] for d in docs]
    try:
        scores = model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)
    except Exception:
        return [0.0] * len(docs)
