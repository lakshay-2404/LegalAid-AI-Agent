"""
embeddings.py — Embedding providers and cross-encoder for LegalAid RAG.

Supports two backends:
  1. Ollama (default)          — USE_ST_EMBED=0
  2. SentenceTransformers      — USE_ST_EMBED=1
"""
from __future__ import annotations

import logging
import os
import builtins
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_base_url() -> Optional[str]:
    return (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "").strip() or None


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip())
    except Exception:
        return default


def _resolve_torch_device(value: Optional[str], default: str = "auto") -> str:
    requested = (value or default).strip().lower()
    if requested in {"", "auto"}:
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if requested.startswith("cuda"):
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable; falling back to CPU.")
                return "cpu"
        except Exception:
            return "cpu"

    return requested


def _prepare_transformers_torch_backend() -> None:
    """
    Force torch-first backend selection before importing transformers/sentence-transformers.
    Also inject builtins.nn as a defensive workaround for rare upstream annotation bugs.
    """
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    try:
        import torch  # type: ignore
        if not hasattr(builtins, "nn"):
            builtins.nn = torch.nn
    except Exception:
        # Keep import path resilient; caller handles downstream failures.
        pass


# ---------------------------------------------------------------------------
# Embedding instructions
# ---------------------------------------------------------------------------

DOC_INSTRUCTION = os.environ.get(
    "EMBED_DOC_INSTRUCTION",
    "Represent this legal document for retrieval of relevant statutes and case law: ",
)
QUERY_INSTRUCTION = os.environ.get(
    "EMBED_QUERY_INSTRUCTION",
    "Represent this legal query for retrieving relevant statutes and legal precedents: ",
)

# ---------------------------------------------------------------------------
# Embedding batch/worker config (read once at module import)
# ---------------------------------------------------------------------------

BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "64"))       # raised from 32
INGEST_WORKERS = int(os.environ.get("INGEST_WORKERS", "4"))        # raised from 1

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

# Default to HuggingFace SentenceTransformers (no Ollama dependency).
# Set USE_ST_EMBED=0 in env to fall back to Ollama.
_use_st = os.environ.get("USE_ST_EMBED", "1").strip().lower() in {"1", "true", "yes", "on"}
_embed_dim_override: Optional[int] = None
_embed_provider = "ollama"
_st_device = _resolve_torch_device(os.environ.get("ST_DEVICE"), default="auto")
_cross_encoder_device = _resolve_torch_device(
    os.environ.get("CROSS_ENCODER_DEVICE") or os.environ.get("ST_DEVICE"),
    default="auto",
)
_st_init_error: Optional[Exception] = None

_embed_documents_fn = None  # (List[str]) -> List[List[float]]
_embed_query_fn = None       # (str) -> List[float]

# --- SentenceTransformers ---------------------------------------------------
if _use_st:
    try:
        _prepare_transformers_torch_backend()
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch

        st_model_name = os.environ.get("ST_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        st_batch_size = _env_int("ST_BATCH_SIZE", 32)

        try:
            _st_model = SentenceTransformer(st_model_name, device=_st_device)
        except Exception as _e:
            if _st_device != "cpu":
                try:
                    _st_model = SentenceTransformer(st_model_name, device="cpu")
                    print(f"CUDA unavailable ({_e}); falling back to CPU for {st_model_name}.")
                    _st_device = "cpu"
                except Exception as _e2:
                    raise RuntimeError(
                        f"Failed to init sentence-transformers {st_model_name} on {_st_device} and cpu: {_e}; {_e2}"
                    ) from _e2
            else:
                raise RuntimeError(f"Failed to init sentence-transformers {st_model_name} on CPU: {_e}") from _e

        _embed_dim_override = int(_st_model.get_sentence_embedding_dimension())

        def _st_embed_documents(texts: List[str]) -> List[List[float]]:
            prefixed = [DOC_INSTRUCTION + t for t in texts]
            try:
                return _st_model.encode(
                    prefixed, batch_size=st_batch_size, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                ).tolist()
            except torch.cuda.OutOfMemoryError:
                print("CUDA OOM on embeddings; retrying on CPU for this batch")
                torch.cuda.empty_cache()
                cpu_m = SentenceTransformer(st_model_name, device="cpu")
                return cpu_m.encode(
                    prefixed, batch_size=max(8, min(64, st_batch_size // 4)),
                    show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
                ).tolist()

        def _st_embed_query(text: str) -> List[float]:
            prefixed = [QUERY_INSTRUCTION + text]
            try:
                return _st_model.encode(
                    prefixed, batch_size=st_batch_size, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )[0].tolist()
            except torch.cuda.OutOfMemoryError:
                print("CUDA OOM on query embed; retrying on CPU")
                torch.cuda.empty_cache()
                cpu_m = SentenceTransformer(st_model_name, device="cpu")
                return cpu_m.encode(
                    prefixed, batch_size=1, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )[0].tolist()

        _embed_documents_fn = _st_embed_documents
        _embed_query_fn = _st_embed_query
        _embed_provider = "sentence-transformers"
        print(f"Using local sentence-transformers embeddings: {st_model_name} (batch={st_batch_size}, device={_st_device})")
    except Exception as e:
        _st_init_error = e
        _use_st = False
        logger.warning(
            "Sentence-transformers init failed (%s); falling back to Ollama embeddings.",
            e,
        )

# --- Ollama -----------------------------------------------------------------
if not _use_st:
    from langchain_ollama import OllamaEmbeddings  # type: ignore

    _ollama_model = OllamaEmbeddings(
        model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=_ollama_base_url(),
        num_ctx=_env_int("OLLAMA_EMBED_NUM_CTX", 2048),
        client_kwargs={"timeout": _env_int("OLLAMA_EMBED_TIMEOUT_S", 15)},
        sync_client_kwargs={"timeout": _env_int("OLLAMA_EMBED_TIMEOUT_S", 15)},
    )
    _embed_documents_fn = lambda texts: _ollama_model.embed_documents([DOC_INSTRUCTION + t for t in texts])
    _embed_query_fn = lambda text: _ollama_model.embed_query(QUERY_INSTRUCTION + text)
    if _st_init_error is not None:
        logger.warning("Ollama fallback is active because ST backend failed to load.")


# ---------------------------------------------------------------------------
# Public embedding API
# ---------------------------------------------------------------------------

def embed_query(text: str) -> List[float]:
    return _embed_query_fn(text)


def _chunk_list(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i: i + size] for i in range(0, len(seq), size)]


def embed_documents_parallel(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts, using threads for Ollama concurrent requests."""
    # SentenceTransformers already batches internally — no extra threads needed.
    if _embed_provider != "ollama" or INGEST_WORKERS <= 1 or len(texts) <= 1:
        return _embed_documents_fn(texts)
    batches = _chunk_list(texts, BATCH_SIZE)
    with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as ex:
        results = list(ex.map(_embed_documents_fn, batches))
    flat: List[List[float]] = []
    for r in results:
        flat.extend(r)
    return flat


def embed_documents_resilient(texts: List[str]) -> List[List[float]]:
    """Embed with exponential-halving retry on failure."""
    if not texts:
        return []
    try:
        return embed_documents_parallel(texts)
    except Exception as e:
        if len(texts) == 1:
            raise
        mid = max(1, len(texts) // 2)
        logger.warning("Embedding batch of %s texts failed; retrying halved: %s", len(texts), e)
        return embed_documents_resilient(texts[:mid]) + embed_documents_resilient(texts[mid:])


def get_embedding_dim_override() -> Optional[int]:
    return _embed_dim_override


# ---------------------------------------------------------------------------
# Cross-encoder (optional reranker)
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_BATCH_SIZE = _env_int("CROSS_ENCODER_BATCH_SIZE", 8)
_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder, _cross_encoder_device
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        _prepare_transformers_torch_backend()
        from sentence_transformers import CrossEncoder  # type: ignore
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=_cross_encoder_device)
        logger.info(
            "Using cross-encoder reranker: %s (device=%s)",
            CROSS_ENCODER_MODEL,
            _cross_encoder_device,
        )
    except Exception as e:
        if _cross_encoder_device != "cpu":
            try:
                _prepare_transformers_torch_backend()
                from sentence_transformers import CrossEncoder  # type: ignore
                _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
                _cross_encoder_device = "cpu"
                logger.warning(
                    "Cross-encoder init failed on GPU (%s); falling back to CPU.",
                    e,
                )
            except Exception as e2:
                logger.warning(
                    "Cross-encoder init failed on GPU and CPU (%s; %s); rerank disabled",
                    e,
                    e2,
                )
                _cross_encoder = None
        else:
            logger.warning("Cross-encoder init failed (%s); rerank disabled", e)
            _cross_encoder = None
    return _cross_encoder


def cross_encoder_score(query: str, docs) -> List[float]:
    if not docs:
        return []
    model = get_cross_encoder()
    if model is None:
        return [0.0] * len(docs)
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = model.predict(pairs, batch_size=CROSS_ENCODER_BATCH_SIZE, convert_to_numpy=True)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)
    except Exception as e:
        logger.warning("Cross-encoder scoring failed: %s", e)
        return [0.0] * len(docs)
