"""Embedding model — HuggingFace sentence-transformers."""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_embedding_model = None


def load_embedding_model(model_name: str):
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
        return _embedding_model
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return None


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    if model is None:
        return []
    try:
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()
    except Exception as exc:
        logger.error("Embedding error: %s", exc)
        return []


def embed_query(model, query: str) -> list[float]:
    results = embed_texts(model, [query])
    return results[0] if results else []
