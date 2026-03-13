"""RAG utilities — load PDFs/TXTs, chunk, build an in-memory vector store, retrieve."""
from __future__ import annotations
import io
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error("TXT extraction failed: %s", exc)
        return ""


def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return extract_text_from_txt(file_bytes)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class SimpleVectorStore:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.vectors: list[list[float]] = []
        self.sources: list[str] = []

    def add_documents(self, chunks: list[str], vectors: list[list[float]], source: str) -> None:
        for chunk, vec in zip(chunks, vectors):
            self.chunks.append(chunk)
            self.vectors.append(vec)
            self.sources.append(source)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        if not self.vectors:
            return []
        scored = [
            {
                "text": self.chunks[i],
                "source": self.sources[i],
                "score": self._cosine_similarity(query_vector, self.vectors[i]),
            }
            for i in range(len(self.vectors))
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


def build_vector_store(
    uploaded_files: list[Any],
    embedding_model,
    chunk_size: int,
    overlap: int,
) -> SimpleVectorStore:
    store = SimpleVectorStore()
    for uf in uploaded_files:
        try:
            raw = uf.read()
            text = extract_text(uf.name, raw)
            if not text.strip():
                logger.warning("No text extracted from %s", uf.name)
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            from models.embeddings import embed_texts
            vectors = embed_texts(embedding_model, chunks)
            if vectors:
                store.add_documents(chunks, vectors, source=uf.name)
                logger.info("Indexed %d chunks from %s", len(chunks), uf.name)
        except Exception as exc:
            logger.error("Failed to process %s: %s", uf.name, exc)
    return store


# Broad/summarization intent keywords — always treated as "relevant" when docs exist
_BROAD_INTENTS = (
    "summarize", "summary", "summarise", "overview", "what does",
    "what is in", "what does this cover", "what topics", "outline",
    "brief", "tldr", "tl;dr", "give me a summary", "describe the document",
    "what does the pdf", "what does the document",
)


def retrieve_context(
    store: SimpleVectorStore,
    query: str,
    embedding_model,
    top_k: int = 4,
    relevance_threshold: float = 0.25,
) -> tuple[list[dict], bool, float]:
    """Retrieve relevant chunks for a query. Returns (contexts, has_relevant, max_score)."""
    try:
        from models.embeddings import embed_query
        query_lower = query.lower()
        is_broad = any(kw in query_lower for kw in _BROAD_INTENTS)

        query_vec = embed_query(embedding_model, query)
        if not query_vec:
            return [], False, 0.0

        effective_top_k = min(top_k * 3, len(store.chunks)) if is_broad else top_k
        results = store.search(query_vec, top_k=effective_top_k)

        if not results:
            return [], False, 0.0

        max_score = max(r["score"] for r in results)
        if is_broad and results:
            has_relevant = True
        else:
            has_relevant = any(r["score"] >= relevance_threshold for r in results)

        return results, has_relevant, max_score
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        return [], False, 0.0


def format_context_for_prompt(contexts: list[dict]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    if not contexts:
        return ""
    parts = []
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] (from {ctx['source']})\n{ctx['text']}")
    return "\n\n".join(parts)
