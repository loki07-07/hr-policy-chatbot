"""Configuration — API keys and app settings."""
from __future__ import annotations
import os

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

DEFAULT_LLM_PROVIDER: str = "Groq"
DEFAULT_GROQ_MODEL: str = "llama-3.1-8b-instant"
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL: str = "gemini-1.5-flash"

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K_RESULTS: int = 4
RAG_RELEVANCE_THRESHOLD: float = 0.25

APP_TITLE: str = "HR Policy Assistant"
APP_ICON: str = "🏢"
