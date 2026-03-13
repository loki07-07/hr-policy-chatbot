# HR Policy Assistant

An AI-powered chatbot that answers employee and manager HR questions by combining **internal policy documents** (RAG) with **live web search** when needed. Built for the NeoStats case study.

---

## 1. Problem Definition & Use Case

### Problem

Employees and managers often struggle to find answers in long HR policy PDFs (leave, benefits, compliance, health checks). HR teams spend time repeating the same policy answers. A single, searchable interface that grounds answers in company policy and supplements with current external information (e.g. labour law) was missing.

### Objective

- Let users **ask in natural language** and get answers **grounded in uploaded HR documents**.
- **Fall back to live web search** when the question is about law, compliance, or facts not in the documents.
- Support **different roles** (Employee, Manager, HR) with appropriate tone and detail.
- Provide **citations** so users can verify answers against the source.

### Target Users

| User type   | Need |
|------------|------|
| **Employees** | “How do I apply for leave?”, “Who do I contact for a health check-up?” |
| **Managers**   | “What do I need to check when approving leave?”, “What does policy say about team benefits?” |
| **HR**         | Policy lookup, compliance checks, and **role management** (assign Employee/Manager/HR to users). |

### Why a Chatbot?

- **Information retrieval**: Fast, semantic search over many pages of PDFs.
- **Decision support**: Combines policy text with external rules (e.g. FMLA, local law) in one answer.
- **Consistency**: Same source of truth and phrasing for everyone.

---

## 2. Features

| Feature | Details |
|--------|---------|
| **RAG** | Upload PDF/TXT HR docs → chunked (configurable size/overlap), embedded (HuggingFace `all-MiniLM-L6-v2`), retrieved via cosine similarity |
| **Embeddings** | `models/embeddings.py` — local, no API key |
| **Web search** | DuckDuckGo (no API key). Used when intent is “web”, or when RAG has no/low-confidence results |
| **RAG–web fallback** | If retrieval returns no relevant chunks or best score is below a confidence threshold, web search is triggered (when enabled) |
| **Agent** | ReAct-style loop in `utils/agents.py`: classify intent → RAG → optional web → combine context → LLM answer |
| **Response modes** | **Concise**: 2–4 sentences. **Detailed**: bullets, reasoning, examples (prompts in `utils/prompt.py`) |
| **Citations** | “Sources & external information” expander; document chunks with score ≥ 0.20, plus web URLs |
| **Reasoning trace** | Expandable “Agent reasoning trace” in the UI |
| **Multi-LLM** | Groq (default), OpenAI, Gemini — abstraction in `models/llm.py` |
| **Auth & roles** | SQLite-backed login; roles (Employee / Manager / HR); only HR can change roles via HR Dashboard |

---

## 3. Architecture

### High-level flow

```
User query
    ↓
Intent classification (conversational | document | web)
    ↓
If conversational → LLM only
    ↓
RAG: embed query → vector search → top-k chunks (with relevance threshold)
    ↓
If no/low-confidence RAG results or web intent → DuckDuckGo web search
    ↓
Build prompt (system + role + response mode) with combined context
    ↓
LLM → Answer + citations + reasoning trace
```

### RAG pipeline

```
PDF/TXT upload → extract text → chunk_text(chunk_size, overlap) → embed_texts() → SimpleVectorStore
Query → embed_query() → store.search(query_vec, top_k) → score ≥ threshold → format_context_for_prompt()
```

### Project structure

```
hr-policy-chatbot/
├── config/
│   └── config.py          # API keys (env), chunk size, top_k, RAG confidence threshold
├── models/
│   ├── embeddings.py      # HuggingFace sentence-transformers load & embed
│   └── llm.py             # get_*_client(), get_chat_response() — Groq / OpenAI / Gemini
├── utils/
│   ├── rag.py             # PDF/TXT extraction, chunking, SimpleVectorStore, retrieve_context()
│   ├── web_search.py      # DuckDuckGo search, needs_web_search()
│   ├── agents.py          # HRAgent: intent, RAG, web, LLM
│   ├── prompt.py          # build_system_prompt(response_mode, role), build_user_prompt()
│   ├── chat_history.py    # Session message list, get_history_for_llm()
│   ├── citations.py       # build_sources_note(), summarize_doc_sources()
│   ├── db.py              # SQLite auth, list_users(), update_user_role()
│   └── export.py          # Export conversation as TXT
├── app.py                 # Streamlit UI, auth, chat, HR Dashboard
└── requirements.txt
```

---

## 4. Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set at least one LLM API key (Groq is default and free)
set GROQ_API_KEY=gsk_...        # Windows
export GROQ_API_KEY="gsk_..."   # Linux/macOS

# 3. Run
streamlit run app.py
```

Web search uses DuckDuckGo and does **not** require an API key. Optional: `OPENAI_API_KEY`, `GEMINI_API_KEY` for other providers.

---

## 5. Deployment (Streamlit Cloud)

### Steps

1. Push the repo to a **public GitHub** repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo, branch, and `app.py` as main file.
3. In **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_..."
# Optional:
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
```

4. **Deploy**. The app will build and run.

### Notes

- **Startup time**: First load can take 30–60 s while the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and dependencies download. Subsequent reruns are faster.
- **Memory**: Embedding model + Streamlit + LLM calls typically need ~1–2 GB. Free tier is usually sufficient for demo use.
- **Secrets**: Never commit API keys. Use Streamlit Cloud Secrets or environment variables only.
- **Requirements**: `requirements.txt` must list all dependencies (streamlit, groq, sentence-transformers, PyMuPDF, duckduckgo-search, etc.). Web search uses DuckDuckGo (no API key).

### Getting API keys

- **Groq** (free): [console.groq.com](https://console.groq.com)
- **OpenAI**: [platform.openai.com](https://platform.openai.com)
- **Gemini**: [aistudio.google.com](https://aistudio.google.com)

---

## 6. Configuration

| Setting | Purpose |
|--------|---------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | RAG chunking (default 500 / 50) |
| `TOP_K_RESULTS` | Number of chunks retrieved per query |
| `RAG_RELEVANCE_THRESHOLD` | Chunk is relevant if score ≥ this; below this, web fallback may run (when enabled) |
| `DEFAULT_GROQ_MODEL` | Default LLM (e.g. `llama-3.1-8b-instant`) |

All in `config/config.py`; API keys via environment variables.
