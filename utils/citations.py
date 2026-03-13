"""Citation utilities for formatting RAG + web sources shown to the user."""
from __future__ import annotations
from typing import List, Dict


def summarize_doc_sources(contexts: List[Dict]) -> str:
    """Build a short summary of which documents were used."""
    contexts = [c for c in contexts if c.get("score", 0) >= 0.20]
    if not contexts:
        return ""

    unique_sources: list[str] = []
    seen: set[str] = set()
    for c in contexts:
        src = (c.get("source") or "unknown").strip()
        if src and src not in seen:
            seen.add(src)
            unique_sources.append(src)

    if not unique_sources:
        return ""

    bullet_list = "\n".join(f"- `{s}`" for s in unique_sources)
    return "**Sources (HR policy documents):**\n" + bullet_list


def summarize_web_sources(results: List[Dict]) -> str:
    """Build a short summary of web links used."""
    if not results:
        return ""

    lines: list[str] = []
    for r in results[:5]:
        title = r.get("title") or "Result"
        link = r.get("link") or ""
        if link:
            lines.append(f"- [{title}]({link})")

    if not lines:
        return ""

    return "**External information (web search):**\n" + "\n".join(lines)


def build_sources_note(doc_contexts: List[Dict], web_results: List[Dict]) -> str:
    parts: list[str] = []
    doc_note = summarize_doc_sources(doc_contexts)
    web_note = summarize_web_sources(web_results)
    if doc_note:
        parts.append(doc_note)
    if web_note:
        parts.append(web_note)
    return "\n\n".join(parts)
