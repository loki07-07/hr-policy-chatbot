"""Web search utility — DuckDuckGo."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_WEB_KEYWORDS = (
    "law", "legal", "regulation", "act", "compliance",
    "government", "latest", "current", "2024", "2025",
    "india", "us", "uk", "eu", "fmla", "gdpr",
)

_DOC_ONLY_KEYWORDS = (
    "summarize", "summary", "summarise", "overview", "what does this",
    "what does the pdf", "what does the document", "in this document",
    "in the pdf", "according to the policy", "this policy",
)


def web_search(query: str, api_key: str = "", max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
        raw = DDGS().text(query, max_results=max_results)
        results: list[dict] = [
            {
                "title": r.get("title", "") or query,
                "link": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in (raw or [])
        ]

        if not results:
            fallback_link = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            results.append(
                {
                    "title": f"Web search results for: {query}",
                    "link": fallback_link,
                    "snippet": "Open this search results page in your browser for more details.",
                }
            )

        logger.info("DuckDuckGo search returned %d results for: %s", len(results), query)
        return results[:max_results]
    except Exception as exc:
        logger.error("DuckDuckGo search error: %s", exc)
        return []


def format_web_results_for_prompt(results: list[dict]) -> str:
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['link']}")
    return "\n\n".join(parts)


def needs_web_search(query: str, has_relevant_docs: bool) -> bool:
    query_lower = query.lower()

    if any(kw in query_lower for kw in _DOC_ONLY_KEYWORDS):
        return False

    if any(kw in query_lower for kw in _WEB_KEYWORDS):
        return True

    return not has_relevant_docs
