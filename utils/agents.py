"""Agent layer — RAG, web search, and LLM orchestration."""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    thought: str
    tool_used: str
    tool_output: str
    final_answer: str = ""


class HRAgent:

    def __init__(
        self,
        vector_store,
        embedding_model,
        llm_provider: str,
        llm_client,
        llm_model: str,
        web_search_enabled: bool,
        top_k: int = 4,
        rag_confidence_threshold: float = 0.25,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.web_search_enabled = web_search_enabled
        self.top_k = top_k
        self.rag_confidence_threshold = rag_confidence_threshold
        self.steps: list[AgentStep] = []

    def _classify_intent(self, query: str) -> str:
        classifier_prompt = (
            "Classify the following user message into exactly one category:\n"
            "- conversational: greetings, small talk, personal statements, "
            "follow-up questions that rely on chat history (e.g. 'hi', 'my name is X', 'what is my name')\n"
            "- document: questions that should be answered from uploaded HR policy documents\n"
            "- web: questions needing real-time or external information (laws, current events, companies)\n\n"
            "Reply with ONLY one word: conversational, document, or web.\n\n"
            f"Message: {query}"
        )
        messages = [{"role": "user", "content": classifier_prompt}]
        response = self._run_llm(
            messages,
            system_prompt="You are an intent classifier. Reply with one word only.",
        )
        intent = response.strip().lower()
        if intent not in ("conversational", "document", "web"):
            return "document"
        return intent

    def _run_rag(self, query: str) -> tuple[list[dict], bool, float, str]:
        try:
            from utils.rag import retrieve_context, format_context_for_prompt
            if self.vector_store is None or self.vector_store.is_empty():
                return [], False, 0.0, ""
            contexts, has_relevant, max_score = retrieve_context(
                self.vector_store,
                query,
                self.embedding_model,
                top_k=self.top_k,
                relevance_threshold=self.rag_confidence_threshold,
            )
            formatted = format_context_for_prompt(contexts) if contexts else ""
            return contexts, has_relevant, max_score, formatted
        except Exception as exc:
            logger.error("Agent RAG error: %s", exc)
            return [], False, 0.0, ""

    def _run_web(self, query: str) -> tuple[list[dict], str]:
        try:
            from utils.web_search import web_search, format_web_results_for_prompt
            results = web_search(query, api_key="", max_results=5)
            formatted = format_web_results_for_prompt(results) if results else ""
            return results, formatted
        except Exception as exc:
            logger.error("Agent web search error: %s", exc)
            return [], ""

    def _run_llm(self, messages: list[dict], system_prompt: str) -> str:
        try:
            from models.llm import get_chat_response
            response, err = get_chat_response(
                self.llm_provider,
                self.llm_client,
                self.llm_model,
                messages,
                system_prompt,
            )
            if err:
                logger.error("Agent LLM error: %s", err)
                return f"⚠️ LLM error: {err}"
            return response
        except Exception as exc:
            logger.error("Agent LLM call failed: %s", exc)
            return f"⚠️ Unexpected error: {exc}"

    def run(
        self,
        user_query: str,
        chat_history: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[AgentStep], list[dict], list[dict]]:
        self.steps = []
        doc_contexts: list[dict] = []
        web_results: list[dict] = []

        intent = self._classify_intent(user_query)

        if intent == "conversational":
            messages = chat_history + [{"role": "user", "content": user_query}]
            answer = self._run_llm(messages, system_prompt)
            self.steps.append(AgentStep(
                thought="Classified as conversational — skipping tools and answering directly.",
                tool_used="LLM only",
                tool_output="Answered from chat history and general knowledge.",
                final_answer=answer,
            ))
            return answer, self.steps, [], []

        rag_context_str = ""
        has_relevant = False
        max_score = 0.0

        if self.vector_store and not self.vector_store.is_empty():
            doc_contexts, has_relevant, max_score, rag_context_str = self._run_rag(user_query)
            self.steps.append(AgentStep(
                thought="Searching uploaded HR policy documents for relevant sections.",
                tool_used="RAG retrieval",
                tool_output=f"Found {len(doc_contexts)} chunks. Relevant: {has_relevant}, max_score: {max_score:.3f}.",
            ))
        else:
            self.steps.append(AgentStep(
                thought="No documents uploaded yet — skipping RAG.",
                tool_used="RAG retrieval",
                tool_output="Skipped (no documents).",
            ))

        if self.vector_store and not self.vector_store.is_empty() and not has_relevant:
            rag_context_str = (
                "No clearly relevant sections were found in the uploaded HR policy "
                "documents for this question."
            )

        from utils.web_search import needs_web_search
        web_context_str = ""

        force_web = intent == "web"
        low_rag_confidence = max_score > 0 and max_score < self.rag_confidence_threshold
        do_web = (
            self.web_search_enabled
            and (force_web or needs_web_search(user_query, has_relevant) or low_rag_confidence)
        )

        if do_web:
            reason = (
                "Query classified as needing web." if force_web else
                "RAG returned no relevant chunks — fallback to web." if not has_relevant else
                f"RAG confidence low (max_score={max_score:.2f} < {self.rag_confidence_threshold}). "
                "Running web search for supplementary information."
            )
            self.steps.append(AgentStep(
                thought=reason,
                tool_used="Web search",
                tool_output="",
            ))
            web_results, web_context_str = self._run_web(user_query)
            self.steps[-1].tool_output = (
                f"Retrieved {len(web_results)} web results."
                if web_results else "No results returned."
            )

        context_parts: list[str] = []
        if rag_context_str:
            context_parts.append("=== HR Policy Documents ===\n" + rag_context_str)
        if web_context_str:
            context_parts.append("=== Web Search Results ===\n" + web_context_str)

        combined_context = "\n\n".join(context_parts)

        from utils.prompt import build_user_prompt
        used_web = bool(web_context_str)
        final_user_message = build_user_prompt(user_query, combined_context, used_web)

        messages = chat_history + [{"role": "user", "content": final_user_message}]

        self.steps.append(AgentStep(
            thought="Synthesising retrieved context and chat history into a final answer.",
            tool_used="LLM",
            tool_output="",
        ))

        final_answer = self._run_llm(messages, system_prompt)
        self.steps[-1].tool_output = "Answer generated."
        self.steps[-1].final_answer = final_answer

        return final_answer, self.steps, doc_contexts, web_results
