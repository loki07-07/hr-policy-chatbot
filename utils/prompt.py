"""Prompt utilities for system and user prompts."""
from __future__ import annotations


ROLE_INSTRUCTIONS = {
    "Employee": (
        "The user is an Employee. Address them as an employee: use phrases like "
        "'you are entitled to', 'you can', 'your leave balance'. Focus on their rights and how to request things."
    ),
    "Manager": (
        "The user is a Manager. Address them as someone who manages a team: use phrases like "
        "'you should', 'your team', 'you may approve'. Include what they need to do or check when handling requests."
    ),
    "HR": (
        "The user is in HR. Address them in an administrative perspective: use phrases like "
        "'policy states', 'the process is', 'compliance requires'. You can be more formal and policy-precise."
    ),
}


def build_system_prompt(response_mode: str, role: str = "Employee") -> str:
    role = role if role in ROLE_INSTRUCTIONS else "Employee"
    role_para = ROLE_INSTRUCTIONS[role]

    base = (
        "You are an HR Policy Assistant for a company.\n"
        "You have access to text excerpts from uploaded HR policy documents "
        "and web search results that are provided in the conversation context.\n"
        "Always ground your answers in those sources when possible.\n\n"
        f"Role-aware rule:\n{role_para}\n\n"
        "Important rules:\n"
        "- If the question is covered by the HR policy documents, prefer those over the web.\n"
        "- If the question is about legal compliance or external law, you may also use web search.\n"
        "- When explaining policies, be clear and neutral.\n"
        "- If the answer is not stated in the provided context, say that it is not stated, "
        "then give a careful, clearly-marked best guess.\n"
        "- For eligibility questions, explain the reasoning step by step in plain language.\n"
        "- Never claim to see documents or systems that are not represented in the context.\n\n"
        "Summarization rule:\n"
        "- If the user asks to summarize or describe the document and context chunks are provided, "
        "ALWAYS produce a helpful summary from those chunks. Do NOT say there is no information. "
        "The chunks ARE the document content — synthesize them into a coherent overview.\n"
    )

    if response_mode == "Concise":
        return (
            base
            + "\nResponse length — CONCISE mode: You MUST keep your answer to 2–4 short sentences. "
              "State only the direct answer and the single most relevant policy point. "
              "Do not add background, caveats, or examples. If the user needs more, they can ask or switch to Detailed mode."
        )

    # Detailed mode
    return (
        base
        + "\nResponse length — DETAILED mode: Give a full, well-structured answer. "
          "Use bullet points or short numbered sections. Include: (1) the direct answer, "
          "(2) the relevant policy reasoning, (3) edge cases or conditions if any, and "
          "(4) a practical example or next step when helpful. Be thorough but clear."
    )


def build_user_prompt(user_query: str, context_str: str, used_web: bool) -> str:
    if not context_str:
        return user_query

    if used_web and context_str.startswith("=== HR Policy"):
        source_note = "Context from HR policy documents and web search results:"
    elif used_web:
        source_note = "Context from web search results:"
    else:
        source_note = "Context from uploaded HR policy documents:"

    return (
        f"{source_note}\n\n"
        f"{context_str}\n\n"
        "---\n\n"
        f"User question: {user_query}"
    )
