"""Chat history utilities for managing Streamlit session state."""
from __future__ import annotations
from typing import TypedDict, Literal, List

Role = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str
    sources: str | None


def init_history(state) -> None:
    """Ensure chat history exists in session_state."""
    if "messages" not in state:
        state.messages: List[ChatMessage] = []


def add_message(
    state,
    role: Role,
    content: str,
    sources: str | None = None,
) -> None:
    """Append a message to session history."""
    init_history(state)
    state.messages.append(
        {
            "role": role,
            "content": content,
            "sources": sources,
        }
    )


def get_history_for_llm(state) -> list[dict]:
    init_history(state)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in state.messages
        if m["role"] in ("user", "assistant")
    ]


def clear_history(state) -> None:
    """Reset the conversation."""
    state.messages = []
