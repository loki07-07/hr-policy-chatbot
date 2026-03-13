"""Export conversation to TXT for HR / tickets."""
from __future__ import annotations
from typing import List


def export_conversation_txt(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "unknown").capitalize()
        content = m.get("content", "").strip()
        lines.append(f"{role}:\n{content}\n")
        if m.get("sources"):
            lines.append(f"  [Sources]\n  {m['sources']}\n")
    return "\n".join(lines).strip()
