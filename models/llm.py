"""LLM — Groq, OpenAI, Gemini."""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_groq_client(api_key: str):
    try:
        from groq import Groq
        if not api_key:
            logger.warning("GROQ_API_KEY is empty.")
            return None
        return Groq(api_key=api_key)
    except Exception as exc:
        logger.error("Failed to initialise Groq client: %s", exc)
        return None


def groq_chat(client, model: str, messages: list[dict], system_prompt: str) -> tuple[str, str | None]:
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        completion = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return completion.choices[0].message.content, None
    except Exception as exc:
        logger.error("Groq chat error: %s", exc)
        return "", str(exc)


def get_openai_client(api_key: str):
    try:
        from openai import OpenAI
        if not api_key:
            logger.warning("OPENAI_API_KEY is empty.")
            return None
        return OpenAI(api_key=api_key)
    except Exception as exc:
        logger.error("Failed to initialise OpenAI client: %s", exc)
        return None


def openai_chat(client, model: str, messages: list[dict], system_prompt: str) -> tuple[str, str | None]:
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        completion = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return completion.choices[0].message.content, None
    except Exception as exc:
        logger.error("OpenAI chat error: %s", exc)
        return "", str(exc)


def get_gemini_model(api_key: str, model_name: str):
    try:
        import google.generativeai as genai
        if not api_key:
            logger.warning("GEMINI_API_KEY is empty.")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        logger.error("Failed to initialise Gemini model: %s", exc)
        return None


def gemini_chat(model, messages: list[dict], system_prompt: str) -> tuple[str, str | None]:
    try:
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

        chat = model.start_chat(history=history)
        user_text = f"{system_prompt}\n\n{messages[-1]['content']}" if not history else messages[-1]["content"]
        response = chat.send_message(user_text)
        return response.text, None
    except Exception as exc:
        logger.error("Gemini chat error: %s", exc)
        return "", str(exc)


def get_llm_client(provider: str, api_key: str, model_name: str) -> tuple[Optional[object], str]:
    if provider == "Groq":
        return get_groq_client(api_key), model_name
    if provider == "OpenAI":
        return get_openai_client(api_key), model_name
    if provider == "Gemini":
        model = get_gemini_model(api_key, model_name)
        return model, model_name
    return None, model_name


def get_chat_response(
    provider: str,
    client_or_model,
    model_name: str,
    messages: list[dict],
    system_prompt: str,
) -> tuple[str, str | None]:
    if provider == "Groq":
        return groq_chat(client_or_model, model_name, messages, system_prompt)
    elif provider == "OpenAI":
        return openai_chat(client_or_model, model_name, messages, system_prompt)
    elif provider == "Gemini":
        return gemini_chat(client_or_model, messages, system_prompt)
    else:
        return "", f"Unknown provider: {provider}"
