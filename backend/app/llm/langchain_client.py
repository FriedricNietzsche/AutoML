"""
LangChain helpers for OpenRouter-backed LLMs.
We keep this thin so agents can choose LC when a key is present, and fallback when not.
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def _validated_key() -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        # Likely an ellipsis placeholder pasted in; treat as unavailable so fallbacks kick in.
        return None
    return key


def get_openrouter_llm(model: Optional[str] = None, temperature: float = 0.0, timeout: int = 20) -> Optional[ChatOpenAI]:
    """
    Returns a ChatOpenAI configured for OpenRouter, or None if no valid API key is set.
    """
    key = _validated_key()
    if not key:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = model or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"
    try:
        return ChatOpenAI(api_key=key, model=model, base_url=base_url, temperature=temperature, timeout=timeout)
    except Exception:
        return None
