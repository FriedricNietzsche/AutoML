"""
Small helper for calling OpenRouter's chat completions API.
Uses the OpenAI-compatible surface at https://openrouter.ai/api/v1/chat/completions.
Falls back to a no-op if the API key is missing.
"""
from __future__ import annotations

import os
import requests
from typing import List, Dict, Any

OPENROUTER_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        # Default to a widely available model; override via OPENROUTER_MODEL if needed.
        self.model = model or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def _validated_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        # HTTP headers must be latin-1 encodable; detect placeholders like "sk-or-…".
        try:
            self.api_key.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "OPENROUTER_API_KEY contains non-ASCII characters (did you paste an ellipsis … placeholder?). "
                "Use the full key (e.g., sk-or-xxxxxxxx)."
            ) from exc
        return self.api_key

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, Any]:
        key = self._validated_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"OpenRouter HTTP error {resp.status_code}: {resp.text}") from exc
        data = resp.json()
        return data

    def chat_text(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        data = self.chat(messages, temperature=temperature)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc
