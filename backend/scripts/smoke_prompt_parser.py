"""Smoke test for PromptParserAgent (Gemini direct).

Run from `backend/` so `app` is importable:

    python3 scripts/smoke_prompt_parser.py

Pre-req:
- Set GEMINI_API_KEY in your environment or in `backend/.env`.

Notes:
- This script makes a real LLM call.
"""

import os

from dotenv import load_dotenv

# Load backend/.env if present (safe no-op if missing)
# `override=True` ensures edits to .env are applied even if the variable is already set.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

from app.agents.prompt_parser import PromptParserAgent  # noqa: E402


def main() -> None:
    print("GEMINI_API_KEY loaded:", bool(os.getenv("GEMINI_API_KEY")))
    agent = PromptParserAgent()

    prompt = (
        "Build a model to predict customer churn from tabular CRM data. "
        "Optimize for F1. Training should finish under 10 minutes."
    )

    parsed = agent.parse(prompt)
    print("PROMPT_PARSED payload:")
    print(parsed)


if __name__ == "__main__":
    main()
