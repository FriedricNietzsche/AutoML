import os
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

try:
    # OpenRouter is OpenAI-compatible; LangChain's ChatOpenAI can target it via base_url.
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None


class PromptParsedPayload(BaseModel):
    """Stage 1 contract payload for event `PROMPT_PARSED`.

    Contract:
      PROMPT_PARSED: {task_type, target, dataset_hint, constraints}
    """

    task_type: str = Field(
        ..., description="classification|regression|clustering|timeseries|nlp|vision|tabular|other"
    )
    target: str = Field(..., description="What to predict/classify (short phrase).")
    dataset_hint: str = Field(
        ..., description="Hint for dataset search: keywords, domain, dataset names, URLs, etc."
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form constraints: metric, time, compute, latency, privacy, licensing, etc.",
    )


class PromptParserAgent:
    """Parses a user prompt into the Stage 1 `PROMPT_PARSED` schema.

    Provider: Gemini 1.5 Flash through OpenRouter.
    - Good + cheap for extraction.
    - Uses structured output to avoid schema drift.

    Environment:
      - OPENROUTER_API_KEY must be set.

    Returns:
      A dict with exactly: {task_type, target, dataset_hint, constraints}
    """

    def __init__(
        self,
        *,
        model: str = None,
        temperature: float = 0.0,
        timeout_s: int = 30,
    ):
        if ChatOpenAI is None:
            raise RuntimeError(
                "Missing dependency 'langchain-openai'. Install it in backend env."
            )

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        # Default to a good/cheap Gemini 1.5 model on OpenRouter.
        self.model = model or "google/gemini-1.5-flash"

        base_llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=self.model,
            temperature=temperature,
            timeout=timeout_s,
        )

        # Force schema-shaped output.
        self.llm = base_llm.with_structured_output(PromptParsedPayload)

    def parse(self, prompt: str) -> Dict[str, Any]:
        """Parse the user prompt into PROMPT_PARSED payload."""
        prompt = (prompt or "").strip()

        # We still return a contract-valid payload for empty input.
        if not prompt:
            return PromptParsedPayload(
                task_type="other",
                target="",
                dataset_hint="",
                constraints={
                    "needs_clarification": True,
                    "questions": [
                        "What do you want the model to do (predict/classify what)?",
                        "What type of ML task is it (classification, regression, etc.)?",
                    ],
                },
            ).model_dump()

        system = (
            "You are a strict prompt parsing component for an AutoML platform.\n"
            "Return ONLY a JSON object matching this schema exactly:\n"
            "{task_type, target, dataset_hint, constraints}\n\n"
            "Rules:\n"
            "- task_type must be one of: classification|regression|clustering|timeseries|nlp|vision|tabular|other.\n"
            "- target must be a short, concrete phrase describing what to predict/classify.\n"
            "- dataset_hint must be short and useful for dataset search; include URLs if user provided them.\n"
            "- constraints must be an object; include only constraints explicitly mentioned by the user.\n"
            "- If anything critical is unclear, set constraints.needs_clarification=true and include constraints.questions=[...].\n"
            "- Do not invent specifics.\n"
        )

        try:
            payload = self.llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
            if not isinstance(payload, PromptParsedPayload):
                payload = PromptParsedPayload.model_validate(payload)

            # Normalize task_type defensively.
            payload.task_type = (payload.task_type or "other").strip().lower()
            allowed = {
                "classification",
                "regression",
                "clustering",
                "timeseries",
                "nlp",
                "vision",
                "tabular",
                "other",
            }
            if payload.task_type not in allowed:
                payload.task_type = "other"

            # If target is missing, force clarification (still in-schema).
            if not payload.target.strip():
                c = dict(payload.constraints or {})
                c["needs_clarification"] = True
                qs = c.get("questions") or []
                q = "What exactly should the model predict/classify?"
                if q not in qs:
                    qs.append(q)
                c["questions"] = qs
                payload.constraints = c

            return payload.model_dump()

        except ValidationError as ve:
            # LLM responded with schema-mismatched output; return a contract-valid clarification.
            return PromptParsedPayload(
                task_type="other",
                target="",
                dataset_hint="",
                constraints={
                    "needs_clarification": True,
                    "questions": [
                        "I couldn't parse your request reliably. Can you restate the goal in one sentence?"
                    ],
                    "validation_error": str(ve),
                },
            ).model_dump()

        except Exception as e:
            # Network/auth/timeouts.
            return PromptParsedPayload(
                task_type="other",
                target="",
                dataset_hint="",
                constraints={
                    "needs_clarification": True,
                    "questions": [
                        "Parsing service error. Try again or provide task_type + target explicitly."
                    ],
                    "error": str(e),
                },
            ).model_dump()