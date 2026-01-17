import json
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.llm.openrouter import OpenRouterClient
from app.llm.langchain_client import get_openrouter_llm


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
    """Parses a user prompt into the Stage 1 `PROMPT_PARSED` schema."""

    def __init__(self, *, temperature: float = 0.0, timeout_s: int = 30):
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.client = OpenRouterClient(timeout=timeout_s)
        self.llm = get_openrouter_llm(temperature=temperature, timeout=timeout_s)
        self.parser = JsonOutputParser()
        format_instructions = "{format_instructions}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict prompt parser for an AutoML platform. "
                    "Return ONLY a JSON object matching schema: {task_type, target, dataset_hint, constraints}. "
                    "Rules: task_type in [classification, regression, clustering, timeseries, nlp, vision, tabular, other]; "
                    "target is a short phrase; dataset_hint is short and useful for dataset search; "
                    "constraints is an object with user-stated constraints; "
                    "if unclear, set constraints.needs_clarification=true and add constraints.questions[]. "
                    f"Use this JSON format:\n{format_instructions}",
                ),
                ("user", "{user_prompt}"),
            ]
        )

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

        # Preferred path: LangChain structured output if LLM available.
        try:
            if self.llm:
                chain = self.prompt | self.llm | self.parser
                payload_obj = chain.invoke(
                    {
                        "user_prompt": prompt,
                        "format_instructions": self.parser.get_format_instructions(),
                    }
                )
                payload = PromptParsedPayload.model_validate(payload_obj)
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

        except Exception:
            # Fall through to legacy OpenRouter client + heuristics
            pass

        # Legacy LLM call (non-LangChain) if available
        try:
            if self.client.available():
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

                raw = self.client.chat_text(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                payload_obj = json.loads(raw)
                payload = PromptParsedPayload.model_validate(payload_obj)
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
        except (ValidationError, json.JSONDecodeError) as ve:
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
            # fallthrough to heuristics below
            pass

        # Heuristic fallback for common intents or errors
        lower = prompt.lower()
        if "cat" in lower and "dog" in lower:
            return PromptParsedPayload(
                task_type="classification",
                target="cat vs dog image",
                dataset_hint="cats and dogs images",
                constraints={},
            ).model_dump()

        return PromptParsedPayload(
            task_type="other",
            target="",
            dataset_hint="",
            constraints={
                "needs_clarification": True,
                "questions": [
                    "Parsing service error. Try again or provide task_type + target explicitly.",
                    "Ensure OPENROUTER_API_KEY is set and valid (no ellipsis placeholders).",
                ],
                "error": "LLM unavailable or parsing failed.",
            },
        ).model_dump()
