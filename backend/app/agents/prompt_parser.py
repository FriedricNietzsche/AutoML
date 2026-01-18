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
        import os
        self.temperature = temperature
        self.timeout_s = timeout_s
        
        print("[PromptParser] Initializing LLM clients...")
        
        # Debug: Check environment variable
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            print(f"[PromptParser] OPENROUTER_API_KEY found: {api_key[:15]}...{api_key[-10:]} (length: {len(api_key)})")
        else:
            print("[PromptParser] ⚠️  OPENROUTER_API_KEY is None or empty!")
        
        # Initialize OpenRouter client
        self.client = OpenRouterClient(timeout=timeout_s)
        print(f"[PromptParser] OpenRouter client available: {self.client.available()}")
        
        # Initialize LangChain LLM
        self.llm = get_openrouter_llm(temperature=temperature, timeout=timeout_s)
        print(f"[PromptParser] LangChain LLM available: {self.llm is not None}")
        
        if not self.client.available() and not self.llm:
            print("[PromptParser] ⚠️  WARNING: No LLM available! Check OPENROUTER_API_KEY in .env")
        
        self.parser = JsonOutputParser(pydantic_object=PromptParsedPayload)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict prompt parser for an AutoML platform. "
                    "Return ONLY a JSON object matching schema: {{task_type, target, dataset_hint, constraints}}. "
                    "Rules: task_type in [classification, regression, clustering, timeseries, nlp, vision, tabular, other]; "
                    "target is a short phrase; dataset_hint is short and useful for dataset search; "
                    "constraints is an object with user-stated constraints; "
                    "if unclear, set constraints.needs_clarification=true and add constraints.questions[]. "
                    "Use this JSON format:\n{format_instructions}",
                ),
                ("user", "{user_prompt}"),
            ]
        )

    def parse(self, prompt: str) -> Dict[str, Any]:
        """Parse the user prompt into PROMPT_PARSED payload."""
        import time
        start_time = time.time()
        
        print(f"\n[PromptParser] Starting parse...")
        print(f"[PromptParser] Prompt length: {len(prompt)} chars")
        
        prompt = (prompt or "").strip()

        # We still return a contract-valid payload for empty input.
        if not prompt:
            print("[PromptParser] Empty prompt detected, returning clarification request")
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
            print(f"[PromptParser] Checking if LangChain LLM is available: {self.llm is not None}")
            if self.llm:
                print("[PromptParser] Using LangChain LLM...")
                chain = self.prompt | self.llm | self.parser
                
                print("[PromptParser] Invoking LLM chain...")
                llm_start = time.time()
                payload_obj = chain.invoke(
                    {
                        "user_prompt": prompt,
                        "format_instructions": self.parser.get_format_instructions(),
                    }
                )
                llm_duration = time.time() - llm_start
                print(f"[PromptParser] ✅ LLM response received in {llm_duration:.2f}s")
                
                print("[PromptParser] Validating response...")
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

                total_duration = time.time() - start_time
                print(f"[PromptParser] ✅ Complete in {total_duration:.2f}s (LangChain)")
                return payload.model_dump()

        except Exception as e:
            print(f"[PromptParser] ⚠️  LangChain failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to legacy OpenRouter client + heuristics
            pass

        # Legacy LLM call (non-LangChain) if available
        try:
            print(f"[PromptParser] Checking if legacy OpenRouter client is available: {self.client.available()}")
            if self.client.available():
                print("[PromptParser] Using legacy OpenRouter client...")
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
            # Don't hide errors - let them bubble up so we know the LLM failed
            print(f"[PromptParser] ❌ Legacy client also failed: {e}")
            raise RuntimeError(f"PromptParser failed - LLM unavailable or errored: {e}")

        # If we get here, both LangChain and legacy client failed
        raise RuntimeError("PromptParser failed - no LLM available")
