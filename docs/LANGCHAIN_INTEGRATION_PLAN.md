# LangChain Integration Plan (Backend Agents)

## Goal
Replace the ad-hoc OpenRouter calls in our agents with LangChain chains while keeping deterministic execution (profiling, preprocessing, training, notebook/export) unchanged. Preserve the existing 8-stage contract and WS events.

## Scope (agents to LangChain)
- **ParseAgent**: structured output (task_type, target, dataset_hint, constraints). Use LC parser + retry for schema compliance; keep heuristic fallback.
- **ModelSelectorAgent**: LC chain ranks candidates from our allowed set (rf/xgb/logreg/linear/auto). Optional constraints handling (latency, interpretability).
- **Preprocess Planner (optional)**: LC chain that consumes schema summary and emits a preprocessing plan; execution stays deterministic sklearn pipelines.
- **Reporter (text only)**: LC chain for concise report prose. Notebook remains deterministic Python template.
- **(Out of scope for now)**: Training agents stay deterministic (sklearn/TF); no LC here.

## Implementation Steps
1) **Dependencies & clients**
   - Add `langchain`, `langchain-core`, `langchain-openai` (for OpenRouter-compatible client) or direct `ChatOpenAI` with base_url pointing to OpenRouter.
   - Wrap OpenRouter in an LC LLM instance; keep existing key env (`OPENROUTER_API_KEY`).
2) **ParseAgent**
   - Introduce LC `ChatPromptTemplate` + `StructuredOutputParser` (or PydanticOutputParser) for the PROMPT_PARSED schema.
   - Add retry/error-handling (LC `with_structured_output` or `RetryWithErrorOutputParser`).
   - Fall back to heuristic for empty/LLM failures.
3) **ModelSelectorAgent**
   - Prompt template consuming task_type/target/constraints; output ranked list from allowed models.
   - Structured parser to enforce allowed values; fallback to deterministic default when LLM unavailable.
4) **Preprocess Planner (optional)**
   - Prompt with schema summary (dtypes, missingness, target info); output JSON plan (imputation, encoding, scaling).
   - Execution still uses current sklearn ColumnTransformer; treat plan as hints (non-breaking if absent).
5) **Reporter (text)**
   - Prompt for ≤200-word summary using LC chain; keep deterministic notebook generation.
6) **Config & wiring**
   - Add settings for model name (OpenRouter model id), temperature, timeout.
   - Pass LC chains into agents via dependency injection so tests can stub with fake LLM.

## Testing Plan
- **Unit**: stub LLM returning fixed JSON; assert parsers coerce invalid outputs; ensure fallbacks work when key missing.
- **Integration**: run existing e2e tabular flow with OPENROUTER key to verify stages/events unchanged and notebook still emitted.
- **Error paths**: simulate bad LLM output and network errors; expect deterministic fallback payloads and no crashes.
- **Contract**: validate PROMPT_PARSED and MODEL_CANDIDATES still conform to FRONTEND_BACKEND_CONTRACT.

## Credentials / Env
- Required: `OPENROUTER_API_KEY` (already available). If using LangSmith tracing, need `LANGSMITH_API_KEY` (optional).
- No new Kaggle requirements (ingestion unchanged).

## Open Questions
- Preferred OpenRouter model for LC (e.g., `openrouter/anthropic/claude-3.5-sonnet`)? Default can remain current.
- Do we want LC tracing (LangSmith) in production, or disable by default?
- Do we enable the optional preprocess planner now or later?

## Definition of Done
- ParseAgent, ModelSelectorAgent, Reporter (text) refactored to LangChain with structured outputs and fallbacks.
- Existing tests green; add LLM-stubbed unit tests for each agent.
- E2E tabular flow still passes; WS events and stage transitions unchanged.
