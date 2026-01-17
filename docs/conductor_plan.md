# Conductor (Task 2.1) — Proposed Plan

## Responsibilities
- Owns per-project workflow state for the ordered stages: PARSE_INTENT → DATA_SOURCE → PROFILE_DATA → PREPROCESS → MODEL_SELECT → TRAIN → REVIEW_EDIT → EXPORT.
- Emits lifecycle events via the shared `event_bus` (primarily `STAGE_STATUS` + `WAITING_CONFIRMATION`).
- Exposes a lightweight API so the frontend can fetch a snapshot and advance stages with explicit confirmations.

## Data Model (in-memory first)
- `StageState`: `{id, status, message?, started_at?, completed_at?, plan_pending?}`.
- `ProjectState`: `{project_id, stages: Dict[StageID, StageState], current_stage: StageID, plan_pending: bool, plan_approved: bool}`.
- Store in an in-memory dict keyed by `project_id` for now; later replace with DB.

## Conductor API surface
- `get_state(project_id) -> ProjectState` (initializes default state if missing).
- `transition_to(project_id, stage_id, status=IN_PROGRESS, message=None)`:
  - Updates `current_stage`, resets downstream stages to `PENDING`.
  - Publishes `STAGE_STATUS` with `{stage_id, status, message}`.
- `waiting_for_confirmation(project_id, stage_id, summary, next_actions)`:
  - Marks stage `WAITING_CONFIRMATION`, stores the request, and emits `WAITING_CONFIRMATION`.
- `confirm(project_id)`:
  - Advances from `current_stage` to the next stage in the canonical order, setting previous to `COMPLETED` and next to `IN_PROGRESS`.
  - Emits `STAGE_STATUS` for both stages (prev completed + next in progress).

## HTTP Endpoints (FastAPI)
- `GET /api/projects/{id}/state` → returns `ProjectState` snapshot for hydration.
- `POST /api/projects/{id}/confirm` → advances the state machine via `confirm(project_id)`.
- Add simple request models for future plan approvals (`plan_id`, `approved: bool`) but keep in-memory for now.

## Event Hooks
- WebSocket connect should pull `ProjectState` from the conductor and send the current `STAGE_STATUS` instead of a hard-coded default.
- Any agent/runner that finishes work on a stage calls `transition_to(... COMPLETED)` and optionally `waiting_for_confirmation` to gate the next step.
- When a confirmation happens, emit `STAGE_STATUS` for the new active stage so the frontend timeline updates immediately.

## Error Handling & Defaults
- If an invalid stage is requested, raise 400 and keep state unchanged.
- When a project is first seen, initialize all stages to `PENDING` with `PARSE_INTENT` as `IN_PROGRESS` (or `PENDING` until prompt arrives, depending on UX decision).
- Keep log messages minimal; rely on `event_bus` to surface statuses to the UI.
