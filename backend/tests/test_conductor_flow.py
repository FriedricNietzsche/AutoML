import asyncio
import os
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import app  # noqa: E402
from app.orchestrator.conductor import conductor  # noqa: E402
from app.events.schema import StageID, StageStatus  # noqa: E402


client = TestClient(app)


class TestConductorFlow(unittest.TestCase):
    def test_initial_state_and_confirm_flow(self):
        project_id = "test-flow"

        resp = client.get(f"/api/projects/{project_id}/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["current_stage"]["id"], StageID.PARSE_INTENT.value)
        self.assertEqual(data["current_stage"]["status"], StageStatus.IN_PROGRESS.value)

        resp2 = client.post(f"/api/projects/{project_id}/confirm")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2["current_stage"]["id"], StageID.DATA_SOURCE.value)
        self.assertEqual(data2["current_stage"]["status"], StageStatus.IN_PROGRESS.value)

    def test_waiting_confirmation_path(self):
        project_id = "test-wait"
        conductor._projects.pop(project_id, None)  # reset for the test

        asyncio.run(
            conductor.waiting_for_confirmation(
                project_id, StageID.DATA_SOURCE, "Review dataset", ["Approve source"]
            )
        )
        state = asyncio.run(conductor.get_state(project_id))
        self.assertEqual(state["waiting_confirmation"]["stage_id"], StageID.DATA_SOURCE.value)
        stage_map = {s["id"]: s for s in state["stages"]}
        self.assertEqual(
            stage_map[StageID.DATA_SOURCE.value]["status"], StageStatus.WAITING_CONFIRMATION.value
        )


if __name__ == "__main__":
    unittest.main()
