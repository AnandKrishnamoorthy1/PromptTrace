from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from prompt_trace import get_logs


def _load_example_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "ADK" / "adk_multi_agent_workflow.py"
    spec = importlib.util.spec_from_file_location("adk_multi_agent_workflow", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load ADK example module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AdkExampleTests(unittest.TestCase):
    def test_local_workflow_writes_nested_traces(self) -> None:
        module = _load_example_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "prompt_trace.db"
            result = module.run_local_workflow("build a multi-agent demo", db_path=db_path)

            self.assertIn("Plan for build a multi-agent demo", result.plan)
            self.assertIn("Recommendation:", result.draft)
            self.assertIn("Approved:", result.review)

            logs = sorted(get_logs(db_path), key=lambda row: row["id"])
            self.assertEqual(
                [row["step_name"] for row in logs],
                ["plan", "market_research", "risk_research", "research_sprint", "draft", "review"],
            )

            research_sprint_log = next(row for row in logs if row["step_name"] == "research_sprint")
            market_log = next(row for row in logs if row["step_name"] == "market_research")
            risk_log = next(row for row in logs if row["step_name"] == "risk_research")

            self.assertEqual(market_log["parent_trace_id"], research_sprint_log["trace_id"])
            self.assertEqual(risk_log["parent_trace_id"], research_sprint_log["trace_id"])


if __name__ == "__main__":
    unittest.main()