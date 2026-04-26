from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from prompt_trace import trace_prompt, trace_run
from prompt_trace.cli import main as cli_main
from prompt_trace.core import get_logs


class TracePromptTests(unittest.TestCase):
    def test_sync_decorator_writes_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="gpt-test", version_tag="v1", db_path=db_path)
            def call_llm(prompt: str, temperature: float = 0.2) -> str:
                return f"response: {prompt} @ {temperature}"

            result = call_llm("hello", temperature=0.7)

            self.assertEqual(result, "response: hello @ 0.7")

            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)

            log = logs[0]
            self.assertEqual(log["model"], "gpt-test")
            self.assertEqual(log["version_tag"], "v1")
            self.assertGreaterEqual(int(log["latency_ms"]), 0)

            prompt_payload = json.loads(log["prompt"])
            response_payload = json.loads(log["response"])

            self.assertEqual(prompt_payload, "hello")
            self.assertEqual(response_payload, "response: hello @ 0.7")

            with closing(sqlite3.connect(db_path)) as conn:
                row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'").fetchone()
                self.assertIsNotNone(row)

    def test_async_decorator_writes_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="async-model", version_tag="async", db_path=db_path)
            async def call_async_llm(prompt: str) -> str:
                return f"async: {prompt}"

            result = asyncio.run(call_async_llm("ping"))

            self.assertEqual(result, "async: ping")
            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["model"], "async-model")
            self.assertEqual(json.loads(logs[0]["prompt"]), "ping")

    def test_decorator_supports_custom_prompt_arg_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(
                model="custom-model",
                version_tag="custom",
                db_path=db_path,
                prompt_arg_name="input_text",
            )
            def call_llm(input_text: str, style: str = "short") -> str:
                return f"{style}: {input_text}"

            call_llm(input_text="hello keyword", style="long")

            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            self.assertEqual(json.loads(logs[0]["prompt"]), "hello keyword")

    def test_decorator_falls_back_to_full_payload_when_no_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="fallback-model", version_tag="fallback", db_path=db_path)
            def call_llm(*, temperature: float) -> str:
                return f"temperature={temperature}"

            call_llm(temperature=0.3)

            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            prompt_payload = json.loads(logs[0]["prompt"])
            self.assertEqual(prompt_payload["args"], [])
            self.assertEqual(prompt_payload["kwargs"], {"temperature": 0.3})

    def test_decorator_uses_provider_reported_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="provider-model", version_tag="provider", db_path=db_path)
            def call_llm(prompt: str) -> Dict[str, Any]:
                return {
                    "text": f"done: {prompt}",
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                }

            call_llm("token test")
            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["prompt_tokens"], 12)
            self.assertEqual(logs[0]["completion_tokens"], 5)
            self.assertEqual(logs[0]["total_tokens"], 17)

    def test_decorator_estimates_usage_when_provider_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="estimate-model", version_tag="estimate", db_path=db_path)
            def call_llm(prompt: str) -> str:
                return f"completion for {prompt}"

            call_llm("estimate this")
            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            self.assertGreater(logs[0]["prompt_tokens"], 0)
            self.assertGreater(logs[0]["completion_tokens"], 0)
            self.assertEqual(
                logs[0]["total_tokens"],
                logs[0]["prompt_tokens"] + logs[0]["completion_tokens"],
            )

    def test_nested_calls_track_parent_child_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="child-model", version_tag="nested", db_path=db_path, agent_name="worker", step_name="child")
            def child(prompt: str) -> str:
                return f"child: {prompt}"

            @trace_prompt(model="parent-model", version_tag="nested", db_path=db_path, agent_name="planner", step_name="parent")
            def parent(prompt: str) -> str:
                return child(f"subtask for {prompt}")

            parent("build plan")

            logs = sorted(get_logs(db_path), key=lambda row: row["id"])
            self.assertEqual(len(logs), 2)

            parent_log = logs[0]
            child_log = logs[1]

            self.assertIsNotNone(parent_log["run_id"])
            self.assertEqual(parent_log["run_id"], child_log["run_id"])
            self.assertEqual(child_log["parent_trace_id"], parent_log["trace_id"])
            self.assertEqual(parent_log["agent_name"], "planner")
            self.assertEqual(child_log["agent_name"], "worker")
            self.assertEqual(parent_log["step_name"], "parent")
            self.assertEqual(child_log["step_name"], "child")

    def test_trace_run_sets_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trace.db"

            @trace_prompt(model="run-model", version_tag="run", db_path=db_path)
            def step(prompt: str) -> str:
                return f"ok: {prompt}"

            with trace_run("run-123"):
                step("hello")

            logs = get_logs(db_path)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["run_id"], "run-123")
            self.assertIsNone(logs[0]["parent_trace_id"])
            self.assertEqual(logs[0]["step_name"], "step")


class CliTests(unittest.TestCase):
    def test_cli_generates_standalone_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            db_path = temp_dir_path / "trace.db"
            output_path = temp_dir_path / "index.html"

            @trace_prompt(model="cli-model", version_tag="cli", db_path=db_path)
            def recorded(prompt: str) -> str:
                return f"ok: {prompt}"

            recorded("generate dashboard")

            with patch(
                "sys.argv",
                [
                    "prompttrace-ui",
                    "--db-path",
                    str(db_path),
                    "--output",
                    str(output_path),
                ],
            ):
                cli_main()

            html = output_path.read_text(encoding="utf-8")
            self.assertIn("https://cdn.tailwindcss.com", html)
            self.assertIn("PromptTrace Dashboard", html)
            self.assertIn("generate dashboard", html)
            self.assertIn("ok: generate dashboard", html)
            self.assertIn("Prompt Tokens", html)
            self.assertIn("Completion Tokens", html)
            self.assertIn("Total Tokens", html)
            self.assertIn("Run", html)
            self.assertIn("Trace", html)
            self.assertIn("Parent Trace", html)
            self.assertIn("Table View", html)
            self.assertIn("Compact Tree View", html)
            self.assertIn("Expand All", html)
            self.assertIn("Collapse All", html)
            self.assertIn("runHeaderRow", html)
            self.assertIn("toggleRun", html)
            self.assertIn("toggleNode", html)
            self.assertIn("expandAllTree", html)
            self.assertIn("collapseAllTree", html)


if __name__ == "__main__":
    unittest.main()