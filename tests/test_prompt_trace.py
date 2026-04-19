from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from prompt_trace import trace_prompt
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

            self.assertEqual(prompt_payload["args"], ["hello"])
            self.assertEqual(prompt_payload["kwargs"], {"temperature": 0.7})
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


if __name__ == "__main__":
    unittest.main()