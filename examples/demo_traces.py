from __future__ import annotations

import argparse
from pathlib import Path

from prompt_trace import trace_prompt


def build_demo_traces(output_db: Path, sample_count: int) -> None:
    @trace_prompt(model="demo-model", version_tag="demo-v1", db_path=output_db)
    def mock_llm(prompt: str) -> str:
        return f"Demo response for: {prompt}"

    prompts = [
        "Write a short product launch tweet.",
        "Summarize the benefits of local tracing.",
        "Draft a JSON schema for a user profile.",
        "Generate three naming ideas for an AI tool.",
        "Explain why SQLite is useful for prototyping.",
    ]

    for index in range(sample_count):
        prompt = prompts[index % len(prompts)]
        mock_llm(f"{prompt} (sample {index + 1})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo PromptTrace logs locally")
    parser.add_argument("--db-path", default="./prompt_trace.db", help="Path to SQLite database")
    parser.add_argument("--count", type=int, default=5, help="Number of demo traces to create")
    args = parser.parse_args()

    output_db = Path(args.db_path)
    build_demo_traces(output_db, args.count)
    print(f"Generated {args.count} demo traces in {output_db.resolve()}")


if __name__ == "__main__":
    main()