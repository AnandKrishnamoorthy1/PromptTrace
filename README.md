# PromptTrace

PromptTrace is a lightweight, open-source Python library for local LLM call tracing.

It is designed for AI developers who want immediate visibility into prompts and responses while iterating quickly, without standing up cloud telemetry stacks.

## Why this is better for rapid prototyping

Enterprise observability tools are powerful, but they are often optimized for production governance and team-wide monitoring. During prototyping, that can add friction.

PromptTrace is intentionally optimized for fast local loops:

- No config: one decorator and a local SQLite file.
- No network dependency: logs stay on your machine.
- Instant iteration: inspect prompt/response changes between code edits.
- Version-aware experimentation: use version_tag to compare prompt variants.
- Easy portability: SQLite DB and generated HTML are simple files.

In short, enterprise tools help at scale. PromptTrace helps you move faster before scale.

## Install

```bash
pip install -e .
```

## Quick start

```python
from prompt_trace import trace_prompt

@trace_prompt(model="gpt-4.1-mini", version_tag="draft-v1")
def call_llm(prompt: str) -> str:
    # Replace with your provider SDK call
    return f"Echo: {prompt}"

print(call_llm("Write me a launch tweet"))
```

After running decorated functions, a local SQLite DB is created automatically at:

- ./prompt_trace.db

## Generate dashboard

```bash
prompttrace-ui --db-path ./prompt_trace.db --output ./index.html
```

This generates a standalone HTML dashboard with:

- Tailwind CSS (CDN)
- Dark-mode aesthetic
- Searchable logs table
- Prompt and response inspection

## Logged schema

Table name: logs

Columns:

- id
- timestamp
- version_tag
- model
- prompt
- response
- latency_ms

## Project structure

```text
prompt_trace/
  __init__.py
  core.py
  cli.py
pyproject.toml
README.md
```
