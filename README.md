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

@trace_prompt(model="gpt-4o-mini", version_tag="draft-v1")
def call_llm(prompt: str) -> str:
    # Replace with your provider SDK call
    return f"Echo: {prompt}"

print(call_llm("Write me a launch tweet"))
```

By default, PromptTrace stores the logical prompt value in the `prompt` column.

Extraction order:

- argument named `prompt`
- keyword `prompt`
- first positional argument
- fallback to full call payload (`args`/`kwargs` JSON)

You can override this with decorator options like `prompt_arg_name="input_text"` or a custom `prompt_extractor` callable.

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
- Token usage columns (prompt/completion/total)
- Run/trace hierarchy columns for multi-agent flows
- Table View and Compact Tree View modes for execution threads
- Collapsible runs and branches in Compact Tree View
- Expand All and Collapse All controls for thread navigation

## Multi-agent parent/child tracing

PromptTrace now tracks nested decorated calls automatically:

- `run_id`: shared across a full traced execution
- `trace_id`: unique ID for each step
- `parent_trace_id`: links child steps to parent steps
- `agent_name`: optional agent label
- `step_name`: step label (defaults to function name)

Example:

```python
from prompt_trace import trace_prompt, trace_run

@trace_prompt(model="gpt-4.1-mini", agent_name="planner", step_name="plan")
def planner(prompt: str) -> str:
  return worker(f"subtask for {prompt}")

@trace_prompt(model="gpt-4.1-mini", agent_name="worker", step_name="execute")
def worker(prompt: str) -> str:
  return f"done: {prompt}"

with trace_run("demo-run"):
  planner("build growth strategy")
```

In the logs, `execute` is automatically linked to `plan` via `parent_trace_id`.

## Token usage tracking

PromptTrace tracks token usage using a provider-first strategy:

- Provider-reported usage from result payloads (for example `result.usage` or `result["usage"]`)
- Local estimation fallback when provider usage is unavailable

Supported provider usage keys:

- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `input_tokens`, `output_tokens` (auto-mapped)

If `tiktoken` is installed, estimation uses model-aware tokenization. Otherwise, PromptTrace uses a lightweight local approximation.

You can provide your own extractor:

```python
@trace_prompt(
  model="gpt-4.1-mini",
  usage_extractor=lambda result: {
    "prompt_tokens": result.meta.in_tokens,
    "completion_tokens": result.meta.out_tokens,
    "total_tokens": result.meta.in_tokens + result.meta.out_tokens,
  },
)
def call_llm(prompt: str) -> MyResult:
  ...
```

## Generate demo traces

```bash
python examples/demo_traces.py --count 5 --db-path ./prompt_trace.db
```

This creates sample logs automatically so you can open the dashboard immediately after install.

## Run tests

```bash
python -m unittest discover -s tests
```

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
- prompt_tokens
- completion_tokens
- total_tokens
- run_id
- trace_id
- parent_trace_id
- agent_name
- step_name

## Project structure

```text
prompt_trace/
  __init__.py
  core.py
  cli.py
pyproject.toml
README.md
```
