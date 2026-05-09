# ADK Multi-Agent Example

This folder contains a small multi-agent orchestration example built with Google ADK and instrumented with PromptTrace.

The example shows a simple planning workflow:

- a planner agent creates the execution plan
- two research agents run in parallel
- a writer agent synthesizes the result
- a critic agent reviews the draft

PromptTrace is used to trace each stage into SQLite so you can inspect the execution tree in the dashboard.

## Install

```bash
pip install google-adk
pip install -e .
```

## Run the local traced workflow

```bash
python examples/ADK/adk_multi_agent_workflow.py --topic "launching a local LLM tool"
```

This runs the workflow locally, writes PromptTrace logs to `./prompt_trace.db`, and prints the final draft and review.

## Build the ADK agent graph

The example also exposes `build_adk_workflow()` for constructing the real ADK agent hierarchy:

- `SequentialAgent` for the plan -> research -> write -> review flow
- `ParallelAgent` for the parallel research step
- `LlmAgent` for each specialist agent

The ADK graph mirrors the local traced workflow, so the same structure is visible in both the code and the PromptTrace logs.
