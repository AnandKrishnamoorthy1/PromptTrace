from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from prompt_trace import trace_prompt, trace_run

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_DB_PATH = Path("./prompt_trace.db")
DEFAULT_RUN_ID = "adk-multi-agent-demo"


@dataclass(frozen=True)
class WorkflowResult:
    plan: str
    market_notes: str
    risk_notes: str
    draft: str
    review: str


def _make_planner(db_path: Path):
    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="planner",
        step_name="plan",
    )
    def planner(topic: str) -> str:
        return (
            f"Plan for {topic}: clarify the goal, gather supporting facts, "
            "draft the response, and review it before publishing."
        )

    return planner


def _make_market_researcher(db_path: Path):
    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="market_researcher",
        step_name="market_research",
    )
    async def market_research(topic: str) -> str:
        return f"Market notes for {topic}: local tracing helps teams debug prompts faster."

    return market_research


def _make_risk_researcher(db_path: Path):
    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="risk_researcher",
        step_name="risk_research",
    )
    async def risk_research(topic: str) -> str:
        return f"Risk notes for {topic}: the main risk is losing visibility into nested handoffs."

    return risk_research


def _make_research_sprint(db_path: Path):
    market_research = _make_market_researcher(db_path)
    risk_research = _make_risk_researcher(db_path)

    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="research_coordinator",
        step_name="research_sprint",
    )
    async def research_sprint(topic: str) -> dict[str, str]:
        market_notes, risk_notes = await asyncio.gather(
            market_research(topic),
            risk_research(topic),
        )
        return {
            "market_notes": market_notes,
            "risk_notes": risk_notes,
        }

    return research_sprint


def _make_writer(db_path: Path):
    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="writer",
        step_name="draft",
    )
    def writer(topic: str, plan: str, market_notes: str, risk_notes: str) -> str:
        return (
            f"Draft for {topic}:\n"
            f"{plan}\n\n"
            f"Supporting notes:\n- {market_notes}\n- {risk_notes}\n\n"
            "Recommendation: ship a small, traced workflow first and expand from there."
        )

    return writer


def _make_reviewer(db_path: Path):
    @trace_prompt(
        model=DEFAULT_MODEL,
        version_tag="adk-local",
        db_path=db_path,
        agent_name="critic",
        step_name="review",
    )
    def reviewer(draft: str) -> str:
        if "Recommendation:" not in draft:
            return "Needs revision: the draft should end with a clear recommendation."
        return "Approved: the draft is clear, structured, and ready to share."

    return reviewer


async def run_local_workflow_async(topic: str, db_path: Path = DEFAULT_DB_PATH) -> WorkflowResult:
    planner = _make_planner(db_path)
    research_sprint = _make_research_sprint(db_path)
    writer = _make_writer(db_path)
    reviewer = _make_reviewer(db_path)

    with trace_run(DEFAULT_RUN_ID):
        plan = planner(topic)
        research = await research_sprint(topic)
        draft = writer(topic, plan, research["market_notes"], research["risk_notes"])
        review = reviewer(draft)

    return WorkflowResult(
        plan=plan,
        market_notes=research["market_notes"],
        risk_notes=research["risk_notes"],
        draft=draft,
        review=review,
    )


def run_local_workflow(topic: str, db_path: Path = DEFAULT_DB_PATH) -> WorkflowResult:
    return asyncio.run(run_local_workflow_async(topic, db_path=db_path))


def build_adk_workflow():
    try:
        from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
    except ImportError as exc:
        raise RuntimeError(
            "google-adk is required to build the ADK workflow. Install it with `pip install google-adk`."
        ) from exc

    planner_agent = LlmAgent(
        name="planner_agent",
        model=DEFAULT_MODEL,
        description="Creates the execution plan for the workflow.",
        instruction=(
            "Create a short execution plan for the user's topic and save it to state['plan']. "
            "Keep the plan practical and concise."
        ),
        output_key="plan",
    )

    market_agent = LlmAgent(
        name="market_research_agent",
        model=DEFAULT_MODEL,
        description="Collects opportunity and adoption notes.",
        instruction=(
            "Analyze the topic from a market perspective. Save the result to state['market_notes']."
        ),
        output_key="market_notes",
    )

    risk_agent = LlmAgent(
        name="risk_research_agent",
        model=DEFAULT_MODEL,
        description="Collects delivery and product risks.",
        instruction=(
            "Analyze the topic from a risk perspective. Save the result to state['risk_notes']."
        ),
        output_key="risk_notes",
    )

    research_sprint = ParallelAgent(
        name="research_sprint",
        sub_agents=[market_agent, risk_agent],
    )

    writer_agent = LlmAgent(
        name="writer_agent",
        model=DEFAULT_MODEL,
        description="Synthesizes the plan and research into a draft recommendation.",
        instruction=(
            "Use {plan}, {market_notes}, and {risk_notes} to write a concise recommendation. "
            "Save the draft to state['draft']."
        ),
        output_key="draft",
    )

    reviewer_agent = LlmAgent(
        name="reviewer_agent",
        model=DEFAULT_MODEL,
        description="Reviews the draft and provides a final judgment.",
        instruction=(
            "Review {draft} for clarity, completeness, and actionability. "
            "Save the verdict to state['review']."
        ),
        output_key="review",
    )

    return SequentialAgent(
        name="prompttrace_adk_orchestrator",
        sub_agents=[planner_agent, research_sprint, writer_agent, reviewer_agent],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PromptTrace + Google ADK multi-agent demo")
    parser.add_argument("--topic", default="launching a local LLM tool", help="Topic to analyze")
    parser.add_argument("--db-path", default="./prompt_trace.db", help="SQLite database path")
    args = parser.parse_args()

    result = run_local_workflow(args.topic, db_path=Path(args.db_path))

    print("Plan:\n" + result.plan)
    print("\nMarket notes:\n" + result.market_notes)
    print("\nRisk notes:\n" + result.risk_notes)
    print("\nDraft:\n" + result.draft)
    print("\nReview:\n" + result.review)


if __name__ == "__main__":
    main()