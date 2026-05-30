import pytest
from langchain_core.messages import AIMessage

from open_deep_research.deep_researcher import (
    get_research_plan_context,
    supervisor_tools,
    validate_conduct_research_plan_item,
)
from open_deep_research.state import (
    ConductResearch,
    UpdateResearchPlan,
    apply_research_plan_changes,
    build_research_plan_item,
    normalize_research_plan_item,
    research_plan_reducer,
)


def make_tool_call(name, args, call_id="call_001"):
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def make_supervisor_state(tool_calls, research_plan=None):
    return {
        "supervisor_messages": [AIMessage(content="", tool_calls=tool_calls)],
        "research_brief": "研究现金流质量。",
        "research_iterations": 1,
        "research_plan": research_plan or [],
        "user_guidance": [],
    }


def test_conduct_research_requires_plan_item_id():
    with pytest.raises(Exception):
        ConductResearch(research_topic="分析现金流质量。")

    tool_args = ConductResearch(
        research_topic="分析现金流质量。",
        plan_item_id="plan_001",
    )

    assert tool_args.plan_item_id == "plan_001"


def test_update_research_plan_creates_pending_plan_item():
    tool_args = UpdateResearchPlan(
        rationale="把研究简报拆成一个可执行计划项。",
        changes=[
            {
                "operation": "add",
                "plan_item_id": "plan_001",
                "title": "现金流质量",
                "description": "分析现金流质量。",
                "priority": 5,
            }
        ],
    )
    plan = apply_research_plan_changes([], tool_args.changes)

    assert plan[0]["id"] == "plan_001"
    assert plan[0]["status"] == "pending"
    assert plan[0]["priority"] == 5


def test_research_plan_reducer_supports_append_upsert_status_and_priority():
    current = research_plan_reducer(
        None,
        [
            {
                "id": "plan_001",
                "title": "现金流",
                "description": "分析现金流质量。",
                "status": "pending",
                "priority": 1,
                "created_from_guidance_id": None,
            }
        ],
    )
    updated = research_plan_reducer(
        current,
        {"type": "update_status", "ids": ["plan_001"], "status": "running"},
    )
    reprioritized = research_plan_reducer(
        updated,
        {"type": "update_priority", "ids": ["plan_001"], "priority": 8},
    )

    assert reprioritized[0]["status"] == "running"
    assert reprioritized[0]["priority"] == 8


def test_research_plan_context_formats_existing_plan_items():
    context = get_research_plan_context(
        {
            "research_plan": [
                {
                    "id": "plan_001",
                    "title": "现金流",
                    "description": "分析现金流质量。",
                    "status": "pending",
                    "priority": 1,
                    "created_from_guidance_id": None,
                }
            ]
        }
    )

    assert "id=plan_001" in context
    assert "pending" in context
    assert "分析现金流质量。" in context


def test_validate_conduct_research_allows_only_active_plan_items():
    pending_item = normalize_research_plan_item(
        {
            "id": "plan_001",
            "title": "现金流",
            "description": "分析现金流质量。",
            "status": "pending",
        }
    )
    cancelled_item = normalize_research_plan_item(
        {
            "id": "plan_002",
            "title": "海外市场",
            "description": "分析海外市场。",
            "status": "cancelled",
        }
    )

    valid_item, valid_error = validate_conduct_research_plan_item(
        {"research_plan": [pending_item, cancelled_item]},
        make_tool_call(
            "ConductResearch",
            {"research_topic": "分析现金流质量。", "plan_item_id": "plan_001"},
        ),
    )
    missing_item, missing_error = validate_conduct_research_plan_item(
        {"research_plan": [pending_item]},
        make_tool_call("ConductResearch", {"research_topic": "分析现金流质量。"}),
    )
    invalid_item, invalid_error = validate_conduct_research_plan_item(
        {"research_plan": [pending_item]},
        make_tool_call(
            "ConductResearch",
            {"research_topic": "分析现金流质量。", "plan_item_id": "plan_x"},
        ),
    )
    inactive_item, inactive_error = validate_conduct_research_plan_item(
        {"research_plan": [pending_item, cancelled_item]},
        make_tool_call(
            "ConductResearch",
            {"research_topic": "分析海外市场。", "plan_item_id": "plan_002"},
        ),
    )

    assert valid_item["id"] == "plan_001"
    assert valid_error is None
    assert missing_item is None
    assert missing_error == "missing_plan_item_id"
    assert invalid_item is None
    assert invalid_error == "invalid_plan_item_id"
    assert inactive_item["id"] == "plan_002"
    assert inactive_error == "inactive_plan_item"


@pytest.mark.asyncio
async def test_supervisor_tools_rejects_conduct_research_without_plan_item_id(monkeypatch):
    class FakeResearcherSubgraph:
        calls = 0

        async def ainvoke(self, *_args, **_kwargs):
            self.calls += 1
            return {"compressed_research": "不应执行"}

    fake_subgraph = FakeResearcherSubgraph()
    monkeypatch.setattr(
        "open_deep_research.deep_researcher.researcher_subgraph",
        fake_subgraph,
    )
    command = await supervisor_tools(
        make_supervisor_state(
            [make_tool_call("ConductResearch", {"research_topic": "分析现金流质量。"})]
        ),
        {"configurable": {}},
    )

    assert fake_subgraph.calls == 0
    assert command.goto == "supervisor"
    assert "plan_item_id is required" in command.update["supervisor_messages"][0].content


@pytest.mark.asyncio
async def test_supervisor_tools_updates_plan_then_runs_active_plan_item(monkeypatch):
    class FakeResearcherSubgraph:
        async def ainvoke(self, state, _config):
            return {
                "compressed_research": f"完成 {state['plan_item_id']}",
                "raw_notes": ["现金流发现"],
                "trace_events": [],
                "node_status": {},
            }

    monkeypatch.setattr(
        "open_deep_research.deep_researcher.researcher_subgraph",
        FakeResearcherSubgraph(),
    )
    command = await supervisor_tools(
        make_supervisor_state(
            [
                make_tool_call(
                    "UpdateResearchPlan",
                    {
                        "rationale": "创建现金流研究计划。",
                        "changes": [
                            {
                                "operation": "add",
                                "plan_item_id": "plan_001",
                                "title": "现金流",
                                "description": "分析现金流质量。",
                            }
                        ],
                    },
                    "call_plan",
                ),
                make_tool_call(
                    "ConductResearch",
                    {"research_topic": "分析现金流质量。", "plan_item_id": "plan_001"},
                    "call_research",
                ),
            ]
        ),
        {"configurable": {"max_concurrent_research_units": 1}},
    )

    plan = command.update["research_plan"]["value"]

    assert plan[0]["id"] == "plan_001"
    assert plan[0]["status"] == "completed"
    assert command.update["raw_notes"] == ["现金流发现"]


@pytest.mark.asyncio
async def test_supervisor_tools_marks_researcher_failure_as_error(monkeypatch):
    class FakeResearcherSubgraph:
        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("研究失败")

    monkeypatch.setattr(
        "open_deep_research.deep_researcher.researcher_subgraph",
        FakeResearcherSubgraph(),
    )
    plan_item = build_research_plan_item(
        "分析现金流质量。",
        plan_item_id="plan_001",
        status="pending",
    )
    command = await supervisor_tools(
        make_supervisor_state(
            [
                make_tool_call(
                    "ConductResearch",
                    {"research_topic": "分析现金流质量。", "plan_item_id": "plan_001"},
                )
            ],
            [plan_item],
        ),
        {"configurable": {"max_concurrent_research_units": 1}},
    )

    plan = command.update["research_plan"]["value"]
    event_types = [event["event_type"] for event in command.update["trace_events"]]

    assert plan[0]["status"] == "error"
    assert "research_task_failed" in event_types
