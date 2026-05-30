from datetime import datetime
from uuid import UUID

from open_deep_research.observability import build_node_status, build_trace_event


def test_build_trace_event_returns_complete_event_when_enabled():
    config = {"configurable": {"enable_traceability": True}}

    events = build_trace_event(
        config,
        node="researcher_tools",
        event_type="tool_call_completed",
        summary="完成工具调用。",
        details={"tool_name": "search"},
    )

    assert len(events) == 1
    event = events[0]
    UUID(event["id"])
    datetime.fromisoformat(event["timestamp"])
    assert event["node"] == "researcher_tools"
    assert event["event_type"] == "tool_call_completed"
    assert event["summary"] == "完成工具调用。"
    assert event["details"] == {"tool_name": "search"}


def test_build_trace_event_returns_empty_list_when_disabled():
    config = {"configurable": {"enable_traceability": False}}

    events = build_trace_event(
        config,
        node="researcher_tools",
        event_type="tool_call_completed",
        summary="完成工具调用。",
    )

    assert events == []


def test_build_node_status_returns_status_when_enabled():
    config = {"configurable": {"enable_traceability": True}}

    statuses = build_node_status(
        config,
        node="research_supervisor",
        status="running",
        summary="正在规划研究任务。",
        details={"iteration": 1},
    )

    assert set(statuses.keys()) == {"research_supervisor"}
    status = statuses["research_supervisor"]
    datetime.fromisoformat(status["updated_at"])
    assert status["status"] == "running"
    assert status["summary"] == "正在规划研究任务。"
    assert status["details"] == {"iteration": 1}


def test_build_node_status_returns_empty_dict_when_disabled():
    config = {"configurable": {"enable_traceability": False}}

    statuses = build_node_status(
        config,
        node="research_supervisor",
        status="running",
        summary="正在规划研究任务。",
    )

    assert statuses == {}
