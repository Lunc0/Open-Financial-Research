import pytest

from open_deep_research.runtime_guidance import (
    aappend_runtime_guidance,
    append_runtime_guidance,
    build_runtime_guidance_state_update,
    build_runtime_guidance_update,
)


class FakeGraph:
    def __init__(self):
        self.calls = []

    def update_state(self, config, values, as_node=None, task_id=None):
        self.calls.append(
            {
                "config": config,
                "values": values,
                "as_node": as_node,
                "task_id": task_id,
            }
        )
        return {"configurable": {"thread_id": config["configurable"]["thread_id"]}}


class FakeAsyncGraph:
    def __init__(self):
        self.calls = []

    async def aupdate_state(self, config, values, as_node=None, task_id=None):
        self.calls.append(
            {
                "config": config,
                "values": values,
                "as_node": as_node,
                "task_id": task_id,
            }
        )
        return {"configurable": {"thread_id": config["configurable"]["thread_id"]}}


def test_build_runtime_guidance_update_appends_structured_guidance():
    update = build_runtime_guidance_update(
        "  重点关注现金流。  ",
        guidance_id="guidance_001",
        created_at="2026-05-30T00:00:00+00:00",
    )
    item = update["user_guidance"]["value"][0]

    assert update["user_guidance"]["type"] == "append"
    assert item["id"] == "guidance_001"
    assert item["content"] == "重点关注现金流。"
    assert item["source"] == "user"
    assert item["priority"] == 10
    assert item["applies_to"] == "supervisor"
    assert item["status"] == "pending"


def test_build_runtime_guidance_update_rejects_empty_content():
    with pytest.raises(ValueError, match="must not be empty"):
        build_runtime_guidance_update("  ")


def test_build_runtime_guidance_state_update_adds_trace_without_full_content():
    update = build_runtime_guidance_state_update(
        {"configurable": {"thread_id": "thread_001", "enable_traceability": True}},
        "重点关注现金流。",
        guidance_id="guidance_001",
        created_at="2026-05-30T00:00:00+00:00",
    )
    trace_event = update["trace_events"][0]

    assert trace_event["event_type"] == "guidance_received"
    assert trace_event["details"]["guidance_ids"] == ["guidance_001"]
    assert trace_event["details"]["content_lengths"] == {"guidance_001": 8}
    assert "重点关注现金流。" not in str(trace_event)


def test_append_runtime_guidance_writes_to_graph_state():
    graph = FakeGraph()
    config = {"configurable": {"thread_id": "thread_001", "enable_traceability": True}}

    result = append_runtime_guidance(
        graph,
        config,
        "补充监管风险。",
        guidance_id="guidance_001",
        created_at="2026-05-30T00:00:00+00:00",
        as_node="runtime_guidance_ingest",
    )

    assert result["configurable"]["thread_id"] == "thread_001"
    assert graph.calls[0]["config"] == config
    assert graph.calls[0]["as_node"] == "runtime_guidance_ingest"
    assert graph.calls[0]["values"]["user_guidance"]["value"][0]["id"] == "guidance_001"


def test_append_runtime_guidance_requires_thread_id():
    with pytest.raises(ValueError, match="thread_id"):
        append_runtime_guidance(FakeGraph(), {"configurable": {}}, "补充监管风险。")


@pytest.mark.asyncio
async def test_aappend_runtime_guidance_writes_to_graph_state():
    graph = FakeAsyncGraph()
    config = {"configurable": {"thread_id": "thread_001", "enable_traceability": True}}

    result = await aappend_runtime_guidance(
        graph,
        config,
        ["补充监管风险。", "关注债务期限。"],
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert result["configurable"]["thread_id"] == "thread_001"
    assert len(graph.calls[0]["values"]["user_guidance"]["value"]) == 2
