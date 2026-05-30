from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from open_deep_research.deep_researcher import (
    clarify_with_user,
    collect_unhandled_chat_guidance,
    get_current_user_message_ids,
    get_user_guidance_context,
    get_user_guidance_items,
    supervisor_tools,
)
from open_deep_research.state import (
    normalize_user_guidance_item,
    normalize_user_guidance_items,
    user_guidance_reducer,
)


def test_normalize_string_guidance_to_structured_item():
    item = normalize_user_guidance_item("重点关注现金流。")

    assert item["id"].startswith("guidance_")
    datetime.fromisoformat(item["created_at"])
    assert item["content"] == "重点关注现金流。"
    assert item["source"] == "user"
    assert item["priority"] == 0
    assert item["applies_to"] is None
    assert item["status"] == "pending"


def test_normalize_structured_guidance_preserves_fields():
    item = normalize_user_guidance_item(
        {
            "id": "guidance_001",
            "content": "聚焦监管政策风险。",
            "created_at": "2026-05-30T00:00:00+00:00",
            "source": "api",
            "priority": 10,
            "applies_to": "supervisor",
            "status": "applied",
        }
    )

    assert item == {
        "id": "guidance_001",
        "content": "聚焦监管政策风险。",
        "created_at": "2026-05-30T00:00:00+00:00",
        "source": "api",
        "priority": 10,
        "applies_to": "supervisor",
        "status": "applied",
    }


def test_guidance_reducer_appends_legacy_and_structured_inputs():
    current = user_guidance_reducer(None, ["重点关注现金流。"])
    result = user_guidance_reducer(
        current,
        {
            "type": "append",
            "value": [
                {
                    "id": "guidance_002",
                    "content": "补充债务期限结构。",
                    "created_at": "2026-05-30T00:00:00+00:00",
                    "source": "user",
                    "priority": 5,
                    "applies_to": None,
                    "status": "pending",
                }
            ],
        },
    )

    assert [item["content"] for item in result] == [
        "重点关注现金流。",
        "补充债务期限结构。",
    ]


def test_guidance_reducer_supports_override_and_mark_status():
    overridden = user_guidance_reducer(
        [{"id": "old", "content": "旧引导"}],
        {
            "type": "override",
            "value": [
                {
                    "id": "guidance_001",
                    "content": "只研究中国内地市场。",
                    "created_at": "2026-05-30T00:00:00+00:00",
                    "source": "user",
                    "priority": 10,
                    "applies_to": None,
                    "status": "pending",
                }
            ],
        },
    )
    marked = user_guidance_reducer(
        overridden,
        {"type": "mark_status", "ids": ["guidance_001"], "status": "applied"},
    )

    assert len(overridden) == 1
    assert overridden[0]["content"] == "只研究中国内地市场。"
    assert marked[0]["status"] == "applied"


def test_get_user_guidance_context_uses_state_guidance_only():
    context = get_user_guidance_context(
        {"user_guidance": ["重点关注现金流。"]},
        {"configurable": {"runtime_user_guidance": "补充债务期限结构。"}},
    )

    assert "重点关注现金流。" in context
    assert "补充债务期限结构。" not in context
    assert "source=user" in context
    assert "source=config" not in context


def test_get_user_guidance_items_sorts_by_priority():
    items = get_user_guidance_items(
        {
            "user_guidance": normalize_user_guidance_items(
                [
                    {
                        "id": "guidance_low",
                        "content": "低优先级引导。",
                        "created_at": "2026-05-30T00:00:00+00:00",
                        "source": "user",
                        "priority": 1,
                        "applies_to": None,
                        "status": "pending",
                    },
                    {
                        "id": "guidance_high",
                        "content": "高优先级引导。",
                        "created_at": "2026-05-30T00:00:00+00:00",
                        "source": "user",
                        "priority": 10,
                        "applies_to": None,
                        "status": "pending",
                    },
                ]
            )
        },
        {"configurable": {}},
    )

    assert [item["id"] for item in items] == ["guidance_high", "guidance_low"]


def test_current_user_message_ids_are_stable_for_initial_messages():
    ids = get_current_user_message_ids(
        {
            "messages": [
                HumanMessage(content="初始问题。", id="msg_initial"),
                AIMessage(content="确认开始研究。", id="msg_ai"),
            ]
        }
    )

    assert ids == ["msg_initial"]


def test_collect_unhandled_chat_guidance_requires_existing_research_brief():
    guidance_items, handled_ids = collect_unhandled_chat_guidance(
        {"messages": [HumanMessage(content="补充监管风险。", id="msg_guidance")]}
    )

    assert guidance_items == []
    assert handled_ids == []


def test_collect_unhandled_chat_guidance_converts_new_studio_message():
    guidance_items, handled_ids = collect_unhandled_chat_guidance(
        {
            "research_brief": "研究油车和电车。",
            "handled_user_message_ids": ["msg_initial"],
            "messages": [
                HumanMessage(content="油车和电车哪个好？", id="msg_initial"),
                AIMessage(content="开始研究。", id="msg_ai"),
                HumanMessage(content="补充二手车保值率。", id="msg_guidance"),
            ],
        }
    )

    assert handled_ids == ["msg_guidance"]
    assert guidance_items[0]["id"] == "guidance_from_msg_guidance"
    assert guidance_items[0]["content"] == "补充二手车保值率。"
    assert guidance_items[0]["source"] == "studio_chat"
    assert guidance_items[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_clarify_with_user_routes_existing_thread_chat_input_as_guidance():
    command = await clarify_with_user(
        {
            "research_brief": "研究油车和电车。",
            "handled_user_message_ids": ["msg_initial"],
            "messages": [
                HumanMessage(content="油车和电车哪个好？", id="msg_initial"),
                AIMessage(content="开始研究。", id="msg_ai"),
                HumanMessage(content="补充二手车保值率。", id="msg_guidance"),
            ],
        },
        {"configurable": {"enable_traceability": True}},
    )

    assert command.goto == "research_supervisor"
    assert command.update["handled_user_message_ids"]["value"] == ["msg_guidance"]
    guidance_item = command.update["user_guidance"]["value"][0]
    assert guidance_item["content"] == "补充二手车保值率。"
    assert guidance_item["source"] == "studio_chat"


@pytest.mark.asyncio
async def test_update_research_plan_marks_pending_guidance_as_applied():
    command = await supervisor_tools(
        {
            "supervisor_messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateResearchPlan",
                            "args": {
                                "rationale": "把用户引导转成研究计划。",
                                "changes": [
                                    {
                                        "operation": "add",
                                        "plan_item_id": "plan_001",
                                        "title": "现金流",
                                        "description": "重点研究现金流质量。",
                                        "created_from_guidance_id": "guidance_001",
                                    }
                                ],
                            },
                            "id": "call_plan",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "research_brief": "研究现金流质量。",
            "research_iterations": 1,
            "research_plan": [],
            "user_guidance": [
                {
                    "id": "guidance_001",
                    "content": "重点研究现金流质量。",
                    "created_at": "2026-05-30T00:00:00+00:00",
                    "source": "user",
                    "priority": 10,
                    "applies_to": "supervisor",
                    "status": "pending",
                }
            ],
        },
        {"configurable": {}},
    )

    assert command.update["user_guidance"] == {
        "type": "mark_status",
        "ids": ["guidance_001"],
        "status": "applied",
    }


def make_pending_guidance_state(tool_calls, completion_attempts=0):
    return {
        "supervisor_messages": [AIMessage(content="", tool_calls=tool_calls)],
        "research_brief": "研究现金流质量。",
        "research_iterations": 1,
        "research_plan": [],
        "pending_guidance_completion_attempts": completion_attempts,
        "user_guidance": [
            {
                "id": "guidance_001",
                "content": "补充监管风险。",
                "created_at": "2026-05-30T00:00:00+00:00",
                "source": "user",
                "priority": 10,
                "applies_to": "supervisor",
                "status": "pending",
            }
        ],
    }


@pytest.mark.asyncio
async def test_supervisor_tools_blocks_completion_when_guidance_is_pending():
    command = await supervisor_tools(
        make_pending_guidance_state(
            [
                {
                    "name": "ResearchComplete",
                    "args": {},
                    "id": "call_complete",
                    "type": "tool_call",
                }
            ]
        ),
        {"configurable": {}},
    )

    assert command.goto == "supervisor"
    assert command.update["pending_guidance_completion_attempts"] == 1
    assert "user_guidance" not in command.update
    assert "UpdateResearchPlan" in command.update["supervisor_messages"][0].content


@pytest.mark.asyncio
async def test_supervisor_tools_marks_pending_guidance_error_after_completion_retry():
    command = await supervisor_tools(
        make_pending_guidance_state(
            [
                {
                    "name": "ResearchComplete",
                    "args": {},
                    "id": "call_complete",
                    "type": "tool_call",
                }
            ],
            completion_attempts=1,
        ),
        {"configurable": {}},
    )

    assert command.goto == END
    assert command.update["user_guidance"] == {
        "type": "mark_status",
        "ids": ["guidance_001"],
        "status": "error",
    }


@pytest.mark.asyncio
async def test_noop_plan_update_does_not_mark_guidance_applied():
    command = await supervisor_tools(
        {
            "supervisor_messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateResearchPlan",
                            "args": {
                                "rationale": "尝试更新不存在的计划项。",
                                "changes": [
                                    {
                                        "operation": "update_status",
                                        "plan_item_id": "missing_plan",
                                        "status": "cancelled",
                                        "guidance_ids": ["guidance_001"],
                                    }
                                ],
                            },
                            "id": "call_plan",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "research_brief": "研究现金流质量。",
            "research_iterations": 1,
            "research_plan": [],
            "user_guidance": [
                {
                    "id": "guidance_001",
                    "content": "补充监管风险。",
                    "created_at": "2026-05-30T00:00:00+00:00",
                    "source": "user",
                    "priority": 10,
                    "applies_to": "supervisor",
                    "status": "pending",
                }
            ],
        },
        {"configurable": {}},
    )

    assert "user_guidance" not in command.update
    assert command.update["research_plan"]["value"] == []
