from datetime import datetime

from open_deep_research.deep_researcher import (
    get_user_guidance_context,
    get_user_guidance_items,
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


def test_get_user_guidance_context_supports_legacy_and_runtime_config():
    context = get_user_guidance_context(
        {"user_guidance": ["重点关注现金流。"]},
        {"configurable": {"runtime_user_guidance": "补充债务期限结构。"}},
    )

    assert "重点关注现金流。" in context
    assert "补充债务期限结构。" in context
    assert "source=user" in context
    assert "source=config" in context


def test_get_user_guidance_items_sorts_by_priority_and_avoids_runtime_duplicate():
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
        {"configurable": {"runtime_user_guidance": "高优先级引导。"}},
    )

    assert [item["id"] for item in items] == ["guidance_high", "guidance_low"]
    assert [item["content"] for item in items].count("高优先级引导。") == 1
