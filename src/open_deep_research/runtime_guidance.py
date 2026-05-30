"""运行中用户引导的服务层写入入口."""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig

from open_deep_research.observability import build_trace_event
from open_deep_research.state import UserGuidanceItem, normalize_user_guidance_item


class StateUpdateTarget(Protocol):
    """支持同步写入 LangGraph 状态的对象."""

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """把 values 写入指定 thread 的图状态."""


class AsyncStateUpdateTarget(Protocol):
    """支持异步写入 LangGraph 状态的对象."""

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """把 values 异步写入指定 thread 的图状态."""


def ensure_thread_config(config: RunnableConfig) -> str:
    """确认运行时引导写入时带有稳定 thread_id."""
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ValueError("runtime guidance updates require config.configurable.thread_id")
    return str(thread_id)


def normalize_guidance_contents(content: str | Sequence[str]) -> list[str]:
    """把聊天输入规整为非空引导文本列表."""
    raw_contents = [content] if isinstance(content, str) else list(content)
    contents = [raw_content.strip() for raw_content in raw_contents if raw_content and raw_content.strip()]
    if not contents:
        raise ValueError("runtime guidance content must not be empty")
    return contents


def build_runtime_guidance_item(
    content: str,
    *,
    guidance_id: str | None = None,
    created_at: str | None = None,
    source: str = "user",
    priority: int = 10,
    applies_to: str | None = "supervisor",
) -> UserGuidanceItem:
    """把一条聊天输入转换为结构化 UserGuidanceItem."""
    return normalize_user_guidance_item(
        {
            "id": guidance_id,
            "content": content.strip(),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "source": source,
            "priority": priority,
            "applies_to": applies_to,
            "status": "pending",
        }
    )


def build_runtime_guidance_update(
    content: str | Sequence[str],
    *,
    guidance_id: str | None = None,
    source: str = "user",
    priority: int = 10,
    applies_to: str | None = "supervisor",
    created_at: str | None = None,
) -> dict[str, Any]:
    """构造追加运行时用户引导的 LangGraph state update."""
    contents = normalize_guidance_contents(content)
    if guidance_id and len(contents) > 1:
        raise ValueError("guidance_id can only be used when appending one guidance item")

    guidance_items = [
        build_runtime_guidance_item(
            guidance_content,
            guidance_id=guidance_id,
            created_at=created_at,
            source=source,
            priority=priority,
            applies_to=applies_to,
        )
        for guidance_content in contents
    ]
    return {
        "user_guidance": {
            "type": "append",
            "value": guidance_items,
        }
    }


def build_guidance_received_trace(
    config: RunnableConfig,
    guidance_items: list[UserGuidanceItem],
) -> dict[str, Any]:
    """构造 guidance_received trace，并避免在 trace 中记录完整用户文本."""
    return {
        "trace_events": build_trace_event(
            config,
            node="runtime_guidance_ingest",
            event_type="guidance_received",
            summary="Runtime user guidance was appended to graph state.",
            details={
                "guidance_ids": [item["id"] for item in guidance_items],
                "content_lengths": {
                    item["id"]: len(item["content"]) for item in guidance_items
                },
                "source": guidance_items[0]["source"] if guidance_items else "user",
                "applies_to": guidance_items[0]["applies_to"] if guidance_items else None,
            },
        )
    }


def build_runtime_guidance_state_update(
    config: RunnableConfig,
    content: str | Sequence[str],
    *,
    guidance_id: str | None = None,
    source: str = "user",
    priority: int = 10,
    applies_to: str | None = "supervisor",
    created_at: str | None = None,
) -> dict[str, Any]:
    """构造包含 user_guidance 与 trace 的完整状态更新."""
    update = build_runtime_guidance_update(
        content,
        guidance_id=guidance_id,
        source=source,
        priority=priority,
        applies_to=applies_to,
        created_at=created_at,
    )
    guidance_items = update["user_guidance"]["value"]
    return {
        **update,
        **build_guidance_received_trace(config, guidance_items),
    }


def append_runtime_guidance(
    graph: StateUpdateTarget,
    config: RunnableConfig,
    content: str | Sequence[str],
    *,
    guidance_id: str | None = None,
    source: str = "user",
    priority: int = 10,
    applies_to: str | None = "supervisor",
    created_at: str | None = None,
    as_node: str | None = None,
    task_id: str | None = None,
) -> RunnableConfig:
    """把运行中聊天输入追加到指定 thread 的 state.user_guidance."""
    ensure_thread_config(config)
    update = build_runtime_guidance_state_update(
        config,
        content,
        guidance_id=guidance_id,
        source=source,
        priority=priority,
        applies_to=applies_to,
        created_at=created_at,
    )
    return graph.update_state(config, update, as_node=as_node, task_id=task_id)


async def aappend_runtime_guidance(
    graph: AsyncStateUpdateTarget,
    config: RunnableConfig,
    content: str | Sequence[str],
    *,
    guidance_id: str | None = None,
    source: str = "user",
    priority: int = 10,
    applies_to: str | None = "supervisor",
    created_at: str | None = None,
    as_node: str | None = None,
    task_id: str | None = None,
) -> RunnableConfig:
    """异步把运行中聊天输入追加到指定 thread 的 state.user_guidance."""
    ensure_thread_config(config)
    update = build_runtime_guidance_state_update(
        config,
        content,
        guidance_id=guidance_id,
        source=source,
        priority=priority,
        applies_to=applies_to,
        created_at=created_at,
    )
    return await graph.aupdate_state(config, update, as_node=as_node, task_id=task_id)
