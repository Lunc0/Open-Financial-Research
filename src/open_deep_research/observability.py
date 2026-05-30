"""研究工作流的进度追踪辅助函数。"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration


def build_trace_event(
    config: RunnableConfig,
    *,
    node: str,
    event_type: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在启用可追踪性时返回 trace 事件载荷。"""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.enable_traceability:
        return []
    return [
        {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": node,
            "event_type": event_type,
            "summary": summary,
            "details": details or {},
        }
    ]


def build_node_status(
    config: RunnableConfig,
    *,
    node: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在启用可追踪性时返回节点状态载荷。"""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.enable_traceability:
        return {}
    return {
        node: {
            "status": status,
            "summary": summary,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
    }
