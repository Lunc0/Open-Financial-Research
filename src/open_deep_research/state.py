"""Deep Research 智能体的图状态定义与数据结构。"""

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出定义
###################
class ConductResearch(BaseModel):
    """调用该工具以针对某个特定主题开展研究。"""
    research_topic: str = Field(
        description="需要研究的主题。应聚焦于单一主题，并使用高细节描述（至少一整段）。",
    )
    plan_item_id: str = Field(
        description="必填研究计划项 ID。必须先通过 UpdateResearchPlan 创建或引用 pending/running 计划项。",
    )

class ResearchPlanChange(BaseModel):
    """UpdateResearchPlan 工具中的单条计划变更。"""

    operation: Literal["add", "replace", "update_status", "update_priority"] = Field(
        description="计划变更类型：新增、替换、更新状态或调整优先级。",
    )
    plan_item_id: Optional[str] = Field(
        default=None,
        description="要新增或更新的计划项 ID；新增时可省略，由系统生成。",
    )
    plan_item_ids: list[str] = Field(
        default_factory=list,
        description="批量更新状态或优先级时的计划项 ID 列表。",
    )
    title: Optional[str] = Field(
        default=None,
        description="计划项标题。新增或替换计划项时使用。",
    )
    description: Optional[str] = Field(
        default=None,
        description="计划项描述。新增或替换计划项时使用。",
    )
    status: Optional[str] = Field(
        default=None,
        description="目标状态：pending/running/completed/cancelled/superseded/error。",
    )
    priority: Optional[int] = Field(
        default=None,
        description="目标优先级，数值越大越优先。",
    )
    created_from_guidance_id: Optional[str] = Field(
        default=None,
        description="触发该计划变更的用户引导 ID。",
    )
    guidance_ids: list[str] = Field(
        default_factory=list,
        description="该计划变更已经纳入的用户引导 ID 列表。",
    )
    supersedes_ids: list[str] = Field(
        default_factory=list,
        description="替换操作中被标记为 superseded 的旧计划项 ID 列表。",
    )


class UpdateResearchPlan(BaseModel):
    """调用该工具以显式创建、替换或调整研究计划。"""

    changes: list[ResearchPlanChange] = Field(
        description="要应用到 research_plan 的计划变更列表。",
    )
    rationale: str = Field(
        description="为什么这些计划变更能够反映 research_brief、user_guidance 和当前研究进展。",
    )

class ResearchComplete(BaseModel):
    """调用该工具以表示研究已经完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用于用户澄清请求的结构化输出模型。"""
    
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="用于向用户澄清报告范围的问题",
    )
    verification: str = Field(
        description="在用户提供必要信息后，我们将开始研究的确认消息。",
    )

class ResearchQuestion(BaseModel):
    """用于指导研究的研究问题与研究简报。"""
    
    research_brief: str = Field(
        description="将用于指导研究过程的研究问题。",
    )


###################
# 状态定义
###################
class UserGuidanceItem(TypedDict):
    """运行时用户引导的结构化状态对象。"""

    id: str
    content: str
    created_at: str
    source: str
    priority: int
    applies_to: Optional[str]
    status: str


class ResearchPlanItem(TypedDict):
    """可被运行时引导修改的研究计划项。"""

    id: str
    title: str
    description: str
    status: str
    priority: int
    created_from_guidance_id: Optional[str]


RESEARCH_PLAN_STATUSES = {
    "pending",
    "running",
    "completed",
    "cancelled",
    "superseded",
    "error",
}
ACTIVE_RESEARCH_PLAN_STATUSES = {"pending", "running"}


def normalize_research_plan_status(status: Any, default: str = "pending") -> str:
    """把计划状态规整到受支持的枚举值。"""
    normalized_status = str(status or default)
    if normalized_status not in RESEARCH_PLAN_STATUSES:
        return default
    return normalized_status


def normalize_user_guidance_item(
    item: str | dict[str, Any],
    *,
    default_source: str = "user",
    default_status: str = "pending",
) -> UserGuidanceItem:
    """把旧字符串格式或字典格式统一转换为结构化用户引导。"""
    if isinstance(item, str):
        return {
            "id": f"guidance_{uuid4().hex}",
            "content": item,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": default_source,
            "priority": 0,
            "applies_to": None,
            "status": default_status,
        }

    return {
        "id": str(item.get("id") or f"guidance_{uuid4().hex}"),
        "content": str(item.get("content", "")),
        "created_at": str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "source": str(item.get("source") or default_source),
        "priority": int(item.get("priority", 0)),
        "applies_to": item.get("applies_to"),
        "status": str(item.get("status") or default_status),
    }


def normalize_user_guidance_items(
    value: Any,
    *,
    default_source: str = "user",
    default_status: str = "pending",
) -> list[UserGuidanceItem]:
    """把任意支持的用户引导输入格式转换为结构化列表。"""
    if value is None:
        return []

    raw_items = value if isinstance(value, list) else [value]
    normalized_items = []
    for item in raw_items:
        if not item:
            continue
        normalized_item = normalize_user_guidance_item(
            item,
            default_source=default_source,
            default_status=default_status,
        )
        if normalized_item["content"]:
            normalized_items.append(normalized_item)
    return normalized_items


def user_guidance_reducer(current_value, new_value):
    """合并运行时用户引导，并支持追加、覆盖和状态标记。"""
    current_items = normalize_user_guidance_items(current_value)

    if isinstance(new_value, dict):
        operation = new_value.get("type")
        if operation == "override":
            return normalize_user_guidance_items(new_value.get("value", []))
        if operation == "append":
            return current_items + normalize_user_guidance_items(new_value.get("value", []))
        if operation == "mark_status":
            target_ids = set(new_value.get("ids", []))
            next_status = str(new_value.get("status", "applied"))
            return [
                {**item, "status": next_status} if item["id"] in target_ids else item
                for item in current_items
            ]

    return current_items + normalize_user_guidance_items(new_value)


def append_unique_reducer(current_value, new_value):
    """合并列表并按字符串值去重，保持已有顺序。"""
    current_items = list(current_value or [])
    if isinstance(new_value, dict):
        operation = new_value.get("type")
        if operation == "override":
            incoming_items = list(new_value.get("value", []))
        elif operation == "append":
            incoming_items = list(new_value.get("value", []))
        else:
            incoming_items = [new_value]
    elif isinstance(new_value, list):
        incoming_items = new_value
    elif new_value is None:
        incoming_items = []
    else:
        incoming_items = [new_value]

    merged_items = [str(item) for item in current_items]
    seen_items = set(merged_items)
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        merged_items = []
        seen_items = set()
    for item in incoming_items:
        normalized_item = str(item)
        if normalized_item not in seen_items:
            merged_items.append(normalized_item)
            seen_items.add(normalized_item)
    return merged_items


def normalize_research_plan_item(item: dict[str, Any]) -> ResearchPlanItem:
    """把字典输入转换为结构化研究计划项。"""
    description = str(item.get("description") or item.get("title") or "")
    return {
        "id": str(item.get("id") or f"plan_{uuid4().hex}"),
        "title": str(item.get("title") or description[:120] or "未命名研究计划"),
        "description": description,
        "status": normalize_research_plan_status(item.get("status")),
        "priority": int(item.get("priority", 0)),
        "created_from_guidance_id": item.get("created_from_guidance_id"),
    }


def normalize_research_plan_items(value: Any) -> list[ResearchPlanItem]:
    """把任意支持的研究计划输入格式转换为结构化列表。"""
    if value is None:
        return []

    raw_items = value if isinstance(value, list) else [value]
    normalized_items = []
    for item in raw_items:
        if not item:
            continue
        normalized_item = normalize_research_plan_item(item)
        if normalized_item["description"]:
            normalized_items.append(normalized_item)
    return normalized_items


def build_research_plan_item(
    research_topic: str,
    *,
    plan_item_id: Optional[str] = None,
    status: str = "pending",
    priority: int = 0,
    created_from_guidance_id: Optional[str] = None,
) -> ResearchPlanItem:
    """根据研究主题构建研究计划项。"""
    return normalize_research_plan_item(
        {
            "id": plan_item_id or f"plan_{uuid4().hex}",
            "title": research_topic[:120],
            "description": research_topic,
            "status": status,
            "priority": priority,
            "created_from_guidance_id": created_from_guidance_id,
        }
    )


def research_plan_change_to_dict(change: Any) -> dict[str, Any]:
    """兼容 Pydantic 工具入参和普通字典入参。"""
    if hasattr(change, "model_dump"):
        return change.model_dump()
    return dict(change)


def get_research_plan_change_ids(change: dict[str, Any]) -> list[str]:
    """提取单条计划变更要影响的计划项 ID。"""
    target_ids = list(change.get("plan_item_ids") or [])
    if change.get("plan_item_id"):
        target_ids.append(str(change["plan_item_id"]))
    return target_ids


def get_research_plan_change_guidance_ids(change: dict[str, Any]) -> list[str]:
    """提取单条计划变更实际吸收的用户引导 ID。"""
    guidance_ids = []
    if change.get("created_from_guidance_id"):
        guidance_ids.append(str(change["created_from_guidance_id"]))
    guidance_ids.extend(str(guidance_id) for guidance_id in change.get("guidance_ids", []))
    return list(dict.fromkeys(guidance_ids))


def apply_research_plan_changes_with_metadata(
    current_value: Any,
    changes: Any,
) -> tuple[list[ResearchPlanItem], list[str]]:
    """应用计划变更，并返回实际成功吸收的用户引导 ID。"""
    current_items = normalize_research_plan_items(current_value)
    merged_items = {item["id"]: item for item in current_items}
    applied_guidance_ids = []

    for raw_change in changes or []:
        change = research_plan_change_to_dict(raw_change)
        operation = change.get("operation")
        change_applied = False

        if operation == "add":
            item = normalize_research_plan_item(
                {
                    "id": change.get("plan_item_id"),
                    "title": change.get("title"),
                    "description": change.get("description") or change.get("title"),
                    "status": change.get("status") or "pending",
                    "priority": change.get("priority", 0),
                    "created_from_guidance_id": change.get("created_from_guidance_id"),
                }
            )
            if item["description"]:
                merged_items[item["id"]] = item
                change_applied = True
            if change_applied:
                applied_guidance_ids.extend(get_research_plan_change_guidance_ids(change))
            continue

        if operation == "replace":
            for plan_item_id in change.get("supersedes_ids") or []:
                if plan_item_id in merged_items:
                    merged_items[plan_item_id] = {
                        **merged_items[plan_item_id],
                        "status": "superseded",
                    }
                    change_applied = True
            if change.get("description") or change.get("title"):
                item = normalize_research_plan_item(
                    {
                        "id": change.get("plan_item_id"),
                        "title": change.get("title"),
                        "description": change.get("description") or change.get("title"),
                        "status": change.get("status") or "pending",
                        "priority": change.get("priority", 0),
                        "created_from_guidance_id": change.get("created_from_guidance_id"),
                    }
                )
                if item["description"]:
                    merged_items[item["id"]] = item
                    change_applied = True
            if change_applied:
                applied_guidance_ids.extend(get_research_plan_change_guidance_ids(change))
            continue

        if operation == "update_status":
            next_status = normalize_research_plan_status(change.get("status"))
            for plan_item_id in get_research_plan_change_ids(change):
                if plan_item_id in merged_items:
                    merged_items[plan_item_id] = {
                        **merged_items[plan_item_id],
                        "status": next_status,
                    }
                    change_applied = True
            if change_applied:
                applied_guidance_ids.extend(get_research_plan_change_guidance_ids(change))
            continue

        if operation == "update_priority":
            next_priority = int(change.get("priority", 0))
            for plan_item_id in get_research_plan_change_ids(change):
                if plan_item_id in merged_items:
                    merged_items[plan_item_id] = {
                        **merged_items[plan_item_id],
                        "priority": next_priority,
                    }
                    change_applied = True
            if change_applied:
                applied_guidance_ids.extend(get_research_plan_change_guidance_ids(change))

    return list(merged_items.values()), list(dict.fromkeys(applied_guidance_ids))


def apply_research_plan_changes(
    current_value: Any,
    changes: Any,
) -> list[ResearchPlanItem]:
    """按显式计划工具语义应用一组计划变更。"""
    updated_plan, _applied_guidance_ids = apply_research_plan_changes_with_metadata(
        current_value,
        changes,
    )
    return updated_plan


def research_plan_reducer(current_value, new_value):
    """合并研究计划，并支持追加、覆盖和按 ID 更新。"""
    current_items = normalize_research_plan_items(current_value)

    if isinstance(new_value, dict):
        operation = new_value.get("type")
        if operation == "override":
            return normalize_research_plan_items(new_value.get("value", []))
        if operation == "append":
            return current_items + normalize_research_plan_items(new_value.get("value", []))
        if operation == "upsert":
            incoming_items = normalize_research_plan_items(new_value.get("value", []))
            merged_items = {item["id"]: item for item in current_items}
            for item in incoming_items:
                merged_items[item["id"]] = {**merged_items.get(item["id"], {}), **item}
            return list(merged_items.values())
        if operation == "update_status":
            target_ids = set(new_value.get("ids", []))
            next_status = normalize_research_plan_status(new_value.get("status"))
            return [
                {**item, "status": next_status} if item["id"] in target_ids else item
                for item in current_items
            ]
        if operation == "update_priority":
            target_ids = set(new_value.get("ids", []))
            next_priority = int(new_value.get("priority", 0))
            return [
                {**item, "priority": next_priority} if item["id"] in target_ids else item
                for item in current_items
            ]
        if operation == "apply_changes":
            return apply_research_plan_changes(current_items, new_value.get("changes", []))

    return current_items + normalize_research_plan_items(new_value)

def override_reducer(current_value, new_value):
    """允许在状态合并时直接覆盖字段值的 reducer 函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

def merge_dict_reducer(current_value, new_value):
    """合并字典状态，并支持 override 语义。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", {})
    if current_value is None:
        current_value = {}
    if new_value is None:
        new_value = {}
    return {**current_value, **new_value}
    
class AgentInputState(MessagesState):
    """输入状态，仅包含 `messages`。"""

class AgentState(MessagesState):
    """主智能体状态，包含消息与研究过程数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    user_guidance: Annotated[list[UserGuidanceItem], user_guidance_reducer] = []
    handled_user_message_ids: Annotated[list[str], append_unique_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], research_plan_reducer] = []
    trace_events: Annotated[list[dict[str, Any]], operator.add] = []
    node_status: Annotated[dict[str, Any], merge_dict_reducer] = {}
    final_report: str

class SupervisorState(TypedDict):
    """研究监督者的状态，用于管理研究任务。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    pending_guidance_completion_attempts: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    user_guidance: Annotated[list[UserGuidanceItem], user_guidance_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], research_plan_reducer] = []
    trace_events: Annotated[list[dict[str, Any]], operator.add] = []
    node_status: Annotated[dict[str, Any], merge_dict_reducer] = {}

class ResearcherState(TypedDict):
    """单个研究员的状态，用于执行具体研究任务。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    plan_item_id: Optional[str]
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    user_guidance: Annotated[list[UserGuidanceItem], user_guidance_reducer] = []
    trace_events: Annotated[list[dict[str, Any]], operator.add] = []
    node_status: Annotated[dict[str, Any], merge_dict_reducer] = {}

class ResearcherOutputState(BaseModel):
    """单个研究员子图的输出状态。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    trace_events: Annotated[list[dict[str, Any]], operator.add] = []
    node_status: Annotated[dict[str, Any], merge_dict_reducer] = {}
