"""Deep Research 智能体的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional

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

def override_reducer(current_value, new_value):
    """允许在状态合并时直接覆盖字段值的 reducer 函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """输入状态，仅包含 `messages`。"""

class AgentState(MessagesState):
    """主智能体状态，包含消息与研究过程数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """研究监督者的状态，用于管理研究任务。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """单个研究员的状态，用于执行具体研究任务。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """单个研究员子图的输出状态。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
