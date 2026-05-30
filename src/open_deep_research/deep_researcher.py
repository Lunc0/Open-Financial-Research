"""Deep Research 智能体的主 LangGraph 实现。"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.domain_profiles import get_domain_profile_prompt
from open_deep_research.observability import build_node_status, build_trace_event
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化可配置模型，供整个智能体流程复用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


def get_user_guidance_context(state: dict, config: RunnableConfig) -> str:
    """Build latest user guidance context from state and config."""
    configurable = Configuration.from_runnable_config(config)
    guidance_items = list(state.get("user_guidance", []) or [])
    if configurable.runtime_user_guidance:
        guidance_items.append(configurable.runtime_user_guidance)
    if not guidance_items:
        return "当前没有运行时用户引导。"
    return "\n".join(f"- {item}" for item in guidance_items if item)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，并在研究范围不清晰时提出澄清问题。
    
    该函数用于判断在进入研究流程前是否需要向用户澄清需求。
    如果关闭澄清功能或当前无需澄清，则直接进入研究阶段。
    
    参数:
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置与偏好
        
    返回:
        Command：要么以澄清问题结束，要么进入研究简报阶段
    """
    # 第 1 步：检查配置中是否启用澄清
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究
        return Command(
            goto="write_research_brief",
            update={
                "trace_events": build_trace_event(
                    config,
                    node="clarify_with_user",
                    event_type="skipped",
                    summary="Clarification disabled; proceeding to research brief.",
                ),
                "node_status": build_node_status(
                    config,
                    node="clarify_with_user",
                    status="skipped",
                    summary="Clarification disabled.",
                ),
            },
        )
    
    # 第 2 步：准备结构化澄清分析所需模型
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 为模型配置结构化输出与重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 第 3 步：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)]) # 把刚才拼好的整段提示词包装成一条“人类消息”
    
    # 第 4 步：根据澄清分析结果路由
    if response.need_clarification:
        # 返回澄清问题并结束当前轮次
        return Command(
            goto=END, 
            update={
                "messages": [AIMessage(content=response.question)],
                "trace_events": build_trace_event(
                    config,
                    node="clarify_with_user",
                    event_type="clarification_requested",
                    summary="Asked the user to clarify the research scope.",
                    details={"question": response.question},
                ),
                "node_status": build_node_status(
                    config,
                    node="clarify_with_user",
                    status="waiting_for_user",
                    summary="Waiting for scope clarification.",
                ),
            }
        )
    else:
        # 返回确认信息并继续研究流程
        return Command(
            goto="write_research_brief", 
            update={
                "messages": [AIMessage(content=response.verification)],
                "trace_events": build_trace_event(
                    config,
                    node="clarify_with_user",
                    event_type="completed",
                    summary="Research scope is clear enough to continue.",
                ),
                "node_status": build_node_status(
                    config,
                    node="clarify_with_user",
                    status="completed",
                    summary="Scope confirmed.",
                ),
            }
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转为结构化研究简报，并初始化监督者上下文。
    
    该函数会分析用户消息并生成聚焦的研究简报，供研究监督者使用。
    同时会设置监督者的初始上下文与提示信息。
    
    参数:
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置
        
    返回:
        Command：携带初始化上下文并进入研究监督者节点
    """
    # 第 1 步：设置用于结构化输出的研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 配置模型以生成结构化研究问题
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 2 步：根据用户消息生成结构化研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
        domain_profile_prompt=get_domain_profile_prompt(configurable.research_profile),
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 第 3 步：用研究简报和指令初始化监督者
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        domain_profile_prompt=get_domain_profile_prompt(configurable.research_profile),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    runtime_guidance = [configurable.runtime_user_guidance] if configurable.runtime_user_guidance else []
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "user_guidance": {"type": "override", "value": runtime_guidance},
            "trace_events": build_trace_event(
                config,
                node="write_research_brief",
                event_type="completed",
                summary="Created research brief and initialized supervisor.",
                details={
                    "research_profile": configurable.research_profile.value,
                    "research_brief_preview": response.research_brief[:500],
                },
            ),
            "node_status": build_node_status(
                config,
                node="write_research_brief",
                status="completed",
                summary="Research brief created.",
                details={"research_profile": configurable.research_profile.value},
            ),
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """研究监督者主节点：规划研究策略并向研究员分派任务。
    
    监督者会分析研究简报并决定如何拆分任务。
    可调用 think_tool 做策略思考，调用 ConductResearch 分派子任务，
    或在结论充分时调用 ResearchComplete。
    
    参数:
        state: 当前监督者状态，包含消息与研究上下文
        config: 运行时配置，包含模型设置
        
    返回:
        Command：进入 supervisor_tools 执行工具调用
    """
    # 第 1 步：使用可用工具配置监督者模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 可用工具：研究任务分派、完成信号、策略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # 为模型配置工具、重试逻辑与模型参数
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 2 步：基于当前上下文生成监督者回复
    supervisor_messages = state.get("supervisor_messages", [])
    guidance_context = get_user_guidance_context(state, config)
    runtime_context = HumanMessage(
        content=(
            "运行时用户引导如下。若这些引导与原研究简报存在冲突，优先按最新用户引导调整后续研究；"
            "若引导只是补充范围，则把它纳入下一轮研究计划。\n"
            f"{guidance_context}"
        )
    )
    response = await research_model.ainvoke(supervisor_messages + [runtime_context])
    
    # 第 3 步：更新状态并进入工具执行阶段
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "trace_events": build_trace_event(
                config,
                node="research_supervisor",
                event_type="iteration",
                summary="Supervisor planned the next research step.",
                details={
                    "iteration": state.get("research_iterations", 0) + 1,
                    "tool_calls": [tool_call["name"] for tool_call in response.tool_calls],
                    "user_guidance": guidance_context,
                },
            ),
            "node_status": build_node_status(
                config,
                node="research_supervisor",
                status="running",
                summary="Supervisor is planning and delegating research.",
                details={"iteration": state.get("research_iterations", 0) + 1},
            ),
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行监督者调用的工具，包括研究分派与策略思考。
    
    该函数处理三类工具调用：
    1. think_tool：策略反思，继续对话
    2. ConductResearch：分派研究任务给子研究员
    3. ResearchComplete：标记研究阶段完成
    
    参数:
        state: 当前监督者状态，包含消息与迭代次数
        config: 运行时配置，包含研究限制与模型设置
        
    返回:
        Command：要么继续监督循环，要么结束研究阶段
    """
    # 第 1 步：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 定义研究阶段退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # 若满足任一终止条件则退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "trace_events": build_trace_event(
                    config,
                    node="supervisor_tools",
                    event_type="completed",
                    summary="Supervisor research phase completed.",
                    details={
                        "reason": (
                            "max_iterations"
                            if exceeded_allowed_iterations
                            else "no_tool_calls"
                            if no_tool_calls
                            else "research_complete"
                        ),
                        "iterations": research_iterations,
                    },
                ),
                "node_status": build_node_status(
                    config,
                    node="research_supervisor",
                    status="completed",
                    summary="Research delegation completed.",
                    details={"iterations": research_iterations},
                ),
            }
        )
    
    # 第 2 步：统一处理全部工具调用（含 think_tool 与 ConductResearch）
    all_tool_messages = []
    update_payload = {
        "supervisor_messages": [],
        "trace_events": build_trace_event(
            config,
            node="supervisor_tools",
            event_type="tool_calls_received",
            summary="Supervisor tool calls are being processed.",
            details={
                "tool_calls": [
                    {
                        "name": tool_call["name"],
                        "args": tool_call.get("args", {}),
                    }
                    for tool_call in most_recent_message.tool_calls
                ]
            },
        ),
        "node_status": build_node_status(
            config,
            node="supervisor_tools",
            status="running",
            summary="Processing supervisor tool calls.",
        ),
    }
    
    # 处理 think_tool 调用（策略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # 处理 ConductResearch 调用（研究任务分派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # 限制并发研究单元数量，避免资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                    "user_guidance": state.get("user_guidance", []),
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # 用研究结果构造工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
                update_payload["trace_events"].extend(observation.get("trace_events", []))
                update_payload["node_status"] = {
                    **update_payload.get("node_status", {}),
                    **observation.get("node_status", {}),
                }
                update_payload["trace_events"].extend(build_trace_event(
                    config,
                    node="supervisor_tools",
                    event_type="research_task_completed",
                    summary="A delegated research task completed.",
                    details={
                        "research_topic": tool_call["args"]["research_topic"],
                        "compressed_length": len(observation.get("compressed_research", "")),
                    },
                ))
            
            # 对超出并发限制的研究调用返回错误消息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 聚合所有研究结果中的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e: # 出现任何错误，都会直接结束supervisor阶段，并返回当前已获取的笔记与研究简报。这是为了保证在工具调用过程中出现问题时，智能体能够优雅地降级并提供部分结果，而不是完全失败。
            # 处理研究执行错误
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # 令牌上限或其他错误：结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                        "trace_events": build_trace_event(
                            config,
                            node="supervisor_tools",
                            event_type="error",
                            summary="Research task execution failed; returning partial findings.",
                            details={"error": str(e)},
                        ),
                        "node_status": build_node_status(
                            config,
                            node="research_supervisor",
                            status="error",
                            summary="Research delegation failed; partial findings returned.",
                            details={"error": str(e)},
                        ),
                    }
                )
    
    # 第 3 步：返回包含全部工具结果的命令
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# 监督者子图构建
# 创建负责研究分派与协调的监督者工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加监督者节点用于研究管理
supervisor_builder.add_node("supervisor", supervisor)           # 监督者主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器

# 定义监督者工作流边
supervisor_builder.add_edge(START, "supervisor")  # 监督者入口节点

# 编译监督者子图，供主工作流使用
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """单个研究员节点：围绕特定主题执行聚焦研究。
    
    研究员从监督者接收具体研究主题，并使用可用工具（搜索、think_tool、MCP）
    收集信息。可在多次搜索之间使用 think_tool 做策略规划。
    
    参数:
        state: 当前研究员状态，包含消息与主题上下文
        config: 运行时配置，包含模型设置与工具可用性
        
    返回:
        Command：进入 researcher_tools 执行工具调用
    """
    # 第 1 步：加载配置并校验工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 获取全部可用研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # 第 2 步：为研究员模型配置工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 若存在 MCP 上下文则注入系统提示
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str(),
        domain_profile_prompt=get_domain_profile_prompt(configurable.research_profile),
        user_guidance_context=get_user_guidance_context(state, config),
    )
    
    # 为模型配置工具、重试逻辑与参数
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 3 步：在系统上下文下生成研究员回复
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # 第 4 步：更新状态并进入工具执行阶段
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
            "trace_events": build_trace_event(
                config,
                node="researcher",
                event_type="iteration",
                summary="Researcher planned or requested tools for a subtask.",
                details={
                    "research_topic": state.get("research_topic", ""),
                    "iteration": state.get("tool_call_iterations", 0) + 1,
                    "tool_calls": [tool_call["name"] for tool_call in response.tool_calls],
                },
            ),
            "node_status": build_node_status(
                config,
                node="researcher",
                status="running",
                summary="Researcher is collecting evidence.",
                details={
                    "research_topic": state.get("research_topic", ""),
                    "iteration": state.get("tool_call_iterations", 0) + 1,
                },
            ),
        }
    )

# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """安全执行工具，并进行错误处理。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索与策略思考。
    compress_research 的职责完全不同：换一个 prompt、换一个模型、整理全文
    
    该函数处理多类研究员工具调用：
    1. think_tool：策略反思，继续研究对话
    2. 搜索工具（tavily_search、web_search）：信息检索
    3. MCP 工具：外部能力集成
    4. ResearchComplete：标记当前研究任务完成
    
    参数:
        state: 当前研究员状态，包含消息与迭代次数
        config: 运行时配置，包含研究限制与工具设置
        
    返回:
        Command：要么继续研究循环，要么进入压缩阶段
    """
    # 第 1 步：提取当前状态并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 若无工具调用（含原生网页搜索）则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or  
        anthropic_websearch_called(most_recent_message)   # 模型原生调用 web search 时，这个项目不手动生成 ToolMessage 来拼接结果；搜索结果直接留在模型返回的 AIMessage 及其 metadata 里，后续靠保留这条 AIMessage 继续传递。
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(
            goto="compress_research",
            update={
                "trace_events": build_trace_event(
                    config,
                    node="researcher_tools",
                    event_type="no_tool_calls",
                    summary="Researcher stopped tool use and moved to compression.",
                    details={"research_topic": state.get("research_topic", "")},
                ),
                "node_status": build_node_status(
                    config,
                    node="researcher_tools",
                    status="completed",
                    summary="No further tools requested.",
                ),
            },
        ) # 有机会重新尝试 但是否重试取决于 supervisor 模型的判断 ；
                                                 # 真出了“异常”，那么异常会向上抛，又因为upervisor的工具调用是放在 try except 里的，所以最终会被捕获到（or true），并优雅地结束整个研究阶段，返回当前已获取的笔记与研究简报。这种设计既保证了智能体的鲁棒性，又能在出现问题时提供部分结果，而不是完全失败。
    
    # 第 2 步：处理其他工具调用（搜索、MCP 等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks) # 把这些工具任务并发跑起来
    tool_trace_events = build_trace_event(
        config,
        node="researcher_tools",
        event_type="tool_calls_executed",
        summary="Researcher tool calls completed.",
        details={
            "research_topic": state.get("research_topic", ""),
            "tool_calls": [
                {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}),
                    "observation_length": len(str(observation)),
                }
                for tool_call, observation in zip(tool_calls, observations)
            ],
        },
    )
    
    # 根据执行结果构造工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # 第 3 步：检查后置退出条件（工具处理后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩阶段
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "trace_events": tool_trace_events,
                "node_status": build_node_status(
                    config,
                    node="researcher_tools",
                    status="completed",
                    summary="Tool budget reached or research marked complete.",
                    details={
                        "exceeded_iterations": exceeded_iterations,
                        "research_complete_called": research_complete_called,
                    },
                ),
            }
        )
    
    # 携带工具结果继续研究循环
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "trace_events": tool_trace_events,
            "node_status": build_node_status(
                config,
                node="researcher_tools",
                status="running",
                summary="Tool observations added; researcher will continue.",
            ),
        }
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究结果压缩并综合为简洁、结构化摘要。
    
    该函数会收集研究员产出的研究结论、工具输出和 AI 消息，
    在保留关键信息的前提下提炼为清晰、完整的总结。
    
    参数:
        state: 当前研究员状态，包含累计研究消息
        config: 运行时配置，包含压缩模型设置
        
    返回:
        包含压缩摘要与原始笔记的字典
    """
    # 第 1 步：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # 第 2 步：准备压缩所需消息
    researcher_messages = state.get("researcher_messages", [])
    
    # 添加指令：从研究模式切换到压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # 第 3 步：执行压缩，并针对令牌上限进行重试
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # 构造聚焦压缩任务的系统提示
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)
            
            # 从所有工具与 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # 返回压缩成功结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                "trace_events": build_trace_event(
                    config,
                    node="compress_research",
                    event_type="completed",
                    summary="Compressed subtask findings.",
                    details={
                        "research_topic": state.get("research_topic", ""),
                        "compressed_length": len(str(response.content)),
                        "raw_notes_length": len(raw_notes_content),
                    },
                ),
                "node_status": build_node_status(
                    config,
                    node="compress_research",
                    status="completed",
                    summary="Subtask findings compressed.",
                    details={"research_topic": state.get("research_topic", "")},
                ),
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # 若超出令牌上限，则移除较早消息后重试
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # 其他错误：继续重试
            continue
    
    # 第 4 步：若重试耗尽则返回错误结果
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
        "trace_events": build_trace_event(
            config,
            node="compress_research",
            event_type="error",
            summary="Compression retries were exhausted.",
            details={
                "research_topic": state.get("research_topic", ""),
                "raw_notes_length": len(raw_notes_content),
            },
        ),
        "node_status": build_node_status(
            config,
            node="compress_research",
            status="error",
            summary="Could not compress subtask findings after retries.",
            details={"research_topic": state.get("research_topic", "")},
        ),
    }

# 研究员子图构建
# 创建面向特定主题的单研究员工作流
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# 添加研究员节点，用于研究执行与结果压缩
researcher_builder.add_node("researcher", researcher)                 # 研究员主逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)   # 研究结果压缩

# 定义研究员工作流边
researcher_builder.add_edge(START, "researcher")           # 研究员入口节点
researcher_builder.add_edge("compress_research", END)      # 压缩后退出节点

# 编译研究员子图，供监督者并行调用
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终综合研究报告，并包含令牌上限重试机制。
    
    该函数会汇总所有研究结果，并使用配置的报告模型生成
    结构完整、内容全面的最终报告。
    
    参数:
        state: 智能体状态，包含研究结果与上下文
        config: 运行时配置，包含模型设置与 API Key
        
    返回:
        包含最终报告与状态清理信息的字典
    """
    # 第 1 步：提取研究结论并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # 第 2 步：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 3 步：生成报告，并在令牌上限场景下重试
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # 构造包含全部研究上下文的综合提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                domain_profile_prompt=get_domain_profile_prompt(configurable.research_profile),
                user_guidance_context=get_user_guidance_context(state, config),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 返回报告生成成功结果
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                "trace_events": build_trace_event(
                    config,
                    node="final_report_generation",
                    event_type="completed",
                    summary="Final report generated.",
                    details={
                        "findings_length": len(findings),
                        "report_length": len(str(final_report.content)),
                    },
                ),
                "node_status": build_node_status(
                    config,
                    node="final_report_generation",
                    status="completed",
                    summary="Final report generated.",
                    details={"report_length": len(str(final_report.content))},
                ),
                **cleared_state
            }
            
        except Exception as e:
            # 处理令牌上限错误：逐步截断内容重试
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # 第一次重试：确定初始截断上限
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            "trace_events": build_trace_event(
                                config,
                                node="final_report_generation",
                                event_type="error",
                                summary="Final report failed because token limit metadata is missing.",
                                details={"error": str(e)},
                            ),
                            "node_status": build_node_status(
                                config,
                                node="final_report_generation",
                                status="error",
                                summary="Token limit exceeded and model limit is unknown.",
                                details={"error": str(e)},
                            ),
                            **cleared_state
                        }
                    # 使用 token 上限的 4 倍作为字符数近似截断阈值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次再减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # 截断研究内容后重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非令牌上限错误：立即返回错误
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    "trace_events": build_trace_event(
                        config,
                        node="final_report_generation",
                        event_type="error",
                        summary="Final report generation failed.",
                        details={"error": str(e)},
                    ),
                    "node_status": build_node_status(
                        config,
                        node="final_report_generation",
                        status="error",
                        summary="Final report generation failed.",
                        details={"error": str(e)},
                    ),
                    **cleared_state
                }
    
    # 第 4 步：若重试全部失败，返回失败结果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        "trace_events": build_trace_event(
            config,
            node="final_report_generation",
            event_type="error",
            summary="Final report retries were exhausted.",
        ),
        "node_status": build_node_status(
            config,
            node="final_report_generation",
            status="error",
            summary="Final report retries exhausted.",
        ),
        **cleared_state
    }

# 主 Deep Researcher 图构建
# 创建从用户输入到最终报告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# 添加主工作流节点，覆盖完整研究流程
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段

# 定义主工作流边，按顺序执行
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # 入口节点
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # 研究到报告
deep_researcher_builder.add_edge("final_report_generation", END)                   # 最终退出节点

# 编译完整的 deep researcher 工作流
deep_researcher = deep_researcher_builder.compile()
