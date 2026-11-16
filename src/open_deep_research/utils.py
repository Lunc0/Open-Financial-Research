"""Deep Research 智能体的工具函数与辅助逻辑。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import (
    Configuration,
    MCPConfig,
    ResearchProfile,
    SearchAPI,
)
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

MCP_HTTP_TRANSPORTS = {"streamable_http", "http", "sse"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

##########################
# Tavily 搜索工具辅助函数
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """从 Tavily 搜索 API 获取并总结搜索结果。

    参数:
        queries: 要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数
        topic: 搜索结果的主题过滤器（general、news 或 finance）
        config: 运行时配置，包含 API Key 和模型设置

    返回:
        包含摘要搜索结果的格式化字符串
    """
    # 第 1 步：异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # 第 2 步：按 URL 去重，避免重复处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # 第 3 步：根据配置初始化摘要模型
    configurable = Configuration.from_runnable_config(config)

    # 字符上限，用于控制在模型 token 限制内（可配置）
    max_char_to_include = configurable.max_content_length
    
    # 初始化带重试逻辑的摘要模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # 第 4 步：创建摘要任务（跳过空内容）
    async def noop():
        """用于无原始内容结果的空操作函数。"""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # 第 5 步：并行执行所有摘要任务
    summaries = await asyncio.gather(*summarization_tasks)
    
    # 第 6 步：将结果与对应摘要合并
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # 第 7 步：格式化最终输出
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """异步执行多个 Tavily 搜索查询。
    
    参数:
        search_queries: 要执行的搜索查询字符串列表
        max_results: 每个查询的最大结果数
        topic: 结果过滤的主题类别
        include_raw_content: 是否包含网页完整内容
        config: 用于获取 API Key 的运行时配置
        
    返回:
        来自 Tavily API 的搜索结果字典列表
    """
    # 使用配置中的 API Key 初始化 Tavily 客户端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # 创建并行执行的搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # 并行执行所有搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型对网页内容进行摘要，并带有超时保护。
    
    参数:
        model: 用于摘要的聊天模型
        webpage_content: 待摘要的网页原始内容
        
    返回:
        带关键摘录的格式化摘要；若摘要失败则返回原始内容
    """
    try:
        # 构造带当前日期上下文的提示词
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # 执行带超时限制的摘要，避免任务卡住
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 摘要超时时间为 60 秒
        )
        
        # 将摘要整理为结构化格式
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # 摘要超时，返回原始内容
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # 其他摘要错误，记录日志并返回原始内容
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# 反思工具辅助函数
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """用于研究进展与决策的策略反思工具。

    在每次搜索后使用该工具，系统性分析结果并规划下一步。
    这会在研究流程中加入一次有意识的停顿，以提升决策质量。

    使用时机：
    - 收到搜索结果后：我找到了哪些关键信息？
    - 决定下一步之前：现有信息是否足够完整回答问题？
    - 评估研究缺口时：我还缺哪些具体信息？
    - 准备结束研究前：我现在是否能给出完整答案？

    反思内容应包括：
    1. 当前发现分析：我已经掌握了哪些具体信息？
    2. 缺口评估：还缺少哪些关键信息？
    3. 质量评估：我是否已有足够证据或例子支撑一个好答案？
    4. 策略决策：应继续搜索还是直接给出答案？

    参数:
        reflection: 你对研究进展、发现、缺口和下一步的详细反思

    返回:
        一条确认反思已被记录、可用于决策的信息
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP 辅助函数
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """通过 OAuth token exchange 将 Supabase token 换成 MCP access token。
    
    参数:
        supabase_token: 有效的 Supabase 身份验证 token
        base_mcp_url: MCP 服务器的基础 URL
        
    返回:
        成功时返回 token 数据字典，失败时返回 None
    """
    try:
        # 准备 OAuth token exchange 请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # 执行 token exchange 请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取 token
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误细节，便于排查
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """读取已存储的认证 token，并校验其是否过期。
    
    参数:
        config: 包含线程和用户标识的运行时配置
        
    返回:
        若 token 有效且未过期则返回 token 字典，否则返回 None
    """
    store = get_store()
    
    # 从配置中提取所需标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # 读取已存储的 token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # 检查 token 是否过期
    expires_in = tokens.value.get("expires_in")  # 距离过期的秒数
    created_at = tokens.created_at  # token 创建时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # token 已过期，清理后返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """将认证 token 存入配置存储。
    
    参数:
        config: 包含线程和用户标识的运行时配置
        tokens: 要存储的 token 字典
    """
    store = get_store()
    
    # 从配置中提取所需标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # 存储 token
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP token，必要时申请新 token。
    
    参数:
        config: 含认证信息的运行时配置
        
    返回:
        有效的 token 字典；若无法获取则返回 None
    """
    # 优先尝试获取已有且仍有效的 token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # 提取用于换取新 token 的 Supabase token
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if hasattr(mcp_config, "model_dump"):
        mcp_config = mcp_config.model_dump()
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # 用 Supabase token 换取 MCP token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储新 token 并返回
    await set_tokens(config, mcp_tokens)
    return mcp_tokens


def normalize_mcp_transport(transport: str | None) -> str:
    """Normalize MCP transport aliases accepted by project configuration."""
    normalized_transport = str(transport or "streamable_http")
    if normalized_transport == "http":
        return "streamable_http"
    return normalized_transport


def normalize_mcp_server_url(
    raw_url: str,
    *,
    transport: str = "streamable_http",
    server_path: str | None = None,
) -> str:
    """Build a usable MCP endpoint URL without appending /mcp twice."""
    if transport not in MCP_HTTP_TRANSPORTS:
        return raw_url

    endpoint_path = server_path if server_path is not None else "/mcp"
    if not endpoint_path:
        return raw_url

    parsed_url = urlparse(raw_url)
    current_path = parsed_url.path.rstrip("/")
    normalized_endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
    normalized_endpoint_path = normalized_endpoint_path.rstrip("/")

    if current_path and current_path != "/":
        return raw_url.rstrip("/")
    return raw_url.rstrip("/") + normalized_endpoint_path


def build_mcp_headers(
    configured_headers: dict[str, str] | None = None,
    *,
    access_token_env: str | None = None,
    mcp_tokens: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Merge static MCP headers with runtime bearer-token sources."""
    headers = dict(configured_headers or {})
    env_token = os.getenv(access_token_env) if access_token_env else None
    access_token = env_token or (mcp_tokens or {}).get("access_token")
    if access_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers or None


def has_mcp_auth_material(
    headers: dict[str, str] | None = None,
    *,
    access_token_env: str | None = None,
    mcp_tokens: dict[str, Any] | None = None,
) -> bool:
    """Return whether MCP auth_required has usable auth material."""
    if (mcp_tokens or {}).get("access_token"):
        return True
    if access_token_env and os.getenv(access_token_env):
        return True
    return bool((headers or {}).get("Authorization"))


def has_mcp_config_auth_material(
    mcp_config,
    *,
    mcp_tokens: dict[str, Any] | None = None,
) -> bool:
    """Return whether any configured MCP auth source is available."""
    if has_mcp_auth_material(
        mcp_config.headers,
        access_token_env=mcp_config.access_token_env,
        mcp_tokens=mcp_tokens,
    ):
        return True
    for server_config in (mcp_config.servers or {}).values():
        if not isinstance(server_config, dict):
            continue
        if has_mcp_auth_material(
            server_config.get("headers"),
            access_token_env=server_config.get("access_token_env"),
            mcp_tokens=mcp_tokens,
        ):
            return True
    return False


def build_mcp_server_config(
    mcp_config,
    *,
    mcp_tokens: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build MultiServerMCPClient connection config from project MCP settings."""
    if not mcp_config:
        return {}

    if mcp_config.servers:
        server_config = {}
        for server_name, raw_server_config in mcp_config.servers.items():
            if not raw_server_config:
                continue
            server_entry = dict(raw_server_config)
            if not server_entry.get("transport") and server_entry.get("command") and not server_entry.get("url"):
                server_entry["transport"] = "stdio"
            else:
                server_entry["transport"] = normalize_mcp_transport(server_entry.get("transport"))
            if "url" in server_entry:
                server_entry["url"] = normalize_mcp_server_url(
                    str(server_entry["url"]),
                    transport=server_entry["transport"],
                    server_path=server_entry.pop("server_path", None),
                )
                server_entry["headers"] = build_mcp_headers(
                    server_entry.get("headers"),
                    access_token_env=server_entry.pop("access_token_env", None),
                    mcp_tokens=mcp_tokens,
                )
            server_config[str(server_name)] = {
                key: value for key, value in server_entry.items() if value is not None
            }
        return server_config

    if not mcp_config.url and not mcp_config.command:
        return {}

    transport = normalize_mcp_transport(mcp_config.transport)
    if mcp_config.command and not mcp_config.url and transport == "streamable_http":
        transport = "stdio"
    server_entry: dict[str, Any] = {"transport": transport}
    if transport == "stdio":
        if not mcp_config.command:
            return {}
        server_entry["command"] = mcp_config.command
        server_entry["args"] = mcp_config.args or []
        server_entry["env"] = mcp_config.env
        server_entry["cwd"] = mcp_config.cwd
    else:
        if not mcp_config.url:
            return {}
        server_entry["url"] = normalize_mcp_server_url(
            mcp_config.url,
            transport=transport,
            server_path=mcp_config.server_path,
        )
        server_entry["headers"] = build_mcp_headers(
            mcp_config.headers,
            access_token_env=mcp_config.access_token_env,
            mcp_tokens=mcp_tokens,
        )
    return {
        "server_1": {
            key: value for key, value in server_entry.items() if value is not None
        }
    }


def env_flag_enabled(name: str) -> bool:
    """Read a boolean feature flag from environment variables."""
    return os.getenv(name, "").strip().lower() in TRUE_ENV_VALUES


def get_effective_mcp_config(configurable: Configuration) -> MCPConfig | None:
    """Return explicit MCP config or the optional default SEC EDGAR MCP config."""
    if configurable.mcp_config:
        return configurable.mcp_config

    research_profile = ResearchProfile(get_config_value(configurable.research_profile))
    sec_edgar_enabled = (
        configurable.enable_sec_edgar_mcp
        or env_flag_enabled("ENABLE_SEC_EDGAR_MCP")
    )
    if research_profile != ResearchProfile.FINANCE or not sec_edgar_enabled:
        return None

    edgar_identity = os.getenv(configurable.sec_edgar_identity_env)
    if not edgar_identity:
        warnings.warn(
            "SEC EDGAR MCP is enabled but EDGAR_IDENTITY is not set; "
            "skipping SEC EDGAR MCP tools."
        )
        return None

    return MCPConfig(
        transport="stdio",
        command=configurable.sec_edgar_mcp_command,
        args=configurable.sec_edgar_mcp_args,
        env={configurable.sec_edgar_identity_env: edgar_identity},
        tools=configurable.sec_edgar_mcp_tools,
    )


def wrap_mcp_authenticate_tool(tool: BaseTool) -> BaseTool:
    """为 MCP 工具包装认证与错误处理逻辑。
    
    参数:
        tool: 要包装的 MCP 结构化工具
        
    返回:
        增强后的工具，包含认证错误处理能力
    """
    original_coroutine = getattr(tool, "coroutine", None)
    if original_coroutine is None:
        return tool
    
    async def authentication_wrapper(**kwargs):
        """带 MCP 错误处理和用户友好提示的增强协程。"""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """递归地在异常链中查找 MCP 错误。"""
            if isinstance(exc, McpError):
                return exc
            
            # 通过检查属性处理 ExceptionGroup（Python 3.11+）
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # 执行原始工具逻辑
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # 在异常链中查找 MCP 特有错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 错误，直接重新抛出原始异常
                raise original_error
            
            # 处理 MCP 特有的错误场景
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # 检查是否为需要认证/用户交互的错误
            if error_code == -32003:  # 需要交互的错误码
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # 如果有更友好的提示文本，则优先使用
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # 如果有 URL，则附加到消息中供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # 其他 MCP 错误仍然抛出原始异常
            raise original_error
    
    # 用增强版本替换工具原有协程
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置带认证能力的 MCP（Model Context Protocol）工具。
    
    参数:
        config: 包含 MCP 服务器详情的运行时配置
        existing_tool_names: 已在使用的工具名集合，用于避免冲突
        
    返回:
        已配置完成、可直接使用的 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_config = get_effective_mcp_config(configurable)

    # 第 1 步：如有需要，先处理认证
    if mcp_config and mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # 第 2 步：校验配置是否满足要求
    config_valid = mcp_config is not None
    
    if not config_valid:
        return []

    auth_valid = (
        not mcp_config.auth_required
        or has_mcp_config_auth_material(
            mcp_config,
            mcp_tokens=mcp_tokens,
        )
    )
    if not auth_valid:
        return []
    
    # 第 3 步：建立 MCP 服务器连接配置
    mcp_server_config = build_mcp_server_config(
        mcp_config,
        mcp_tokens=mcp_tokens,
    )
    if not mcp_server_config:
        return []

    # 第 4 步：从 MCP 服务器加载工具
    try:
        client = MultiServerMCPClient(
            mcp_server_config,
            tool_name_prefix=mcp_config.tool_name_prefix,
        )
        available_mcp_tools = await client.get_tools()
    except Exception:
        # MCP 服务器连接失败时返回空列表
        return []
    
    # 第 5 步：筛选并配置工具
    configured_tools = []
    requested_tool_names = set(mcp_config.tools or [])
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # 仅保留配置中明确指定的工具
        if requested_tool_names and mcp_tool.name not in requested_tool_names:
            continue
        
        # 为工具包装认证处理后加入结果列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# 工具装配辅助函数
##########################


def get_tool_name(available_tool: BaseTool | dict[str, Any]) -> str:
    """返回工具名，并兼容模型提供方原生工具字典。"""
    if hasattr(available_tool, "name"):
        return available_tool.name
    return available_tool.get("name", "web_search")


def extend_tools_without_conflicts(
    tools: list[BaseTool | dict[str, Any]],
    incoming_tools: list[BaseTool | dict[str, Any]],
) -> None:
    """追加工具，同时跳过已经存在的工具名。"""
    existing_tool_names = {get_tool_name(existing_tool) for existing_tool in tools}
    for incoming_tool in incoming_tools:
        incoming_tool_name = get_tool_name(incoming_tool)
        if incoming_tool_name in existing_tool_names:
            warnings.warn(
                f"Tool '{incoming_tool_name}' conflicts with an existing tool name - skipping"
            )
            continue
        tools.append(incoming_tool)
        existing_tool_names.add(incoming_tool_name)


async def load_mcp_tools_safely(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载 MCP 工具；失败时返回空列表以保留搜索兜底。"""
    try:
        return await load_mcp_tools(config, existing_tool_names)
    except Exception as error:
        warnings.warn(f"MCP tools could not be loaded - falling back to search tools: {error}")
        return []


async def get_search_tool(search_api: SearchAPI):
    """根据指定的 API 提供方配置并返回搜索工具。
    
    参数:
        search_api: 要使用的搜索 API 提供方（Anthropic、OpenAI、Tavily 或 None）
        
    返回:
        为指定提供方配置好的搜索工具对象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic 原生网页搜索，带使用次数限制
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI 的网页搜索预览功能
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # 为 Tavily 搜索工具补充元数据
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # 不配置搜索功能
        return []
        
    # 未知搜索 API 类型的默认兜底
    return []
    
async def get_all_tools(config: RunnableConfig):
    """组装完整工具集，包含研究、搜索和 MCP 工具。
    
    参数:
        config: 指定搜索 API 和 MCP 设置的运行时配置
        
    返回:
        研究流程可用的全部已配置工具列表
    """
    # 先加入核心研究工具
    tools = [tool(ResearchComplete), think_tool]

    configurable = Configuration.from_runnable_config(config)
    research_profile = ResearchProfile(get_config_value(configurable.research_profile))
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)

    if research_profile == ResearchProfile.FINANCE:
        # 金融研究同时暴露 MCP 和通用搜索；保留 search 名称，避免 MCP 覆盖通用搜索工具。
        existing_tool_names = {
            get_tool_name(existing_tool)
            for existing_tool in [*tools, *search_tools]
        }
        mcp_tools = await load_mcp_tools_safely(config, existing_tool_names)
        extend_tools_without_conflicts(tools, search_tools)
        extend_tools_without_conflicts(tools, mcp_tools)
        return tools

    # 通用研究保持原有行为：先通用搜索，再补充 MCP 工具。
    extend_tools_without_conflicts(tools, search_tools)
    existing_tool_names = {get_tool_name(existing_tool) for existing_tool in tools}
    mcp_tools = await load_mcp_tools_safely(config, existing_tool_names)
    extend_tools_without_conflicts(tools, mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取笔记内容。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型提供方原生 Web Search 辅助函数
##########################

def anthropic_websearch_called(response):
    """检测响应中是否使用了 Anthropic 的原生网页搜索。
    
    参数:
        response: Anthropic API 返回的响应对象
        
    返回:
        如果调用了网页搜索则返回 True，否则返回 False
    """
    try:
        # 逐层读取响应元数据结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # 检查服务端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # 查找网页搜索请求次数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # 只要存在网页搜索请求即返回 True
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # 处理响应结构不符合预期的情况
        return False

def openai_websearch_called(response):
    """检测响应中是否使用了 OpenAI 的网页搜索功能。
    
    参数:
        response: OpenAI API 返回的响应对象
        
    返回:
        如果调用了网页搜索则返回 True，否则返回 False
    """
    # 检查响应元数据中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # 在工具输出中查找网页搜索调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token 超限辅助函数
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否表示 token/上下文长度已超限。
    
    参数:
        exception: 要分析的异常对象
        model_name: 可选的模型名称，用于优化提供方识别
        
    返回:
        若异常表示 token 超限则返回 True，否则返回 False
    """
    error_str = str(exception).lower()
    
    # 第 1 步：如果提供了模型名，则先判断提供方
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # 第 2 步：检查提供方特有的 token 超限模式
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # 第 3 步：如果提供方未知，则检查所有提供方规则
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 OpenAI token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断这是否是 OpenAI 异常
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # 检查典型的 OpenAI token 超限错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # 在错误消息中查找与 token 相关的关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # 检查 OpenAI 特定错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Anthropic token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断这是否是 Anthropic 异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # 检查 Anthropic 特有的错误模式
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic 对 token 超限使用特定错误消息
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Google/Gemini token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断这是否是 Google/Gemini 异常
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # 检查 Google 特有的资源耗尽错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # 检查 Google API 特定的资源耗尽模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# 注意：这里的数据可能已经过时，或不适用于你当前使用的模型，请按需更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查询指定模型的 token 上限。
    
    参数:
        model_string: 要查询的模型标识字符串
        
    返回:
        若命中则返回整数形式的 token 上限，否则返回 None
    """
    # 在已知模型 token 上限表中查找
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # 查不到该模型时返回 None
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过删除最后一条 AI 消息之前的内容来截断消息历史。
    
    这对处理 token 超限错误很有用，因为它能移除最近的一段上下文。
    
    参数:
        messages: 需要截断的消息对象列表
        
    返回:
        截断后的消息列表，保留到最后一条 AI 消息之前（不含该消息）
    """
    # 从后往前查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回最后一条 AI 消息之前的所有内容（不含该消息）
            return messages[:i]
    
    # 未找到 AI 消息，返回原始列表
    return messages

##########################
# 杂项辅助函数
##########################

def get_today_str() -> str:
    """获取用于提示词和输出展示的当前日期字符串。
    
    返回:
        形如 “Mon Jan 15, 2024” 的可读日期字符串
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置值中提取实际值，并兼容枚举和 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置中获取指定模型的 API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置中获取 Tavily API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
