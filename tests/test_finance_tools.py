import pytest

from open_deep_research import utils as research_utils
from open_deep_research.configuration import Configuration, ResearchProfile, SearchAPI


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def tool_names(tools):
    return [research_utils.get_tool_name(tool) for tool in tools]


@pytest.mark.asyncio
async def test_finance_profile_exposes_search_and_mcp_tools(monkeypatch):
    async def fake_load_mcp_tools(_config, existing_tool_names):
        assert "ResearchComplete" in existing_tool_names
        assert "think_tool" in existing_tool_names
        assert "web_search" in existing_tool_names
        return [FakeTool("finance_mcp_filings"), FakeTool("finance_mcp_macro_data")]

    async def fake_get_search_tool(_search_api):
        return [{"name": "web_search"}]

    monkeypatch.setattr(research_utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(research_utils, "get_search_tool", fake_get_search_tool)

    tools = await research_utils.get_all_tools({
        "configurable": {
            "research_profile": ResearchProfile.FINANCE.value,
            "search_api": SearchAPI.TAVILY.value,
        }
    })
    names = tool_names(tools)

    assert "web_search" in names
    assert "finance_mcp_filings" in names
    assert "finance_mcp_macro_data" in names


@pytest.mark.asyncio
async def test_finance_profile_does_not_allow_mcp_to_shadow_search(monkeypatch):
    async def fake_load_mcp_tools(_config, existing_tool_names):
        return [
            tool
            for tool in [FakeTool("web_search"), FakeTool("finance_mcp_filings")]
            if tool.name not in existing_tool_names
        ]

    async def fake_get_search_tool(_search_api):
        return [{"name": "web_search"}]

    monkeypatch.setattr(research_utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(research_utils, "get_search_tool", fake_get_search_tool)

    tools = await research_utils.get_all_tools({
        "configurable": {
            "research_profile": ResearchProfile.FINANCE.value,
            "search_api": SearchAPI.TAVILY.value,
        }
    })
    names = tool_names(tools)

    assert names.count("web_search") == 1
    assert "finance_mcp_filings" in names


@pytest.mark.asyncio
async def test_general_profile_keeps_search_before_mcp(monkeypatch):
    async def fake_load_mcp_tools(_config, existing_tool_names):
        assert "web_search" in existing_tool_names
        return [FakeTool("finance_mcp_filings")]

    async def fake_get_search_tool(_search_api):
        return [{"name": "web_search"}]

    monkeypatch.setattr(research_utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(research_utils, "get_search_tool", fake_get_search_tool)

    tools = await research_utils.get_all_tools({
        "configurable": {
            "research_profile": ResearchProfile.GENERAL.value,
            "search_api": SearchAPI.TAVILY.value,
        }
    })
    names = tool_names(tools)

    assert names.index("web_search") < names.index("finance_mcp_filings")


@pytest.mark.asyncio
async def test_finance_profile_falls_back_to_search_when_mcp_is_empty(monkeypatch):
    async def fake_load_mcp_tools(_config, _existing_tool_names):
        return []

    async def fake_get_search_tool(_search_api):
        return [{"name": "web_search"}]

    monkeypatch.setattr(research_utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(research_utils, "get_search_tool", fake_get_search_tool)

    tools = await research_utils.get_all_tools({
        "configurable": {
            "research_profile": ResearchProfile.FINANCE.value,
            "search_api": SearchAPI.TAVILY.value,
        }
    })

    assert "web_search" in tool_names(tools)


@pytest.mark.asyncio
async def test_finance_profile_falls_back_to_search_when_mcp_load_fails(monkeypatch):
    async def fake_load_mcp_tools(_config, _existing_tool_names):
        raise RuntimeError("placeholder MCP not configured")

    async def fake_get_search_tool(_search_api):
        return [{"name": "web_search"}]

    monkeypatch.setattr(research_utils, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(research_utils, "get_search_tool", fake_get_search_tool)

    with pytest.warns(UserWarning, match="falling back to search tools"):
        tools = await research_utils.get_all_tools({
            "configurable": {
                "research_profile": ResearchProfile.FINANCE.value,
                "search_api": SearchAPI.TAVILY.value,
            }
        })

    assert "web_search" in tool_names(tools)


def test_build_mcp_server_config_does_not_append_mcp_twice(monkeypatch):
    monkeypatch.setenv("FINANCE_MCP_TOKEN", "test-token")
    configurable = Configuration.from_runnable_config({
        "configurable": {
            "mcp_config": {
                "url": "http://localhost:8000/mcp",
                "headers": {"X-Client": "open-deep-research"},
                "access_token_env": "FINANCE_MCP_TOKEN",
            }
        }
    })

    server_config = research_utils.build_mcp_server_config(configurable.mcp_config)

    assert server_config == {
        "server_1": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "X-Client": "open-deep-research",
                "Authorization": "Bearer test-token",
            },
        }
    }


def test_build_mcp_server_config_supports_multi_server_configs():
    configurable = Configuration.from_runnable_config({
        "configurable": {
            "mcp_config": {
                "servers": {
                    "filings": {
                        "url": "https://finance.example.com",
                        "transport": "http",
                    },
                    "local_macro": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                    },
                }
            }
        }
    })

    server_config = research_utils.build_mcp_server_config(configurable.mcp_config)

    assert server_config["filings"]["transport"] == "streamable_http"
    assert server_config["filings"]["url"] == "https://finance.example.com/mcp"
    assert server_config["local_macro"] == {
        "transport": "stdio",
        "command": "python",
        "args": ["server.py"],
    }


@pytest.mark.asyncio
async def test_load_mcp_tools_allows_empty_tool_whitelist(monkeypatch):
    class FakeMCPClient:
        last_connections = None
        last_tool_name_prefix = None

        def __init__(self, connections, *, tool_name_prefix=False):
            self.__class__.last_connections = connections
            self.__class__.last_tool_name_prefix = tool_name_prefix

        async def get_tools(self):
            return [FakeTool("finance_filings"), FakeTool("macro_data")]

    monkeypatch.setattr(research_utils, "MultiServerMCPClient", FakeMCPClient)

    tools = await research_utils.load_mcp_tools(
        {
            "configurable": {
                "mcp_config": {
                    "url": "http://localhost:8000",
                    "tool_name_prefix": True,
                }
            }
        },
        existing_tool_names=set(),
    )

    assert tool_names(tools) == ["finance_filings", "macro_data"]
    assert FakeMCPClient.last_connections["server_1"]["url"] == "http://localhost:8000/mcp"
    assert FakeMCPClient.last_tool_name_prefix is True


@pytest.mark.asyncio
async def test_load_mcp_tools_still_filters_when_whitelist_is_configured(monkeypatch):
    class FakeMCPClient:
        def __init__(self, connections, *, tool_name_prefix=False):
            pass

        async def get_tools(self):
            return [FakeTool("finance_filings"), FakeTool("macro_data")]

    monkeypatch.setattr(research_utils, "MultiServerMCPClient", FakeMCPClient)

    tools = await research_utils.load_mcp_tools(
        {
            "configurable": {
                "mcp_config": {
                    "url": "http://localhost:8000",
                    "tools": ["macro_data"],
                }
            }
        },
        existing_tool_names=set(),
    )

    assert tool_names(tools) == ["macro_data"]


@pytest.mark.asyncio
async def test_finance_profile_can_auto_attach_sec_edgar_mcp(monkeypatch):
    class FakeMCPClient:
        last_connections = None

        def __init__(self, connections, *, tool_name_prefix=False):
            self.__class__.last_connections = connections

        async def get_tools(self):
            return [FakeTool("edgar_company"), FakeTool("edgar_search")]

    monkeypatch.setenv("EDGAR_IDENTITY", "Open Deep Research tests test@example.com")
    monkeypatch.setattr(research_utils, "MultiServerMCPClient", FakeMCPClient)

    tools = await research_utils.load_mcp_tools(
        {
            "configurable": {
                "research_profile": ResearchProfile.FINANCE.value,
                "enable_sec_edgar_mcp": True,
            }
        },
        existing_tool_names={"web_search"},
    )

    assert tool_names(tools) == ["edgar_company", "edgar_search"]
    assert FakeMCPClient.last_connections == {
        "server_1": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["--from", "edgartools[ai]", "edgartools-mcp"],
            "env": {"EDGAR_IDENTITY": "Open Deep Research tests test@example.com"},
        }
    }


@pytest.mark.asyncio
async def test_sec_edgar_mcp_requires_identity(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    with pytest.warns(UserWarning, match="EDGAR_IDENTITY is not set"):
        tools = await research_utils.load_mcp_tools(
            {
                "configurable": {
                    "research_profile": ResearchProfile.FINANCE.value,
                    "enable_sec_edgar_mcp": True,
                }
            },
            existing_tool_names=set(),
        )

    assert tools == []
