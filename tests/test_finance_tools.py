import pytest

from open_deep_research import utils as research_utils
from open_deep_research.configuration import ResearchProfile, SearchAPI


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
