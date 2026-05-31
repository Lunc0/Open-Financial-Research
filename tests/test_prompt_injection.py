from open_deep_research.deep_researcher import extract_source_lines_from_text
from open_deep_research.domain_profiles import FINANCE_PROFILE_PROMPT
from open_deep_research.prompts import (
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)


def test_research_topic_prompt_contains_domain_profile_placeholder():
    assert "{domain_profile_prompt}" in transform_messages_into_research_topic_prompt
    rendered = transform_messages_into_research_topic_prompt.format(
        messages="用户需要研究一家公司的现金流。",
        date="2026-05-30",
        domain_profile_prompt="使用金融研究标准。",
    )
    assert "使用金融研究标准。" in rendered


def test_lead_researcher_prompt_contains_domain_profile_placeholder():
    assert "{domain_profile_prompt}" in lead_researcher_prompt
    rendered = lead_researcher_prompt.format(
        date="2026-05-30",
        domain_profile_prompt="使用金融研究标准。",
        max_researcher_iterations=3,
        max_concurrent_research_units=2,
    )
    assert "使用金融研究标准。" in rendered
    assert "UpdateResearchPlan" in rendered
    assert "plan_item_id" in rendered


def test_research_system_prompt_contains_profile_and_guidance_placeholders():
    assert "{domain_profile_prompt}" in research_system_prompt
    assert "{user_guidance_context}" in research_system_prompt
    rendered = research_system_prompt.format(
        date="2026-05-30",
        domain_profile_prompt="使用金融研究标准。",
        user_guidance_context="重点关注现金流。",
        mcp_prompt="",
    )
    assert "使用金融研究标准。" in rendered
    assert "重点关注现金流。" in rendered


def test_finance_profile_prompt_uses_search_and_mcp_collaboratively():
    rendered = research_system_prompt.format(
        date="2026-05-31",
        domain_profile_prompt=FINANCE_PROFILE_PROMPT,
        user_guidance_context="",
        mcp_prompt="可用 MCP 工具由运行时配置提供。",
    )

    assert "协同工作" in rendered
    assert "搜索发现线索" in rendered
    assert "用 MCP 获取" in rendered
    assert "应与搜索工具协同使用" in rendered
    assert "Sources" in rendered
    assert "来源标题" in rendered
    assert "URL" in rendered
    assert "来源机构" in rendered
    assert "工具名" in rendered
    assert "报告期" in rendered
    assert "币种" in rendered
    assert "口径" in rendered
    assert "检索时间" in rendered
    assert "evidence_items" not in rendered


def test_final_report_prompt_contains_profile_and_guidance_placeholders():
    assert "{domain_profile_prompt}" in final_report_generation_prompt
    assert "{user_guidance_context}" in final_report_generation_prompt
    rendered = final_report_generation_prompt.format(
        research_brief="研究一家公司的盈利质量。",
        domain_profile_prompt="使用金融研究标准。",
        user_guidance_context="重点关注现金流。",
        messages="用户要求输出中文报告。",
        date="2026-05-30",
        findings="发现一：现金流改善。",
    )
    assert "使用金融研究标准。" in rendered
    assert "重点关注现金流。" in rendered
    assert "发现一：现金流改善。" in rendered
    assert "不要编造未在研究材料中出现的 URL" in rendered


def test_compress_research_prompt_requires_lightweight_sources():
    rendered = compress_research_system_prompt.format(date="2026-05-31")

    assert "Findings:" in rendered
    assert "Sources:" in rendered
    assert "全部相关发现" in rendered
    assert "Sources 最多 10 条" in rendered
    assert "不要复制网页正文" in rendered
    assert "不允许编造 URL" in rendered
    assert "不要输出 Limitations 独立章节" in rendered


def test_extract_source_lines_from_english_sources_section():
    text = """Findings:
- 电车保养成本较低 [S1]。

Sources:
- [S1] AAA ownership cost report | https://example.com/aaa | tavily_search
- [S1] AAA ownership cost report | https://example.com/aaa | tavily_search
- [S2] KBB resale analysis | https://example.com/kbb | tavily_search

Next Section:
- 不应被提取
"""

    assert extract_source_lines_from_text(text) == [
        "[S1] AAA ownership cost report | https://example.com/aaa | tavily_search",
        "[S2] KBB resale analysis | https://example.com/kbb | tavily_search",
    ]


def test_extract_source_lines_from_chinese_sources_section_with_limits():
    long_source = "很长的来源标题" * 80
    text = "\n".join(
        ["Findings:", "来源："]
        + [f"- 来源 {index} | https://example.com/{index}" for index in range(12)]
        + [f"- {long_source}"]
    )

    sources = extract_source_lines_from_text(text, max_sources=10)

    assert len(sources) == 10
    assert sources[0] == "来源 0 | https://example.com/0"
    assert all(len(source) <= 303 for source in sources)


def test_extract_source_lines_returns_empty_when_no_sources_section():
    assert extract_source_lines_from_text("Findings:\n- 没有来源段。") == []


def test_extract_source_lines_ignores_no_source_placeholder():
    assert extract_source_lines_from_text(
        "Findings:\n- 暂无可追溯来源。\n\nSources:\n未识别到可追溯来源"
    ) == []
    assert extract_source_lines_from_text(
        "Findings:\n- No sourced result.\n\nSources:\nNo traceable sources."
    ) == []
