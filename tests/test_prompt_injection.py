from open_deep_research.prompts import (
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
