"""领域研究模式的提示词配置。"""

from open_deep_research.configuration import ResearchProfile


GENERAL_PROFILE_PROMPT = """使用通用深度研究标准：优先选择权威、一手、可核验来源；区分事实、观点与不确定性；在结论中保留来源链路。"""

FINANCE_PROFILE_PROMPT = """使用金融研究标准：
- 优先使用一手或高可信来源：监管公告、交易所公告、公司财报与年报、招股书、投资者关系材料、央行与统计机构数据、评级机构报告、主流财经数据源和原始论文。
- 对股票、债券、基金、宏观、行业和公司研究，必须区分历史事实、市场一致预期、模型假设和主观判断。
- 明确时间点、报告期、币种、会计口径、同比/环比、名义/实际、合并/母公司、经调整/GAAP 等口径差异。
- 关注估值、盈利质量、现金流、资产负债表、杠杆、流动性、监管、竞争格局、周期位置、风险情景和潜在催化剂。
- 避免把新闻标题当作结论；重要价格、财务指标、政策条款和公司行动必须能追溯到来源。
- 报告需要给出关键假设、证据强弱、主要不确定性和可验证的后续跟踪指标。
- 不提供个性化投资建议；若涉及投资判断，应以研究分析和风险披露方式表达。"""


def get_domain_profile_prompt(profile: ResearchProfile | str) -> str:
    """根据研究模式返回对应的提示词指令。"""
    profile_value = profile.value if isinstance(profile, ResearchProfile) else str(profile)
    if profile_value == ResearchProfile.FINANCE.value:
        return FINANCE_PROFILE_PROMPT
    return GENERAL_PROFILE_PROMPT
