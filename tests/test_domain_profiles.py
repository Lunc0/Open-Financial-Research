from open_deep_research.configuration import ResearchProfile
from open_deep_research.domain_profiles import (
    FINANCE_PROFILE_PROMPT,
    GENERAL_PROFILE_PROMPT,
    get_domain_profile_prompt,
)


def test_general_profile_returns_general_prompt():
    assert get_domain_profile_prompt(ResearchProfile.GENERAL) == GENERAL_PROFILE_PROMPT


def test_finance_profile_returns_finance_prompt_for_enum_and_string():
    assert get_domain_profile_prompt(ResearchProfile.FINANCE) == FINANCE_PROFILE_PROMPT
    assert get_domain_profile_prompt("finance") == FINANCE_PROFILE_PROMPT


def test_unknown_profile_falls_back_to_general_prompt():
    assert get_domain_profile_prompt("unknown") == GENERAL_PROFILE_PROMPT
