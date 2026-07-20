"""Tests for hard-gate logic (gate.py)."""
from __future__ import annotations

import pytest

from src.gate import _check_location, _check_seniority, _levels_in_text, _matches_region, check
from tests.conftest import make_posting


# ---------------------------------------------------------------------------
# Location — pass cases
# ---------------------------------------------------------------------------

def test_unstated_location_passes(profile):
    result = check(make_posting(location=None), profile)
    assert result.location_pass

def test_remote_passes(profile):
    result = check(make_posting(location="Remote"), profile)
    assert result.location_pass

def test_fully_remote_passes(profile):
    result = check(make_posting(location="Fully Remote (global)"), profile)
    assert result.location_pass

def test_anywhere_passes(profile):
    result = check(make_posting(location="Anywhere"), profile)
    assert result.location_pass

def test_sweden_hybrid_passes(profile):
    result = check(make_posting(location="Stockholm, Sweden (hybrid)"), profile)
    assert result.location_pass

def test_sundsvall_onsite_passes(profile):
    result = check(make_posting(location="Sundsvall, on-site"), profile)
    assert result.location_pass


# ---------------------------------------------------------------------------
# Location — fail cases
# ---------------------------------------------------------------------------

def test_london_onsite_fails(profile):
    result = check(make_posting(location="London, UK (on-site)"), profile)
    assert not result.location_pass
    assert not result.passed

def test_us_only_fails(profile):
    result = check(make_posting(location="New York, USA"), profile)
    assert not result.location_pass

def test_nairobi_fails(profile):
    result = check(make_posting(location="Nairobi, Kenya (on-site required)"), profile)
    assert not result.location_pass


# ---------------------------------------------------------------------------
# Per-user location constraints — _check_location / _matches_region
# (country-agnostic: no hardcoded Sweden fallback any more — a posting must match
# accept_fully_remote / accept_hybrid_in / accept_onsite_in, or be unstated, to pass)
# ---------------------------------------------------------------------------

def test_matches_region_word_boundary_us_does_not_match_russia():
    assert _matches_region("moscow, russia", ["US"]) is False

def test_matches_region_word_boundary_uk_does_not_match_fukuoka():
    assert _matches_region("fukuoka, japan", ["UK"]) is False

def test_matches_region_matches_standalone_token():
    assert _matches_region("austin, us", ["US"]) is True

def test_matches_region_empty_tokens_never_matches():
    assert _matches_region("berlin, germany", []) is False

def test_matches_region_skips_falsy_tokens():
    assert _matches_region("berlin, germany", ["", None, "germany"]) is True


def test_check_location_hybrid_region_match_passes():
    assert _check_location(
        "Berlin, Germany (hybrid)",
        {"accept_fully_remote": False, "accept_hybrid_in": ["Germany"], "accept_onsite_in": []},
    ) is True

def test_check_location_hybrid_without_region_match_fails():
    assert _check_location(
        "Berlin, Germany (hybrid)",
        {"accept_fully_remote": False, "accept_hybrid_in": ["France"], "accept_onsite_in": []},
    ) is False

def test_check_location_hybrid_region_match_without_hybrid_keyword_fails():
    """Onsite region tokens alone don't satisfy the hybrid branch — "hybrid" must
    literally appear in the posting text."""
    assert _check_location(
        "Berlin, Germany",
        {"accept_fully_remote": False, "accept_hybrid_in": ["Germany"], "accept_onsite_in": []},
    ) is False

def test_check_location_onsite_region_match_passes_without_hybrid_keyword():
    assert _check_location(
        "On-site in Berlin",
        {"accept_fully_remote": False, "accept_hybrid_in": [], "accept_onsite_in": ["Berlin"]},
    ) is True

def test_check_location_empty_constraint_lists_fail_stated_non_remote_posting():
    """No remote flag, no hybrid/onsite regions selected → a stated, non-remote
    location fails. Bias-toward-false-positives only covers UNSTATED fields, not an
    explicit location the user has no matching accept condition for — this is the
    replacement for the old Sweden-mention catch-all, which is now deleted."""
    assert _check_location(
        "Berlin, Germany",
        {"accept_fully_remote": False, "accept_hybrid_in": [], "accept_onsite_in": []},
    ) is False

def test_check_location_empty_constraints_dict_fails_stated_posting():
    assert _check_location("Berlin, Germany", {}) is False

def test_check_location_word_boundary_us_does_not_match_russia():
    """Regression: a US-based user's accept_onsite_in=["US"] must not false-match
    "Russia" via raw substring containment."""
    assert _check_location(
        "Moscow, Russia",
        {"accept_fully_remote": False, "accept_hybrid_in": [], "accept_onsite_in": ["US"]},
    ) is False

def test_check_location_word_boundary_us_matches_standalone_token():
    assert _check_location(
        "Austin, US",
        {"accept_fully_remote": False, "accept_hybrid_in": [], "accept_onsite_in": ["US"]},
    ) is True


# ---------------------------------------------------------------------------
# Seniority — pass cases
# ---------------------------------------------------------------------------

def test_unstated_seniority_passes(profile):
    result = check(make_posting(seniority=None), profile)
    assert result.seniority_pass

def test_senior_passes(profile):
    result = check(make_posting(seniority="Senior Data Analyst, 7+ years"), profile)
    assert result.seniority_pass

def test_director_passes(profile):
    result = check(make_posting(seniority="Director level, 10+ years experience"), profile)
    assert result.seniority_pass


# ---------------------------------------------------------------------------
# Seniority — fail cases
# ---------------------------------------------------------------------------

def test_junior_fails(profile):
    result = check(make_posting(seniority="Junior Data Analyst"), profile)
    assert not result.seniority_pass
    assert not result.passed

def test_entry_level_fails(profile):
    result = check(make_posting(seniority="Entry-level position"), profile)
    assert not result.seniority_pass

def test_graduate_fails(profile):
    result = check(make_posting(seniority="Graduate Analyst Programme"), profile)
    assert not result.seniority_pass

def test_intern_fails(profile):
    result = check(make_posting(seniority="Internship — 6 months"), profile)
    assert not result.seniority_pass


# ---------------------------------------------------------------------------
# Per-user accept_levels — _check_seniority / _levels_in_text
# ---------------------------------------------------------------------------

def test_levels_in_text_maps_lead_to_senior():
    assert _levels_in_text("Lead Data Scientist") == {"senior"}


def test_levels_in_text_maps_principal_to_director():
    assert _levels_in_text("Principal Engineer") == {"director"}


def test_levels_in_text_maps_head_of_to_director():
    assert _levels_in_text("Head of Data") == {"director"}


def test_levels_in_text_maps_staff_to_senior():
    assert _levels_in_text("Staff Engineer") == {"senior"}


def test_levels_in_text_maps_vp_to_director():
    assert _levels_in_text("VP of Engineering") == {"director"}


def test_levels_in_text_word_boundary_lead_does_not_match_leadership():
    assert _levels_in_text("Leadership Coach") == set()


def test_levels_in_text_word_boundary_mid_does_not_match_amid():
    assert _levels_in_text("Working amid uncertainty") == set()


def test_levels_in_text_word_boundary_sr_matches_standalone_token():
    assert _levels_in_text("Sr Data Analyst") == {"senior"}


def test_levels_in_text_matches_mid_level_phrase():
    assert _levels_in_text("Mid-Level Analyst") == {"mid"}


def test_levels_in_text_can_match_multiple_buckets():
    assert _levels_in_text("Senior Director of Data") == {"senior", "director"}


def test_levels_in_text_unmappable_text_returns_empty_set():
    assert _levels_in_text("Data Analyst") == set()


def test_check_seniority_empty_accept_levels_is_unfiltered():
    """No selection = no filter, regardless of posting text."""
    assert _check_seniority("Junior Data Analyst", {"accept_levels": []}) is True
    assert _check_seniority("Junior Data Analyst", {}) is True


def test_check_seniority_unstated_posting_passes_regardless_of_accept_levels():
    assert _check_seniority(None, {"accept_levels": ["senior", "director"]}) is True


def test_check_seniority_unmappable_text_passes_regardless_of_accept_levels():
    assert _check_seniority("Data Analyst", {"accept_levels": ["senior", "director"]}) is True


def test_check_seniority_passes_when_matched_level_in_accept_levels():
    assert _check_seniority("Senior Data Analyst", {"accept_levels": ["senior", "director"]}) is True


def test_check_seniority_fails_when_matched_level_not_in_accept_levels():
    assert _check_seniority("Junior Data Analyst", {"accept_levels": ["senior", "director"]}) is False


def test_check_seniority_passes_when_any_matched_level_in_accept_levels():
    """A posting matching multiple buckets passes if ANY matched bucket is accepted."""
    assert _check_seniority("Senior Director of Data", {"accept_levels": ["mid"]}) is False
    assert _check_seniority("Senior Director of Data", {"accept_levels": ["director"]}) is True


def test_check_seniority_user_who_only_wants_entry_level_rejects_senior_roles():
    """Personalization cuts both ways — a user who only selected intern/junior
    should be gated OUT of senior/director postings, unlike the old fail-list gate
    which let everything through except its hardcoded junior/intern keywords."""
    assert _check_seniority("Senior Data Analyst, 7+ years", {"accept_levels": ["intern", "junior"]}) is False


# ---------------------------------------------------------------------------
# Combined gate result
# ---------------------------------------------------------------------------

def test_reason_populated_on_failure(profile):
    result = check(make_posting(location="London, UK", seniority="Junior"), profile)
    assert not result.passed
    assert result.reason is not None
    assert len(result.reason) > 0

def test_reason_none_on_pass(profile):
    result = check(make_posting(location="Remote", seniority="Senior, 8+ years"), profile)
    assert result.passed
    assert result.reason is None


# ---------------------------------------------------------------------------
# rejection_reason — diagnostic-only categorisation, does not affect passed/*_pass
# ---------------------------------------------------------------------------

def test_rejection_reason_location_when_only_location_fails(profile):
    result = check(make_posting(location="London, UK (on-site)", seniority="Senior, 8+ years"), profile)
    assert not result.location_pass
    assert result.seniority_pass
    assert result.rejection_reason == "location"


def test_rejection_reason_seniority_when_only_seniority_fails(profile):
    result = check(make_posting(location="Remote", seniority="Junior Data Analyst"), profile)
    assert result.location_pass
    assert not result.seniority_pass
    assert result.rejection_reason == "seniority"


def test_rejection_reason_hard_constraints_when_both_fail(profile):
    result = check(make_posting(location="London, UK", seniority="Junior"), profile)
    assert not result.location_pass
    assert not result.seniority_pass
    assert result.rejection_reason == "hard_constraints"


def test_rejection_reason_none_on_pass(profile):
    result = check(make_posting(location="Remote", seniority="Senior, 8+ years"), profile)
    assert result.passed
    assert result.rejection_reason is None
