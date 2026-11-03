"""Tests for src/gate_v2.py — the v2 hard-gate, evaluated against a jobs row's
structured enrichment instead of v1's raw posting text. Uses the same `profile`
fixture as test_gate.py (accept_fully_remote=True, accept_hybrid_in=[Sweden,
Sundsvall], accept_onsite_in=[Sundsvall], accept_levels=[mid, senior, director])
so the two gates' behavior is directly comparable."""
from __future__ import annotations

from src.gate_v2 import check


def _job(location_field=None, seniority=None, mode="unspecified", remote_scope="unspecified",
         eligible_regions=None, restriction_type=None, posting_type=None) -> dict:
    return {
        "location_field": location_field,
        "enrichment": {
            "seniority": seniority or [],
            "posting_type": posting_type,
            "work_arrangement": {
                "mode": mode, "remote_scope": remote_scope,
                "eligible_regions": eligible_regions or [], "restriction_type": restriction_type,
                "timezone_constraint": None,
            },
        },
    }


# ---------------------------------------------------------------------------
# Location — pass cases
# ---------------------------------------------------------------------------

def test_unstated_location_passes(profile):
    result = check(_job(location_field=None, mode="unspecified"), profile)
    assert result.location_pass


def test_remote_global_passes(profile):
    result = check(_job(location_field="Remote, Global", mode="remote", remote_scope="global"), profile)
    assert result.location_pass


def test_remote_unspecified_scope_passes(profile):
    """bias toward false-positives: remote with no stated scope restriction passes."""
    result = check(_job(location_field="Remote", mode="remote", remote_scope="unspecified"), profile)
    assert result.location_pass


def test_remote_region_restricted_matching_user_location_passes(profile):
    result = check(
        _job(location_field="Remote, Sweden", mode="remote", remote_scope="region_restricted",
             eligible_regions=["SE"]),
        profile,
    )
    assert result.location_pass


def test_sweden_hybrid_passes(profile):
    result = check(_job(location_field="Stockholm, Sweden (hybrid)", mode="hybrid"), profile)
    assert result.location_pass


def test_sundsvall_onsite_passes(profile):
    result = check(_job(location_field="Sundsvall, on-site", mode="onsite"), profile)
    assert result.location_pass


# ---------------------------------------------------------------------------
# Location — fail cases
# ---------------------------------------------------------------------------

def test_london_onsite_fails(profile):
    result = check(_job(location_field="London, UK (on-site)", mode="onsite"), profile)
    assert not result.location_pass
    assert not result.passed


def test_remote_region_restricted_non_matching_region_fails(profile):
    """The key behavior difference from v1: v1's keyword substring match can't see
    a region restriction on a "Remote" posting at all and would pass it; v2 can
    and correctly fails it when the restriction doesn't cover the user."""
    result = check(
        _job(location_field="Remote, US only", mode="remote", remote_scope="region_restricted",
             eligible_regions=["US"]),
        profile,
    )
    assert not result.location_pass


def test_nairobi_onsite_fails(profile):
    result = check(_job(location_field="Nairobi, Kenya", mode="onsite"), profile)
    assert not result.location_pass


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

def test_unstated_seniority_passes(profile):
    result = check(_job(location_field="Remote", mode="remote", remote_scope="global", seniority=[]), profile)
    assert result.seniority_pass


def test_matching_seniority_passes(profile):
    result = check(_job(location_field="Remote", mode="remote", remote_scope="global", seniority=["senior"]), profile)
    assert result.seniority_pass


def test_non_matching_seniority_fails(profile):
    result = check(_job(location_field="Remote", mode="remote", remote_scope="global", seniority=["intern"]), profile)
    assert not result.seniority_pass
    assert result.rejection_reason == "seniority"


def test_multi_bucket_seniority_passes_if_any_bucket_matches(profile):
    result = check(
        _job(location_field="Remote", mode="remote", remote_scope="global", seniority=["intern", "senior"]),
        profile,
    )
    assert result.seniority_pass


# ---------------------------------------------------------------------------
# rejection_reason categorisation
# ---------------------------------------------------------------------------

def test_rejection_reason_both_fail(profile):
    result = check(_job(location_field="Nairobi, Kenya", mode="onsite", seniority=["intern"]), profile)
    assert result.rejection_reason == "hard_constraints"


def test_rejection_reason_location_only(profile):
    result = check(_job(location_field="Nairobi, Kenya", mode="onsite", seniority=["senior"]), profile)
    assert result.rejection_reason == "location"


def test_rejection_reason_seniority_only(profile):
    result = check(_job(location_field="Remote", mode="remote", remote_scope="global", seniority=["intern"]), profile)
    assert result.rejection_reason == "seniority"


def test_rejection_reason_none_when_passed(profile):
    result = check(_job(location_field="Remote", mode="remote", remote_scope="global", seniority=["senior"]), profile)
    assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# posting_type
# ---------------------------------------------------------------------------

def _passable_job(**overrides):
    defaults = dict(location_field="Remote", mode="remote", remote_scope="global", seniority=["senior"])
    defaults.update(overrides)
    return _job(**defaults)


def test_posting_type_null_treated_as_job(profile):
    result = check(_passable_job(posting_type=None), profile)
    assert result.posting_type_pass
    assert result.passed


def test_posting_type_job_passes(profile):
    result = check(_passable_job(posting_type="job"), profile)
    assert result.posting_type_pass
    assert result.passed


def test_posting_type_volunteer_excluded(profile):
    result = check(_passable_job(posting_type="volunteer"), profile)
    assert not result.posting_type_pass
    assert not result.passed
    assert result.rejection_reason == "posting_type"


def test_posting_type_funding_excluded(profile):
    result = check(_passable_job(posting_type="funding"), profile)
    assert not result.posting_type_pass
    assert not result.passed
    assert result.rejection_reason == "posting_type"


def test_posting_type_speculative_excluded(profile):
    """Added after the Animal Advocacy Careers recon: several of its listings
    are literal "general application"/"expression of interest" posts with no
    specific role — exactly the enrichment prompt's own speculative
    definition — and would otherwise reach Tier 1 scoring."""
    result = check(_passable_job(posting_type="speculative"), profile)
    assert not result.posting_type_pass
    assert not result.passed
    assert result.rejection_reason == "posting_type"


def test_posting_type_internship_fellowship_program_not_excluded(profile):
    """Per-user opt-in for these is future work — for now they pass by default,
    same as "job". program specifically is NOT treated like speculative: it's
    a structured course/accelerator/incubator, a different category, and no
    confirmed mis-filed posting has been found there yet."""
    for value in ["internship", "fellowship", "program"]:
        result = check(_passable_job(posting_type=value), profile)
        assert result.posting_type_pass, f"{value} should not be excluded by default"


def test_posting_type_and_location_both_fail_categorized_as_hard_constraints(profile):
    result = check(_job(location_field="Nairobi, Kenya", mode="onsite", seniority=["senior"], posting_type="volunteer"), profile)
    assert result.rejection_reason == "hard_constraints"
    assert not result.location_pass
    assert not result.posting_type_pass
    assert result.seniority_pass
