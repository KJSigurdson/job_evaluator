"""Tests for src/gate_v2.py — the v2 hard-gate, evaluated against a jobs row's
structured enrichment instead of v1's raw posting text. Uses the same `profile`
fixture as test_gate.py (accept_fully_remote=True, accept_hybrid_in=[Sweden,
Sundsvall], accept_onsite_in=[Sundsvall], accept_levels=[mid, senior, director])
so the two gates' behavior is directly comparable."""
from __future__ import annotations

from src.gate_v2 import check


def _job(location_field=None, seniority=None, mode="unspecified", remote_scope="unspecified",
         eligible_regions=None, restriction_type=None) -> dict:
    return {
        "location_field": location_field,
        "enrichment": {
            "seniority": seniority or [],
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
