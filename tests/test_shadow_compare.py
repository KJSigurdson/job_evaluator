"""Tests for src/shadow_compare.py: matching v1's live scrape to v2's persisted
jobs by canonical_url, and diffing gate.check() vs gate_v2.check() verdicts."""
from __future__ import annotations

import json

from src import shadow_compare
from src.job_normalize import normalize_job_url
from tests.conftest import make_posting


def _profile_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", experience=["Built a BI function"], skills=["sql"],
        career_goals="Data leadership", cause_priorities=["global health"],
        location="Sundsvall, Sweden", seniority_level="Senior",
        comp_needs="Market rate", values_notes="GWWC pledge",
    )
    defaults.update(kwargs)
    return defaults


def _weights_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", cause_mission_fit=0.25, role_function_fit=0.25, location_compatibility=0.15,
        seniority_match=0.10, comp_adequacy=0.10, values_alignment=0.10, skill_growth=0.05,
        location_rule=json.dumps({
            "accept_fully_remote": True, "accept_hybrid_in": ["Sweden"], "accept_onsite_in": ["Sweden"],
        }),
        seniority_rule=json.dumps({"accept_levels": ["mid", "senior", "director"]}),
        insert_threshold=0.75, near_miss_floor=0.65,
    )
    defaults.update(kwargs)
    return defaults


def _v2_job(url="https://jobs.80000hours.org/role", **overrides) -> dict:
    defaults = dict(
        id="job-1", title="Head of Data", org="GiveDirectly",
        canonical_url=normalize_job_url(url),
        location_field="Remote, Global",
        enrichment={
            "seniority": ["senior"],
            "work_arrangement": {"mode": "remote", "remote_scope": "global", "eligible_regions": [],
                                  "restriction_type": None, "timezone_constraint": None},
        },
    )
    defaults.update(overrides)
    return defaults


def test_no_users_returns_zero(fake_client, monkeypatch):
    monkeypatch.setattr(shadow_compare.eightyk, "fetch", lambda: [])
    result = shadow_compare.run(client=fake_client)
    assert result == {"users": 0, "postings_matched": 0, "pairs_compared": 0, "divergences": [],
                       "v1_passed_v2_failed": 0, "v1_failed_v2_passed": 0}


def test_unmatched_v2_job_excluded_from_comparison(fake_client, monkeypatch):
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])
    fake_client.seed("jobs", [_v2_job(url="https://jobs.80000hours.org/no-v1-match", status="enriched", source_board="80000_hours")])
    monkeypatch.setattr(shadow_compare.eightyk, "fetch", lambda: [])  # v1 scrape finds nothing

    result = shadow_compare.run(client=fake_client)

    assert result["postings_matched"] == 0
    assert result["pairs_compared"] == 0


def test_agreement_no_divergence(fake_client, monkeypatch):
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])
    url = "https://jobs.80000hours.org/remote-role"
    fake_client.seed("jobs", [_v2_job(url=url, status="enriched", source_board="80000_hours")])
    monkeypatch.setattr(
        shadow_compare.eightyk, "fetch",
        lambda: [make_posting(url=url, location="Remote, Global", seniority="Senior")],
    )

    result = shadow_compare.run(client=fake_client)

    assert result["pairs_compared"] == 1
    assert result["divergences"] == []


def test_divergence_recorded_when_v1_and_v2_disagree(fake_client, monkeypatch):
    """v1's keyword substring match can't see a region restriction on "Remote" text
    alone and passes it; v2's structured remote_scope/eligible_regions correctly
    fails it for a user outside that region — the exact divergence class this tool
    exists to surface."""
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])
    url = "https://jobs.80000hours.org/us-only-remote-role"
    fake_client.seed("jobs", [_v2_job(
        url=url, status="enriched", source_board="80000_hours",
        location_field="Remote (US only)",
        enrichment={
            "seniority": ["senior"],
            "work_arrangement": {"mode": "remote", "remote_scope": "region_restricted",
                                  "eligible_regions": ["US"], "restriction_type": "authorization",
                                  "timezone_constraint": None},
        },
    )])
    monkeypatch.setattr(
        shadow_compare.eightyk, "fetch",
        lambda: [make_posting(url=url, location="Remote (US only)", seniority="Senior")],
    )

    result = shadow_compare.run(client=fake_client)

    assert result["pairs_compared"] == 1
    assert len(result["divergences"]) == 1
    d = result["divergences"][0]
    assert d.v1_passed is True   # v1 sees "remote" keyword, passes
    assert d.v2_passed is False  # v2 sees US-only restriction, Swedish user fails
    assert result["v1_passed_v2_failed"] == 1
    assert result["v1_failed_v2_passed"] == 0


def test_multiple_users_multiply_pairs_compared(fake_client, monkeypatch):
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1"), _weights_row(user_id="u2")])
    url = "https://jobs.80000hours.org/remote-role"
    fake_client.seed("jobs", [_v2_job(url=url, status="enriched", source_board="80000_hours")])
    monkeypatch.setattr(
        shadow_compare.eightyk, "fetch",
        lambda: [make_posting(url=url, location="Remote, Global", seniority="Senior")],
    )

    result = shadow_compare.run(client=fake_client)

    assert result["users"] == 2
    assert result["postings_matched"] == 1
    assert result["pairs_compared"] == 2
