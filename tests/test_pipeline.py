"""Integration tests for pipeline.py: fetch-once/loop-over-users against a fake
Supabase client, stubbed sources, and monkeypatched LLM calls (no real network/API)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from src import enrich as enrich_mod
from src import pipeline
from src import quota_store
from src import scoring as scoring_mod
from src.dedup import canonical_url
from src.sources import eightyk, iap, probablygood
from tests.conftest import make_posting, make_tier1
from src.schemas import Tier2EnrichmentOutput


def _weights_row(user_id: str, *, threshold: float, near_miss_floor: float = 0.6) -> dict:
    # location_rule/seniority_rule are `text` columns in real Postgres — PostgREST
    # returns them as JSON strings, not parsed objects. Stored as strings here so the
    # gate.check() call in these integration tests exercises the real end-to-end path
    # (user_store._parse_rule -> gate.py), the same path that crashed on live data.
    return {
        "user_id": user_id,
        "cause_mission_fit": 1 / 7, "role_function_fit": 1 / 7,
        "location_compatibility": 1 / 7, "seniority_match": 1 / 7,
        "comp_adequacy": 1 / 7, "values_alignment": 1 / 7, "skill_growth": 1 / 7,
        "location_rule": json.dumps({"accept_fully_remote": True, "accept_sweden_hybrid": False, "accept_onsite_locations": []}),
        "seniority_rule": json.dumps({"min_years_experience": 5}),
        "insert_threshold": threshold,
        "near_miss_floor": near_miss_floor,
    }


def _profile_row(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "experience": ["Built a BI function"], "skills": ["sql"],
        "career_goals": "Data leadership", "cause_priorities": ["global health"],
        "location": "Remote", "location_constraints": "Remote preferred",
        "seniority_level": "Senior", "comp_needs": "Market rate",
        "values_notes": "GWWC pledge",
    }


def _seed_user(client, user_id: str, *, threshold: float, near_miss_floor: float = 0.6) -> None:
    client.seed("profiles", [_profile_row(user_id)])
    client.seed("scoring_weights", [_weights_row(user_id, threshold=threshold, near_miss_floor=near_miss_floor)])


def _stub_sources(monkeypatch, postings: list):
    monkeypatch.setattr(iap, "fetch", lambda: postings)
    monkeypatch.setattr(eightyk, "fetch", lambda: [])
    monkeypatch.setattr(probablygood, "fetch", lambda: [])


def _stub_llms(monkeypatch, *, fit: float = 0.8):
    dims = {dim: fit for dim in (
        "cause_mission_fit", "role_function_fit", "location_compatibility",
        "seniority_match", "comp_adequacy", "values_alignment", "skill_growth",
    )}
    monkeypatch.setattr(scoring_mod, "_call_llm", lambda posting, profile, rubric: make_tier1(**dims))
    monkeypatch.setattr(
        enrich_mod, "_call_llm",
        lambda scored, profile: Tier2EnrichmentOutput(
            org_summary="A great org.", why_fits="Fits well.", why_not_fits="Minor gaps.",
            emphasize_in_cv=["BI leadership"], deemphasize=["IT portfolio"],
        ),
    )


# ---------------------------------------------------------------------------
# No users
# ---------------------------------------------------------------------------

def test_run_with_no_users_returns_empty_run_log(fake_client, monkeypatch):
    _stub_sources(monkeypatch, [])
    log = pipeline.run(_client=fake_client)
    assert log.user_results == []
    assert log.counts.scraped == 0


# ---------------------------------------------------------------------------
# Two users, different thresholds, same posting
# ---------------------------------------------------------------------------

def test_different_thresholds_produce_different_verdicts(fake_client, monkeypatch):
    _seed_user(fake_client, "u-low", threshold=0.5, near_miss_floor=0.3)
    _seed_user(fake_client, "u-high", threshold=0.95, near_miss_floor=0.6)

    posting = make_posting(
        url="https://example.org/job/shared",
        source="iap", location="Remote", seniority=None, deadline=date(2030, 1, 1),
    )
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    results = {r.user_id: r for r in log.user_results}
    assert results["u-low"].inserted == 1
    assert results["u-high"].inserted == 0
    assert results["u-high"].near_misses == 1  # 0.8 >= near_miss_floor(0.6), < threshold(0.95)

    assert fetch_match_urls(fake_client, "u-low") == {canonical_url(posting.url)}
    assert fetch_match_urls(fake_client, "u-high") == set()


def fetch_match_urls(client, user_id: str) -> set[str]:
    return {r["canonical_url"] for r in client.table("matches").rows if r["user_id"] == user_id}


# ---------------------------------------------------------------------------
# Existing match for one user does not block another user
# ---------------------------------------------------------------------------

def test_existing_match_skipped_for_that_user_only(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    _seed_user(fake_client, "u2", threshold=0.5)

    posting = make_posting(
        url="https://example.org/job/existing",
        source="iap", location="Remote", deadline=date(2030, 1, 1),
    )
    cu = canonical_url(posting.url)
    fake_client.seed("matches", [{"user_id": "u1", "canonical_url": cu, "title": "x", "org": "x"}])

    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    results = {r.user_id: r for r in log.user_results}
    assert results["u1"].new_after_dedup == 0   # already in u1's matches, filtered before gate
    assert results["u2"].new_after_dedup == 1
    assert results["u2"].inserted == 1


# ---------------------------------------------------------------------------
# 14-day stale cutoff for deadline-less, posted_at-less postings
# ---------------------------------------------------------------------------

def test_stale_deadline_less_posting_dropped_from_shared_pool(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)

    posting = make_posting(
        url="https://example.org/job/stale", source="iap",
        location="Remote", deadline=None,  # no deadline, no posted_at
    )
    old_first_seen = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    fake_client.seed("seen", [{
        "user_id": "someone-else", "canonical_url": canonical_url(posting.url),
        "verdict": "below_threshold", "fit_score": 0.4, "first_seen": old_first_seen,
    }])

    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    assert log.counts.stale_dropped == 1
    assert log.counts.shared_pool_size == 0
    assert log.user_results[0].new_after_dedup == 0


def test_fresh_deadline_less_posting_kept_when_never_seen(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    posting = make_posting(url="https://example.org/job/fresh", source="iap", deadline=None)

    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    assert log.counts.stale_dropped == 0
    assert log.counts.shared_pool_size == 1


# ---------------------------------------------------------------------------
# --limit is a global cap on the shared pool
# ---------------------------------------------------------------------------

def test_limit_caps_shared_pool(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.99, near_miss_floor=0.99)  # nothing inserts; just count postings
    postings = [
        make_posting(url=f"https://example.org/job/{i}", source="iap", deadline=date(2030, 1, 1))
        for i in range(5)
    ]
    _stub_sources(monkeypatch, postings)
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client, limit=2)

    assert log.counts.shared_pool_size == 2
    assert log.user_results[0].new_after_dedup == 2


# ---------------------------------------------------------------------------
# dry-run does not write seen/matches
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    posting = make_posting(url="https://example.org/job/dry", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client, dry_run=True)

    assert log.user_results[0].inserted == 1  # still counted
    assert fake_client.table("matches").rows == []
    assert fake_client.table("seen").rows == []


# ---------------------------------------------------------------------------
# only_user_id — one-off single-user run + search_quota status write-back
# ---------------------------------------------------------------------------

def test_one_off_scores_only_the_requested_user(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    _seed_user(fake_client, "u2", threshold=0.5)
    posting = make_posting(url="https://example.org/job/one-off", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client, only_user_id="u1")

    assert [r.user_id for r in log.user_results] == ["u1"]
    assert fetch_match_urls(fake_client, "u1") == {canonical_url(posting.url)}
    assert fetch_match_urls(fake_client, "u2") == set()


def test_one_off_success_writes_running_then_complete(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    posting = make_posting(url="https://example.org/job/quota-ok", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    calls: list[tuple[str, str]] = []
    real_set_status = quota_store.set_status

    def spy(client, user_id, status, **kwargs):
        calls.append((user_id, status))
        return real_set_status(client, user_id, status, **kwargs)

    monkeypatch.setattr(pipeline, "set_status", spy)

    pipeline.run(_client=fake_client, only_user_id="u1")

    assert calls == [("u1", "running"), ("u1", "complete")]
    row = fake_client.table("search_quota").rows[0]
    assert row["status"] == "complete"
    assert row["completed_at"]


def test_one_off_zero_matches_still_marks_complete(fake_client, monkeypatch):
    """Zero matches written is still a successful run — status must be 'complete', not 'failed'."""
    _seed_user(fake_client, "u1", threshold=0.99, near_miss_floor=0.99)
    posting = make_posting(url="https://example.org/job/no-match", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.5)

    log = pipeline.run(_client=fake_client, only_user_id="u1")

    assert log.user_results[0].inserted == 0
    assert fake_client.table("search_quota").rows[0]["status"] == "complete"


def test_one_off_exception_writes_failed_and_reraises(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    _stub_sources(monkeypatch, [])

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "load_seen_urls", boom)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(_client=fake_client, only_user_id="u1")

    row = fake_client.table("search_quota").rows[0]
    assert row["status"] == "failed"
    assert "completed_at" not in row


def test_one_off_user_not_found_writes_failed(fake_client):
    with pytest.raises(RuntimeError, match="no profile"):
        pipeline.run(_client=fake_client, only_user_id="ghost-user")

    row = fake_client.table("search_quota").rows[0]
    assert row["status"] == "failed"


def test_daily_mode_never_touches_search_quota(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    posting = make_posting(url="https://example.org/job/daily", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    pipeline.run(_client=fake_client)  # only_user_id=None

    assert fake_client.table("search_quota").rows == []


def test_one_off_dry_run_never_touches_search_quota(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    posting = make_posting(url="https://example.org/job/one-off-dry", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    pipeline.run(_client=fake_client, only_user_id="u1", dry_run=True)

    assert fake_client.table("search_quota").rows == []
