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


def _profile_row(user_id: str, *, email_on_match: bool = False) -> dict:
    return {
        "user_id": user_id,
        "experience": ["Built a BI function"], "skills": ["sql"],
        "career_goals": "Data leadership", "cause_priorities": ["global health"],
        "location": "Remote", "location_constraints": "Remote preferred",
        "seniority_level": "Senior", "comp_needs": "Market rate",
        "values_notes": "GWWC pledge", "email_on_match": email_on_match,
    }


def _seed_user(
    client, user_id: str, *, threshold: float, near_miss_floor: float = 0.6, email_on_match: bool = False,
) -> None:
    client.seed("profiles", [_profile_row(user_id, email_on_match=email_on_match)])
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
# Seen-table writes are batched per user, not per posting
# ---------------------------------------------------------------------------

def test_seen_writes_batched_not_per_posting(fake_client, monkeypatch):
    """Regression for the workflow timeout: 50 gated-out postings for one user must
    produce ONE seen upsert() call, not 50 sequential round-trips."""
    _seed_user(fake_client, "u1", threshold=0.5)
    postings = [
        # London on-site fails the hard gate — cheap way to generate many gated_out
        # verdicts without needing 50 distinct LLM stub responses.
        make_posting(url=f"https://example.org/job/{i}", source="iap", location="London, UK (on-site)")
        for i in range(50)
    ]
    _stub_sources(monkeypatch, postings)
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    assert log.user_results[0].gated_out == 50
    assert len(fake_client.table("seen").upsert_calls) == 1
    assert len(fake_client.table("seen").upsert_calls[0]) == 50
    assert len(fake_client.table("seen").rows) == 50


def test_seen_writes_batched_across_gate_and_scoring_verdicts(fake_client, monkeypatch):
    """gated_out (from the gate loop) and below_threshold/inserted (from Tier1/Tier2)
    verdicts for the same user must land in the SAME batch flush, not separate calls."""
    _seed_user(fake_client, "u1", threshold=0.9, near_miss_floor=0.5)  # 0.8 fit -> near-miss, not inserted
    gated = make_posting(url="https://example.org/job/gated", source="iap", location="London, UK (on-site)")
    scored = make_posting(url="https://example.org/job/scored", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [gated, scored])
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client)

    assert log.user_results[0].gated_out == 1
    assert log.user_results[0].near_misses == 1
    assert len(fake_client.table("seen").upsert_calls) == 1
    assert len(fake_client.table("seen").upsert_calls[0]) == 2
    verdicts = {r["canonical_url"]: r["verdict"] for r in fake_client.table("seen").rows}
    assert verdicts[canonical_url(gated.url)] == "gated_out"
    assert verdicts[canonical_url(scored.url)] == "below_threshold"


def test_seen_writes_batched_separately_per_user(fake_client, monkeypatch):
    """Each user's batch is independent — one upsert call per user, not merged."""
    _seed_user(fake_client, "u1", threshold=0.5)
    _seed_user(fake_client, "u2", threshold=0.5)
    postings = [
        make_posting(url=f"https://example.org/job/{i}", source="iap", location="London, UK (on-site)")
        for i in range(10)
    ]
    _stub_sources(monkeypatch, postings)
    _stub_llms(monkeypatch, fit=0.8)

    pipeline.run(_client=fake_client)

    assert len(fake_client.table("seen").upsert_calls) == 2
    assert all(len(call) == 10 for call in fake_client.table("seen").upsert_calls)


def test_seen_writes_dry_run_makes_no_upsert_calls(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5)
    postings = [
        make_posting(url=f"https://example.org/job/{i}", source="iap", location="London, UK (on-site)")
        for i in range(5)
    ]
    _stub_sources(monkeypatch, postings)
    _stub_llms(monkeypatch, fit=0.8)

    log = pipeline.run(_client=fake_client, dry_run=True)

    assert log.user_results[0].gated_out == 5
    assert fake_client.table("seen").upsert_calls == []
    assert fake_client.table("seen").rows == []


class _ReverseRandom:
    """Deterministic stand-in for random.Random — reverses instead of shuffling, so
    tests can assert the pool was reordered before --limit is applied without relying
    on a real (unpredictable) shuffle outcome."""
    def __init__(self, seed):
        pass

    def shuffle(self, lst):
        lst.reverse()


class _SpyRandom:
    """Records whether .shuffle() was ever called, without changing list order."""
    calls: list = []

    def __init__(self, seed):
        pass

    def shuffle(self, lst):
        _SpyRandom.calls.append(True)


def test_limit_draws_from_multiple_sources_not_just_first_scraped(fake_client, monkeypatch):
    """Regression test: iap is scraped first and contributes far more postings than
    --limit. Before the fix, fresh[:limit] was a positional slice of scrape order, so
    a capped run only ever saw iap postings — 80k and probably_good were starved
    entirely whenever iap alone exceeded the cap."""
    _seed_user(fake_client, "u1", threshold=0.5)

    iap_postings = [
        make_posting(url=f"https://example.org/iap/{i}", source="iap", deadline=date(2030, 1, 1))
        for i in range(5)
    ]
    eightyk_postings = [make_posting(url="https://example.org/80k/0", source="80k", deadline=date(2030, 1, 1))]
    pg_postings = [make_posting(url="https://example.org/pg/0", source="probably_good", deadline=date(2030, 1, 1))]

    monkeypatch.setattr(iap, "fetch", lambda: iap_postings)
    monkeypatch.setattr(eightyk, "fetch", lambda: eightyk_postings)
    monkeypatch.setattr(probablygood, "fetch", lambda: pg_postings)
    _stub_llms(monkeypatch, fit=0.8)
    monkeypatch.setattr(pipeline.random, "Random", _ReverseRandom)

    log = pipeline.run(_client=fake_client, limit=2)

    assert log.counts.shared_pool_size == 2
    inserted_sources = {r["source"] for r in fake_client.table("matches").rows}
    assert len(inserted_sources) > 1
    assert inserted_sources != {"iap"}


def test_no_shuffle_when_limit_is_none(fake_client, monkeypatch):
    """An uncapped run must stay in natural scrape order — the shuffle only exists to
    make --limit fair across sources, and must not run otherwise."""
    _seed_user(fake_client, "u1", threshold=0.5)

    iap_posting = make_posting(url="https://example.org/iap/0", source="iap", deadline=date(2030, 1, 1))
    eightyk_posting = make_posting(url="https://example.org/80k/0", source="80k", deadline=date(2030, 1, 1))
    pg_posting = make_posting(url="https://example.org/pg/0", source="probably_good", deadline=date(2030, 1, 1))

    monkeypatch.setattr(iap, "fetch", lambda: [iap_posting])
    monkeypatch.setattr(eightyk, "fetch", lambda: [eightyk_posting])
    monkeypatch.setattr(probablygood, "fetch", lambda: [pg_posting])
    _stub_llms(monkeypatch, fit=0.8)

    _SpyRandom.calls = []
    monkeypatch.setattr(pipeline.random, "Random", _SpyRandom)

    pipeline.run(_client=fake_client)  # limit=None

    assert _SpyRandom.calls == []
    inserted_sources_in_order = [r["source"] for r in fake_client.table("matches").rows]
    assert inserted_sources_in_order == ["iap", "80k", "probably_good"]


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


# ---------------------------------------------------------------------------
# Opt-in match-digest email
# ---------------------------------------------------------------------------

def _spy_digest(monkeypatch):
    calls: list[tuple[str, list]] = []
    monkeypatch.setattr(pipeline, "send_match_digest", lambda user_id, matches: calls.append((user_id, matches)))
    return calls


def test_email_on_match_true_with_insert_triggers_one_send(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=True)
    posting = make_posting(url="https://example.org/job/digest-yes", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)
    calls = _spy_digest(monkeypatch)

    pipeline.run(_client=fake_client)

    assert len(calls) == 1
    user_id, matches = calls[0]
    assert user_id == "u1"
    assert len(matches) == 1
    assert matches[0]["url"] == posting.url
    assert matches[0]["title"] == posting.title
    assert matches[0]["fit_score"] == pytest.approx(0.8, rel=0.05)


def test_email_on_match_false_triggers_no_send(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=False)
    posting = make_posting(url="https://example.org/job/digest-no", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)
    calls = _spy_digest(monkeypatch)

    pipeline.run(_client=fake_client)

    assert calls == []


def test_email_on_match_true_with_zero_inserts_triggers_no_send(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.99, near_miss_floor=0.99, email_on_match=True)
    posting = make_posting(url="https://example.org/job/digest-no-insert", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.5)
    calls = _spy_digest(monkeypatch)

    pipeline.run(_client=fake_client)

    assert calls == []


def test_email_on_match_only_includes_new_matches_this_run_not_preexisting(fake_client, monkeypatch):
    """Regression: an already-existing match for this user (filtered out by the
    skip-set before scoring) must not appear in the digest — only matches actually
    inserted THIS run."""
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=True)
    existing = make_posting(url="https://example.org/job/pre-existing", source="iap", deadline=date(2030, 1, 1))
    new = make_posting(url="https://example.org/job/brand-new", source="iap", deadline=date(2030, 1, 1))
    fake_client.seed("matches", [{
        "user_id": "u1", "canonical_url": canonical_url(existing.url), "title": "x", "org": "x",
    }])
    _stub_sources(monkeypatch, [existing, new])
    _stub_llms(monkeypatch, fit=0.8)
    calls = _spy_digest(monkeypatch)

    pipeline.run(_client=fake_client)

    assert len(calls) == 1
    _, matches = calls[0]
    assert [m["url"] for m in matches] == [new.url]


def test_email_digest_dry_run_sends_nothing(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=True)
    posting = make_posting(url="https://example.org/job/digest-dry", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)
    calls = _spy_digest(monkeypatch)

    pipeline.run(_client=fake_client, dry_run=True)

    assert calls == []


def test_email_digest_failure_does_not_fail_run_or_flip_quota(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=True)
    posting = make_posting(url="https://example.org/job/digest-fails", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    def boom(user_id, matches):
        raise RuntimeError("email service is down")

    monkeypatch.setattr(pipeline, "send_match_digest", boom)

    log = pipeline.run(_client=fake_client, only_user_id="u1")  # exercises quota tracking too

    assert log.user_results[0].inserted == 1
    assert fake_client.table("search_quota").rows[0]["status"] == "complete"


def test_email_digest_failure_does_not_interrupt_other_users(fake_client, monkeypatch):
    _seed_user(fake_client, "u1", threshold=0.5, email_on_match=True)
    _seed_user(fake_client, "u2", threshold=0.5, email_on_match=True)
    posting = make_posting(url="https://example.org/job/digest-multi", source="iap", deadline=date(2030, 1, 1))
    _stub_sources(monkeypatch, [posting])
    _stub_llms(monkeypatch, fit=0.8)

    calls: list[str] = []

    def flaky(user_id, matches):
        calls.append(user_id)
        if user_id == "u1":
            raise RuntimeError("email service is down for u1")

    monkeypatch.setattr(pipeline, "send_match_digest", flaky)

    log = pipeline.run(_client=fake_client)

    results = {r.user_id: r for r in log.user_results}
    assert results["u1"].inserted == 1
    assert results["u2"].inserted == 1
    assert calls == ["u1", "u2"]
