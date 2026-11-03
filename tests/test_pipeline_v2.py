"""Tests for src/pipeline_v2.py: orchestration/wiring for both the scheduled
full pipeline and the manual DB-only path, skip logic for incomplete
profiles, already-matched/valid-verdict exclusion, digest payload mapping,
and search_quota bookkeeping.

Phase 2: the old single _match_and_notify_user() per-user function is split
into _prepare_tier1_request() (Phase A: gate -> exclude matched/valid-verdict
-> vector top-N, no LLM calls) and _finish_user_matching() (Phase C: Tier 2
candidates from tier1_verdicts, Tier 2, digest) — Phase B (ONE Tier 1 Batches
submission across every user) sits between them in run().

Submodules (ingest/enrich_jobs/match_v2/tier1_rerank/tier2_enrich/
profile_embed) each have their own dedicated test suite already — these
tests monkeypatch them at the pipeline_v2 call site and verify pipeline_v2
calls them correctly (right function, right order, right args), not their
internal correctness.
"""
from __future__ import annotations

import json

import pytest

from src import pipeline_v2
from src.pipeline_v2 import (
    _build_digest_payload,
    _finish_user_matching,
    _prepare_tier1_request,
    _send_queued_digest,
    run,
)


def _profile_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", experience="Work: Data leader at OrgA", skills="SQL (expert)",
        career_goals="Move into data leadership.", cause_priorities="Global health",
        location="Sundsvall, Sweden", seniority_level="Senior",
        comp_needs="Market rate", values_notes="GWWC pledge", email_on_match=True,
    )
    defaults.update(kwargs)
    return defaults


def _weights_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", cause_mission_fit=0.25, role_function_fit=0.25, location_compatibility=0.15,
        seniority_match=0.10, comp_adequacy=0.10, values_alignment=0.10, skill_growth=0.05,
        location_rule=json.dumps({"accept_fully_remote": True, "accept_hybrid_in": [], "accept_onsite_in": []}),
        seniority_rule=json.dumps({"accept_levels": ["mid", "senior", "director"]}),
        insert_threshold=0.75, near_miss_floor=0.65,
    )
    defaults.update(kwargs)
    return defaults


def _job(job_id="job-1", **overrides) -> dict:
    defaults = dict(
        id=job_id, title="Head of Data", org="GiveDirectly",
        canonical_url=f"https://example.org/{job_id}", location_field="Remote, Global",
        raw_description="text", source_board="80000_hours", enriched_at="2026-10-01T00:00:00+00:00",
        enrichment={
            "summary": "Lead data.", "seniority": ["senior"], "cause_area": ["global health"],
            "skill_tags": ["SQL"],
            # gate_v2 decides "fully remote" from this structured field, NOT a
            # location_field text match (see src/gate_v2.py's module
            # docstring) — without it, a job with no stated work_arrangement
            # never passes accept_fully_remote=True, regardless of
            # location_field's text.
            "work_arrangement": {
                "mode": "remote", "remote_scope": "global", "eligible_regions": [],
                "timezone_constraint": None, "restriction_type": None,
            },
        },
    )
    defaults.update(overrides)
    return defaults


def _seed_active_user(fake_client, *, profile_embedding=None, **profile_overrides):
    row = _profile_row(**profile_overrides)
    row["profile_embedding"] = profile_embedding
    fake_client.seed("profiles", [row])
    fake_client.seed("scoring_weights", [_weights_row()])


def _user_context(profile: dict, rubric: dict | None = None):
    from src.schemas import UserContext
    # threshold: 0.65 mirrors _weights_row()'s old-style default — 0.65*100==65,
    # so score=40/score=80 fixtures elsewhere mean what their comments say
    # without every one needing its own rubric override.
    return UserContext(user_id="u1", profile=profile, rubric=rubric or {"threshold": 0.65})


def _prep(user_id="u1", *, skipped=None, gate_survivor_count=1, profile_content_hash=None, job_cards=None) -> dict:
    return {
        "user_id": user_id, "skipped": skipped, "gate_survivor_count": gate_survivor_count,
        "profile_content_hash": profile_content_hash,
        "job_cards": job_cards if job_cards is not None else [
            {"id": "job-1", "title": "Head of Data", "org": "GiveDirectly", "enrichment": {}},
        ],
    }


def _candidate(job_id="job-1", tier1_score=80, tier1_reason="Great fit", job=None) -> dict:
    return {
        "user_id": "u1", "profile": {}, "job": job or _job(job_id),
        "tier1_score": tier1_score, "tier1_reason": tier1_reason,
    }


# ---------------------------------------------------------------------------
# _build_digest_payload — v2 matches_v2 rows -> v1's exact digest shape
# ---------------------------------------------------------------------------

def test_build_digest_payload_maps_v2_fields_to_v1_shape():
    matches_v2_rows = [{"job_id": "job-1", "tier1_score": 82, "tier2": {"why_fits": "- Strong fit"}}]
    jobs_by_id = {"job-1": _job()}
    payload = _build_digest_payload(matches_v2_rows, jobs_by_id)
    assert payload == [{
        "title": "Head of Data", "org": "GiveDirectly", "location": "Remote, Global",
        "fit_score": 0.82, "why_fits": "- Strong fit", "url": "https://example.org/job-1",
    }]


def test_build_digest_payload_scales_tier1_score_to_v1s_0_to_1_range():
    """matches_v2.tier1_score is stored 0-100; v1's contract (and the Edge
    Function's existing template) expects fit_score on 0.0-1.0 — confirmed
    real bug fixed 2026-11-03: a tier1_score of 78 must render as 0.78, not
    the raw unscaled 78."""
    matches_v2_rows = [{"job_id": "job-1", "tier1_score": 78, "tier2": {}}]
    payload = _build_digest_payload(matches_v2_rows, {"job-1": _job()})
    assert payload[0]["fit_score"] == pytest.approx(0.78)


def test_build_digest_payload_skips_row_with_no_matching_job():
    payload = _build_digest_payload([{"job_id": "missing", "tier1_score": 80, "tier2": {}}], {})
    assert payload == []


# ---------------------------------------------------------------------------
# _send_queued_digest — delivery tracking via matches_v2.notified_at,
# decoupled from "new this run" (added 2026-11-03 after the 9 real Adam
# Holt matches from run 56a07def went unnotified with no way to retry)
# ---------------------------------------------------------------------------

def _match_row(match_id, job_id, *, created_at, notified_at=None, tier1_score=80):
    return {
        "id": match_id, "user_id": "u1", "job_id": job_id, "tier1_score": tier1_score,
        "tier2": {"why_fits": "- Great fit"}, "created_at": created_at, "notified_at": notified_at,
    }


def test_send_queued_digest_false_when_email_on_match_off(fake_client):
    user = _user_context({"email_on_match": False})
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [_match_row("m1", "job-1", created_at="2026-11-01T00:00:00+00:00")])

    assert _send_queued_digest(fake_client, user) is False


def test_send_queued_digest_false_when_nothing_queued(fake_client):
    user = _user_context({"email_on_match": True})
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [
        _match_row("m1", "job-1", created_at="2026-11-01T00:00:00+00:00", notified_at="2026-11-01T00:05:00+00:00"),
    ])

    assert _send_queued_digest(fake_client, user) is False


def test_send_queued_digest_stamps_only_sent_ids_on_success(fake_client, monkeypatch):
    """A successful send must stamp notified_at on exactly the rows that
    were actually included in the sent payload — a different user's (or a
    DIFFERENT job's) un-sent row must stay untouched."""
    user = _user_context({"email_on_match": True})
    fake_client.seed("jobs", [_job("job-1"), _job("job-2")])
    fake_client.seed("matches_v2", [
        _match_row("m1", "job-1", created_at="2026-11-01T00:00:00+00:00"),
        _match_row("m2", "job-2", created_at="2026-11-01T00:01:00+00:00"),
        _match_row("other-user", "job-1", created_at="2026-11-01T00:00:00+00:00"),  # different user_id below
    ])
    fake_client.table("matches_v2").rows[2]["user_id"] = "someone-else"
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: True)

    result = _send_queued_digest(fake_client, user)

    assert result is True
    rows_by_id = {r["id"]: r for r in fake_client.table("matches_v2").rows}
    assert rows_by_id["m1"]["notified_at"] is not None
    assert rows_by_id["m2"]["notified_at"] is not None
    assert rows_by_id["other-user"]["notified_at"] is None  # not this user's row — untouched


def test_send_queued_digest_failed_send_leaves_notified_at_null_for_retry(fake_client, monkeypatch):
    """A failed send (non-2xx, surfaced as a False return) must leave
    notified_at NULL so the identical rows are picked up again on a later
    call — this is the exact mechanism that replaces the old "new this
    run only" logic that lost the 9 real Adam Holt matches."""
    user = _user_context({"email_on_match": True})
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [_match_row("m1", "job-1", created_at="2026-11-01T00:00:00+00:00")])
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: False)

    first = _send_queued_digest(fake_client, user)
    assert first is False
    [row] = fake_client.table("matches_v2").rows
    assert row["notified_at"] is None

    # Retried on a later call (e.g. next run) with the endpoint now healthy:
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: True)
    second = _send_queued_digest(fake_client, user)
    assert second is True
    assert fake_client.table("matches_v2").rows[0]["notified_at"] is not None


def test_send_queued_digest_caps_at_batch_size_oldest_first(fake_client, monkeypatch):
    """More than _DIGEST_BATCH_CAP eligible rows exist for this user — only
    the oldest _DIGEST_BATCH_CAP get sent/stamped; the rest stay queued for
    a later run."""
    jobs = [_job(f"job-{i}") for i in range(25)]
    fake_client.seed("jobs", jobs)
    # Oldest first: job-0 is oldest (created_at increases with i), so the
    # 20 oldest are job-0..job-19 — job-20..job-24 must stay un-stamped.
    rows = [
        _match_row(f"m{i}", f"job-{i}", created_at=f"2026-11-01T00:{i:02d}:00+00:00")
        for i in range(25)
    ]
    fake_client.seed("matches_v2", rows)
    sent_payloads = []
    monkeypatch.setattr(
        pipeline_v2, "send_match_digest",
        lambda user_id, matches: (sent_payloads.append(matches), True)[1],
    )
    user = _user_context({"email_on_match": True})

    result = _send_queued_digest(fake_client, user)

    assert result is True
    [payload] = sent_payloads
    assert len(payload) == pipeline_v2._DIGEST_BATCH_CAP
    sent_urls = {m["url"] for m in payload}
    assert sent_urls == {f"https://example.org/job-{i}" for i in range(20)}

    rows_by_job = {r["job_id"]: r for r in fake_client.table("matches_v2").rows}
    for i in range(20):
        assert rows_by_job[f"job-{i}"]["notified_at"] is not None
    for i in range(20, 25):
        assert rows_by_job[f"job-{i}"]["notified_at"] is None


# ---------------------------------------------------------------------------
# _prepare_tier1_request — Phase A: skip logic, exclusion, gate rejections
# ---------------------------------------------------------------------------

def test_prepare_tier1_request_skips_empty_profile(fake_client):
    user = _user_context({"career_goals": None, "cause_priorities": None, "experience_inventory": None,
                           "skills": None, "values_notes": None, "hard_constraints": {}})

    result = _prepare_tier1_request(fake_client, user, [])

    assert result["skipped"] == "empty_profile"
    assert result["job_cards"] == []


def test_prepare_tier1_request_skips_no_profile_embedding(fake_client):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": None}])

    result = _prepare_tier1_request(fake_client, user, [])

    assert result["skipped"] == "no_profile_embedding"


def test_prepare_tier1_request_excludes_already_matched_jobs(fake_client):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-already-matched"}])
    fake_client.rpc_responses["match_v2_top_jobs"] = lambda params: [
        {"id": jid, "similarity": 0.9} for jid in params["candidate_job_ids"]
    ]

    jobs_pool = [_job("job-already-matched"), _job("job-new")]
    result = _prepare_tier1_request(fake_client, user, jobs_pool)

    job_ids = [c["id"] for c in result["job_cards"]]
    assert "job-already-matched" not in job_ids
    assert "job-new" in job_ids


def test_prepare_tier1_request_excludes_valid_verdict_jobs(fake_client):
    """A job this user already has a VALID tier1_verdicts row for must not
    be sent to Tier 1 again — the whole point of Phase 2's verdict cache."""
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]", "profile_content_hash": "hash-abc"}])
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-already-scored",
        "job_enriched_at": "2026-10-01T00:00:00+00:00", "profile_content_hash": "hash-abc",
    }])
    fake_client.rpc_responses["match_v2_top_jobs"] = lambda params: [
        {"id": jid, "similarity": 0.9} for jid in params["candidate_job_ids"]
    ]

    jobs_pool = [_job("job-already-scored"), _job("job-new")]
    # fetch_valid_verdict_job_ids() checks the verdict's job_enriched_at
    # against the job's CURRENT jobs.enriched_at — needs the `jobs` table
    # itself seeded (not just jobs_pool, which is a separate in-memory arg).
    fake_client.seed("jobs", jobs_pool)
    result = _prepare_tier1_request(fake_client, user, jobs_pool)

    job_ids = [c["id"] for c in result["job_cards"]]
    assert "job-already-scored" not in job_ids
    assert "job-new" in job_ids


def test_prepare_tier1_request_resends_job_whose_verdict_is_stale(fake_client):
    """A job re-enriched since it was scored has an INVALID verdict (stale
    job_enriched_at) — it must be resent to Tier 1, not skipped forever."""
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]", "profile_content_hash": "hash-abc"}])
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-1",
        "job_enriched_at": "2026-09-01T00:00:00+00:00",  # stale — job's real enriched_at is 2026-10-01
        "profile_content_hash": "hash-abc",
    }])
    fake_client.rpc_responses["match_v2_top_jobs"] = lambda params: [
        {"id": jid, "similarity": 0.9} for jid in params["candidate_job_ids"]
    ]

    result = _prepare_tier1_request(fake_client, user, [_job("job-1")])

    assert [c["id"] for c in result["job_cards"]] == ["job-1"]


def test_prepare_tier1_request_no_gate_survivors_returns_empty_job_cards(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "gate_survivors_with_rejections", lambda jobs, profile: ([], {}))

    result = _prepare_tier1_request(fake_client, user, [])

    assert result == {
        "user_id": "u1", "skipped": None, "gate_survivor_count": 0,
        "profile_content_hash": None, "job_cards": [],
    }


def test_prepare_tier1_request_records_gate_rejections_into_out_param(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(
        pipeline_v2, "gate_survivors_with_rejections",
        lambda jobs, profile: ([], {"location": 3, "seniority": 1}),
    )

    gate_rejections: dict = {}
    _prepare_tier1_request(fake_client, user, [], gate_rejections_out=gate_rejections)

    assert gate_rejections == {"u1": {"location": 3, "seniority": 1}}


def test_prepare_tier1_request_no_rejections_not_recorded(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "gate_survivors_with_rejections", lambda jobs, profile: ([], {}))

    gate_rejections: dict = {}
    _prepare_tier1_request(fake_client, user, [], gate_rejections_out=gate_rejections)

    assert gate_rejections == {}


# ---------------------------------------------------------------------------
# _finish_user_matching — Phase C: Tier 2 candidates (fetch_tier2_candidates),
# Tier 2, digest
# ---------------------------------------------------------------------------

def test_finish_user_matching_skipped_prep_returns_skip_result(fake_client):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    prep = _prep(skipped="empty_profile")

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert result == {"user_id": "u1", "skipped": "empty_profile"}


def test_finish_user_matching_dry_run_skips_tier2_entirely(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    prep = _prep(gate_survivor_count=3)

    def _should_not_run(*a, **k):
        raise AssertionError("dry_run must not call fetch_tier2_candidates/Tier 2")

    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", _should_not_run)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=True)

    assert result == {"user_id": "u1", "gate_survivors": 3, "tier1_scored": 0, "tier2_enriched": 0, "notified": False}


def test_finish_user_matching_no_candidates_returns_zero_counts(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    prep = _prep(gate_survivor_count=5, job_cards=[])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [])

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert result == {"user_id": "u1", "gate_survivors": 5, "tier1_scored": 0, "tier2_enriched": 0, "notified": False}


def test_finish_user_matching_scales_threshold_from_0_1_to_0_100(fake_client, monkeypatch):
    user = _user_context(
        {"career_goals": "x", "hard_constraints": {}, "email_on_match": False},
        rubric={"threshold": 0.8},
    )
    prep = _prep()
    captured = {}
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda client, *, user_id, profile, profile_content_hash, threshold: (
        captured.update(threshold=threshold), [_candidate()])[1])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 0, "skipped_below_threshold": 1, "failed": 0,
    })

    _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert captured["threshold"] == 80  # 0.8 * 100, not the raw 0.8
    assert isinstance(captured["threshold"], int)


@pytest.mark.parametrize("stored_threshold,expected_scaled", [(0.65, 65), (0.7, 70), (0.75, 75), (0.725, 73)])
def test_finish_user_matching_scales_real_production_threshold_values(fake_client, monkeypatch, stored_threshold, expected_scaled):
    """Pins the *100 scaling against the ACTUAL values confirmed live against
    production's scoring_weights.insert_threshold (0.65/0.7/0.75 — fractional
    0-1 storage, not pre-scaled 0-100): a log line showing "threshold 65.0"
    is this scaling's correct OUTPUT, not evidence the column is stored as
    65. If insert_threshold were ever actually stored as 65 instead of 0.65,
    this *100 would produce 6500 — unreachable against a 0-100 tier1_score
    — so this test is the regression guard for that scenario, not proof it's
    currently happening (it isn't, per the live query).

    0.725 is the fractional case: tier1_verdicts.score is an INTEGER column,
    and 0.725 * 100 = 72.5 sent as a .gte() filter value gets stringified
    into the request URL as "72.5" — Postgres rejects that outright trying
    to parse it as an integer (22P02 invalid input syntax for type integer).
    math.ceil() rounds UP to 73, the direction that preserves ">=" semantics
    against an integer column."""
    user = _user_context(
        {"career_goals": "x", "hard_constraints": {}, "email_on_match": False},
        rubric={"threshold": stored_threshold},
    )
    prep = _prep()
    captured = {}
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda client, *, user_id, profile, profile_content_hash, threshold: (
        captured.update(threshold=threshold), [])[1])

    _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert captured["threshold"] == expected_scaled
    assert isinstance(captured["threshold"], int)


def test_finish_user_matching_calls_fetch_candidates_then_tier2(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": False})
    prep = _prep()
    calls = []
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: (
        calls.append("fetch_candidates"), [_candidate()])[1])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: (
        calls.append("tier2"), {"considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0})[1])

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert calls == ["fetch_candidates", "tier2"]
    assert result["tier1_scored"] == len(prep["job_cards"])
    assert result["tier2_enriched"] == 1


def test_finish_user_matching_sends_digest_when_email_on_match_and_new_matches(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": True})
    prep = _prep()
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [{"id": "m1", "user_id": "u1", "job_id": "job-1", "tier1_score": 80,
                                      "tier2": {"why_fits": "- Great fit"}, "created_at": "2026-11-01T00:00:00+00:00",
                                      "notified_at": None}])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })
    digest_calls = []

    def _fake_send(user_id, matches):
        digest_calls.append((user_id, matches))
        return True  # confirmed 2xx

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _fake_send)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert result["notified"] is True
    [(user_id, matches)] = digest_calls
    assert user_id == "u1"
    assert matches == [{"title": "Head of Data", "org": "GiveDirectly", "location": "Remote, Global",
                         "fit_score": 0.8, "why_fits": "- Great fit", "url": "https://example.org/job-1"}]
    [row] = fake_client.table("matches_v2").rows
    assert row["notified_at"] is not None


def test_finish_user_matching_notified_false_when_digest_returns_false(fake_client, monkeypatch):
    """send_match_digest() returning False (e.g. a non-2xx response) must
    surface as notified=False — confirmed real bug fixed 2026-11-03: this
    call site used to set notified=True unconditionally right after calling
    send_match_digest(), regardless of whether the POST actually succeeded."""
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": True})
    prep = _prep()
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [{"id": "m1", "user_id": "u1", "job_id": "job-1", "tier1_score": 80,
                                      "tier2": {"why_fits": "- Great fit"}, "created_at": "2026-11-01T00:00:00+00:00",
                                      "notified_at": None}])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: False)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)

    assert result["notified"] is False
    [row] = fake_client.table("matches_v2").rows
    assert row["notified_at"] is None  # failed send must not stamp


def test_finish_user_matching_no_digest_when_email_on_match_false(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": False})
    prep = _prep()
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1", "tier1_score": 80, "tier2": {}}])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })

    def _should_not_run(*a, **k):
        raise AssertionError("must not send digest")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _should_not_run)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)
    assert result["notified"] is False


def test_finish_user_matching_digest_failure_does_not_raise(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": True})
    prep = _prep()
    fake_client.seed("jobs", [_job()])
    fake_client.seed("matches_v2", [{"id": "m1", "user_id": "u1", "job_id": "job-1", "tier1_score": 80, "tier2": {},
                                      "created_at": "2026-11-01T00:00:00+00:00", "notified_at": None}])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })

    def _raise(*a, **k):
        raise RuntimeError("digest edge function down")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _raise)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)
    assert result["notified"] is False  # swallowed, not raised


def test_finish_user_matching_tier2_pending_defers_digest(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}, "email_on_match": True})
    prep = _prep()
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "status": "pending", "batch_id": "batch_x", "considered": 1, "enriched": 0,
        "skipped_below_threshold": 0, "failed": 0,
    })

    def _should_not_run(*a, **k):
        raise AssertionError("must not send digest while tier 2 is pending")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _should_not_run)

    result = _finish_user_matching(fake_client, object(), user, prep, dry_run=False)
    assert result["notified"] is False


# ---------------------------------------------------------------------------
# _ADAPTER_REGISTRY — IAP disabled, its .fetch() (and thus its Google
# Sheets credentials) never touched; 80k/ProbablyGood remain active.
# ---------------------------------------------------------------------------

def test_adapter_registry_iap_is_disabled():
    from src.adapters.iap import IAPAdapter
    entries = dict(pipeline_v2._ADAPTER_REGISTRY)
    assert entries[IAPAdapter] is False


def test_adapter_registry_80k_probably_good_and_aac_are_enabled():
    from src.adapters.aac import AACAdapter
    from src.adapters.eighty_thousand_hours import EightyThousandHoursAdapter
    from src.adapters.probably_good import ProbablyGoodAdapter
    entries = dict(pipeline_v2._ADAPTER_REGISTRY)
    assert entries[EightyThousandHoursAdapter] is True
    assert entries[ProbablyGoodAdapter] is True
    assert entries[AACAdapter] is True


def test_adapter_registry_aac_is_last():
    from src.adapters.aac import AACAdapter
    assert pipeline_v2._ADAPTER_REGISTRY[-1][0] is AACAdapter


def test_adapters_derived_list_excludes_disabled_iap():
    from src.adapters.iap import IAPAdapter
    assert IAPAdapter not in pipeline_v2._ADAPTERS
    assert len(pipeline_v2._ADAPTERS) == 3


def test_adapters_derived_list_has_aac_last():
    from src.adapters.aac import AACAdapter
    assert pipeline_v2._ADAPTERS[-1] is AACAdapter


def test_run_scheduled_mode_logs_active_and_skipped_adapters(fake_client, monkeypatch, caplog):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)

    with caplog.at_level("INFO"):
        run(_client=fake_client, _anthropic_client=object())

    [log_line] = [r.message for r in caplog.records if r.message.startswith("Adapters active:")]
    assert "EightyThousandHoursAdapter" in log_line
    assert "ProbablyGoodAdapter" in log_line
    assert "AACAdapter" in log_line
    assert "IAPAdapter" in log_line  # named in the skipped half, not silently dropped


def test_run_manual_mode_does_not_log_adapter_line(fake_client, monkeypatch, caplog):
    """Manual mode never touches adapters at all — the active/skipped log line
    is scoped to the scheduled-mode adapter-fetch step, not printed as a
    no-op in a mode that never reaches it."""
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(skipped="test"))
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    with caplog.at_level("INFO"):
        run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    assert not any(r.message.startswith("Adapters active:") for r in caplog.records)


# ---------------------------------------------------------------------------
# run() — scheduled (full) mode vs manual DB-only mode
# ---------------------------------------------------------------------------

class _NoopAdapter:
    source_board = "noop"

    def fetch(self):
        from src.adapters.base import FetchResult
        return FetchResult(postings=[], fetched=0, nb_hits=None, complete=True)


def _patch_full_pipeline_noops(
    monkeypatch, *, ingest_result=None, enrich_result=None, embed_result=None,
    closure_result=None, tier1_result=None,
):
    from src.ingest import IngestCounts
    # Two, not three — matches the real active-adapter count now that IAP is
    # disabled (see _ADAPTER_REGISTRY); these are dummy classes either way, so
    # the exact count has no bearing on what this helper actually verifies.
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_NoopAdapter, _NoopAdapter])
    monkeypatch.setattr(pipeline_v2, "run_ingest", lambda postings, client, dry_run, **k: ingest_result or IngestCounts())
    monkeypatch.setattr(pipeline_v2, "close_jobs", lambda *a, **k: closure_result or {
        "closed": 0, "by_source": {}, "by_rule": {"complete_source_stale": 0, "safety_ceiling": 0},
    })
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", lambda **k: enrich_result or {
        "selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": True,
    })
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: embed_result or {
        "selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0,
    })
    # Phase A/B defaults: every user is a clean no-op unless a test overrides
    # _prepare_tier1_request/_finish_user_matching/rerank_tier1_batch itself —
    # mirrors the old single _match_and_notify_user override's simplicity.
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda client, user, jobs_pool, **k: _prep(user.user_id, skipped="test"))
    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", lambda *a, **k: tier1_result or {"status": "complete", "scored": 0, "failed": 0})
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda client, anthropic_client, user, prep, **k: {"user_id": user.user_id, "skipped": "test"})


# ---------------------------------------------------------------------------
# pipeline_runs open/close — run_metrics.run_id is a foreign key to
# pipeline_runs.run_id, so every stage's run_metrics write this run makes
# depends on this row existing FIRST. See _open_pipeline_run/_close_pipeline_run.
# ---------------------------------------------------------------------------

def test_run_opens_pipeline_runs_row_before_any_stage_runs(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)

    run(_client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "complete"
    assert row["trigger"] == "scheduled"
    assert row["started_at"] is not None
    assert row["completed_at"] is not None


def test_run_opens_pipeline_runs_row_with_same_run_id_as_run_metrics(fake_client, monkeypatch):
    """The actual bug this fixes: every run_metrics row this run writes must
    reference a run_id that a pipeline_runs row with that same id already
    exists for (run_metrics.run_id -> pipeline_runs.run_id FK)."""
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda client, user, jobs_pool, gate_rejections_out=None, **k: (
        gate_rejections_out.update({user.user_id: {"location": 1}}) if gate_rejections_out is not None else None,
        _prep(user.user_id, gate_survivor_count=0, job_cards=[]))[1])

    run(_client=fake_client, _anthropic_client=object())

    [pipeline_run] = fake_client.table("pipeline_runs").rows
    [metrics_row] = [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"]
    assert metrics_row["run_id"] == pipeline_run["run_id"]


def test_run_manual_mode_opens_pipeline_runs_row_with_trigger_manual(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(skipped="test"))
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["trigger"] == "manual"
    assert row["status"] == "complete"


def test_run_closes_pipeline_runs_row_as_failed_on_exception(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })

    def _boom(*a, **k):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep())
    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", lambda *a, **k: {"status": "complete", "scored": 0, "failed": 0})
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", _boom)

    with pytest.raises(RuntimeError):
        run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "failed"
    assert row["completed_at"] is not None


def test_run_no_users_manual_mode_still_opens_and_closes_pipeline_runs_row_failed(fake_client):
    with pytest.raises(RuntimeError):
        run(only_user_id="nonexistent", _client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "failed"


def test_run_dry_run_never_writes_pipeline_runs_row(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)

    run(dry_run=True, _client=fake_client, _anthropic_client=object())

    assert fake_client.table("pipeline_runs").rows == []


def test_run_scheduled_mode_calls_adapters_ingest_enrich_embed(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    calls = []

    class _TrackingAdapter:
        source_board = "tracking"

        def fetch(self):
            from src.adapters.base import FetchResult
            calls.append("adapter")
            return FetchResult(postings=[], fetched=0, nb_hits=None, complete=True)

    from src.ingest import IngestCounts
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_TrackingAdapter])
    monkeypatch.setattr(pipeline_v2, "run_ingest", lambda postings, client, dry_run, **k: (
        calls.append("ingest"), IngestCounts())[1])
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", lambda **k: (
        calls.append("enrich"), {"selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": True})[1])
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: (
        calls.append("embed"), {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0})[1])
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(skipped="test"))
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    run(_client=fake_client, _anthropic_client=object())

    assert calls == ["adapter", "ingest", "enrich", "embed"]


# ---------------------------------------------------------------------------
# source_fetch_log — one row per adapter per scheduled run, including on a
# failed fetch (complete=False), so closure has a fetch-completeness signal
# to key off per source board.
# ---------------------------------------------------------------------------

def test_run_writes_source_fetch_log_row_per_adapter(fake_client, monkeypatch):
    from src.adapters.base import FetchResult

    class _Adapter80k:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=5, nb_hits=5, complete=True)

    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_Adapter80k])

    run(_client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("source_fetch_log").rows
    assert row["source_board"] == "80000_hours"
    assert row["fetched"] == 5
    assert row["nb_hits"] == 5
    assert row["complete"] is True
    assert row["fetched_at"] is not None


def test_run_writes_source_fetch_log_row_with_complete_false_on_adapter_failure(fake_client, monkeypatch):
    class _FailingAdapter:
        source_board = "80000_hours"

        def fetch(self):
            raise RuntimeError("boom")

    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_FailingAdapter])

    run(_client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("source_fetch_log").rows
    assert row["source_board"] == "80000_hours"
    assert row["complete"] is False
    assert row["fetched"] == 0


def test_run_dry_run_writes_no_source_fetch_log(fake_client, monkeypatch):
    from src.adapters.base import FetchResult

    class _Adapter80k:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=5, nb_hits=5, complete=True)

    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_Adapter80k])

    run(dry_run=True, _client=fake_client, _anthropic_client=object())

    assert fake_client.table("source_fetch_log").rows == []


# ---------------------------------------------------------------------------
# close_jobs() wiring — called after ingest, before enrich, in scheduled
# mode only; complete_sources derived from fetch_log_rows (this run's actual
# results), not a historical query.
# ---------------------------------------------------------------------------

def test_run_calls_close_jobs_after_ingest_before_enrich(fake_client, monkeypatch):
    from src.ingest import IngestCounts

    calls = []
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "run_ingest", lambda *a, **k: (calls.append("ingest"), IngestCounts())[1])
    monkeypatch.setattr(pipeline_v2, "close_jobs", lambda *a, **k: (calls.append("closure"), {"closed": 0, "by_source": {}, "by_rule": {}})[1])
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", lambda **k: (calls.append("enrich"), {
        "selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": True,
    })[1])

    run(_client=fake_client, _anthropic_client=object())

    assert calls == ["ingest", "closure", "enrich"]


def test_run_passes_this_runs_complete_sources_to_close_jobs(fake_client, monkeypatch):
    from src.adapters.base import FetchResult

    class _CompleteAdapter:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=5, nb_hits=5, complete=True)

    class _IncompleteAdapter:
        source_board = "probably_good"

        def fetch(self):
            return FetchResult(postings=[], fetched=3, nb_hits=10, complete=False)

    captured = {}
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_CompleteAdapter, _IncompleteAdapter])
    monkeypatch.setattr(pipeline_v2, "close_jobs", lambda client, *, complete_sources, **k: (
        captured.update(complete_sources=complete_sources), {"closed": 0, "by_source": {}, "by_rule": {}})[1])

    run(_client=fake_client, _anthropic_client=object())

    assert captured["complete_sources"] == frozenset({"80000_hours"})


def test_run_passes_disabled_registry_boards_to_close_jobs(fake_client, monkeypatch):
    """disabled_boards must come from _ADAPTER_REGISTRY's disabled entries
    (IAP, today) — not _ADAPTERS, which already excludes them entirely and
    so can't be used to recover which ones were skipped."""
    from src.adapters.base import FetchResult
    from src.adapters.iap import IAPAdapter

    class _ActiveAdapter:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=0, nb_hits=0, complete=True)

    captured = {}
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_ActiveAdapter])
    monkeypatch.setattr(pipeline_v2, "_ADAPTER_REGISTRY", [(_ActiveAdapter, True), (IAPAdapter, False)])
    monkeypatch.setattr(pipeline_v2, "close_jobs", lambda client, *, disabled_boards, **k: (
        captured.update(disabled_boards=disabled_boards), {"closed": 0, "by_source": {}, "by_rule": {}})[1])

    run(_client=fake_client, _anthropic_client=object())

    assert captured["disabled_boards"] == frozenset({"iap"})


def test_run_dry_run_never_calls_close_jobs(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)

    def _should_not_run(*a, **k):
        raise AssertionError("dry_run must not call close_jobs")

    monkeypatch.setattr(pipeline_v2, "close_jobs", _should_not_run)

    run(dry_run=True, _client=fake_client, _anthropic_client=object())


# ---------------------------------------------------------------------------
# _check_source_fetch_health — identifies (does not raise for) sources with 2
# consecutive complete=False source_fetch_log rows. _raise_if_source_unhealthy
# is the separate, deferred step that actually fails the run — see run()'s
# call sites, all placed at the END of each return path so a broken source
# never blocks ingest/enrich/matching for a healthy one or an active user
# (the ProbablyGood AlgoliaSearchKey outage went unnoticed for 7 weeks
# precisely because nothing failed the step loud enough before).
# ---------------------------------------------------------------------------

def test_check_source_fetch_health_flags_two_consecutive_incomplete(fake_client):
    fake_client.seed("source_fetch_log", [
        {"source_board": "probably_good", "complete": False, "fetched_at": "2026-11-01T06:00:00+00:00"},
        {"source_board": "probably_good", "complete": False, "fetched_at": "2026-11-02T06:00:00+00:00"},
    ])

    assert pipeline_v2._check_source_fetch_health(fake_client, ["probably_good"]) == ["probably_good"]


def test_check_source_fetch_health_tolerates_one_off_incomplete(fake_client):
    fake_client.seed("source_fetch_log", [
        {"source_board": "probably_good", "complete": True, "fetched_at": "2026-11-01T06:00:00+00:00"},
        {"source_board": "probably_good", "complete": False, "fetched_at": "2026-11-02T06:00:00+00:00"},
    ])

    assert pipeline_v2._check_source_fetch_health(fake_client, ["probably_good"]) == []


def test_check_source_fetch_health_ignores_fewer_than_two_rows(fake_client):
    fake_client.seed("source_fetch_log", [
        {"source_board": "probably_good", "complete": False, "fetched_at": "2026-11-02T06:00:00+00:00"},
    ])

    assert pipeline_v2._check_source_fetch_health(fake_client, ["probably_good"]) == []


def test_raise_if_source_unhealthy_raises_when_nonempty():
    with pytest.raises(RuntimeError, match="probably_good"):
        pipeline_v2._raise_if_source_unhealthy(["probably_good"])


def test_raise_if_source_unhealthy_noop_when_empty():
    pipeline_v2._raise_if_source_unhealthy([])  # must not raise


def test_check_source_fetch_health_ignores_disabled_source_not_passed_in(fake_client):
    """Only checks the source_boards actually passed — a disabled adapter's
    old history must not fail a run that no longer includes it."""
    fake_client.seed("source_fetch_log", [
        {"source_board": "iap", "complete": False, "fetched_at": "2026-11-01T06:00:00+00:00"},
        {"source_board": "iap", "complete": False, "fetched_at": "2026-11-02T06:00:00+00:00"},
    ])

    assert pipeline_v2._check_source_fetch_health(fake_client, ["80000_hours", "probably_good"]) == []


def test_run_fails_when_enabled_source_incomplete_two_runs_straight(fake_client, monkeypatch):
    from src.adapters.base import FetchResult

    class _FailingAdapter:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=0, nb_hits=100, complete=False)

    fake_client.seed("source_fetch_log", [
        {"source_board": "80000_hours", "complete": False, "fetched_at": "2026-10-31T06:00:00+00:00"},
    ])
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_FailingAdapter])

    with pytest.raises(RuntimeError, match="80000_hours"):
        run(_client=fake_client, _anthropic_client=object())


def test_run_still_matches_healthy_users_before_failing_on_unhealthy_source(fake_client, monkeypatch):
    """The actual bug this placement fixes: a broken source must fail the
    step, but must NOT prevent matching from running for active users first
    — raising eagerly (right after the fetch loop) would have skipped this
    entirely."""
    from src.adapters.base import FetchResult

    class _FailingAdapter:
        source_board = "80000_hours"

        def fetch(self):
            return FetchResult(postings=[], fetched=0, nb_hits=100, complete=False)

    matched_users = []
    fake_client.seed("source_fetch_log", [
        {"source_board": "80000_hours", "complete": False, "fetched_at": "2026-10-31T06:00:00+00:00"},
    ])
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_FailingAdapter])
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda client, anthropic_client, user, prep, **k: (
        matched_users.append(user.user_id), {"user_id": user.user_id, "skipped": "test"})[1])

    with pytest.raises(RuntimeError, match="80000_hours"):
        run(_client=fake_client, _anthropic_client=object())

    assert matched_users == ["u1"]


def test_run_manual_mode_skips_adapters_ingest_enrich(fake_client, monkeypatch):
    _seed_active_user(fake_client)

    def _should_not_run(*a, **k):
        raise AssertionError("manual mode must not scrape/ingest/enrich")

    monkeypatch.setattr(pipeline_v2, "run_ingest", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(skipped="test"))
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    result = run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    assert result["ingest"] is None
    assert result["enrich"] is None


def test_run_manual_mode_never_submits_tier1_batch(fake_client, monkeypatch):
    """Manual mode still runs Phase A/B/C for its one user (Tier 1/Tier 2 ARE
    part of the manual path's cost), but never touches adapters/ingest/enrich
    — covered by the sibling test above. This test just confirms Phase B
    still runs for the single user in manual mode."""
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "gate_survivors_with_rejections", lambda jobs, profile: ([], {}))
    calls = []
    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", lambda *a, **k: (
        calls.append("tier1_batch"), {"status": "complete", "scored": 0, "failed": 0})[1])
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", lambda *a, **k: [])

    run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    assert calls == ["tier1_batch"]


def test_run_manual_mode_sets_quota_running_then_complete(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(skipped="test"))
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    [quota_row] = fake_client.table("search_quota").rows
    assert quota_row["status"] == "complete"


def test_run_manual_mode_sets_quota_failed_on_exception(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })

    def _boom(*a, **k):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep())
    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", lambda *a, **k: {"status": "complete", "scored": 0, "failed": 0})
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", _boom)

    with pytest.raises(RuntimeError):
        run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    [quota_row] = fake_client.table("search_quota").rows
    assert quota_row["status"] == "failed"


def test_run_scheduled_mode_never_touches_quota(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)

    run(_client=fake_client, _anthropic_client=object())

    assert fake_client.table("search_quota").rows == []


def test_run_no_users_scheduled_mode_returns_empty_no_error(fake_client, monkeypatch):
    _patch_full_pipeline_noops(monkeypatch)

    result = run(_client=fake_client, _anthropic_client=object())

    assert result["users"] == []


def test_run_no_users_manual_mode_raises_and_sets_quota_failed(fake_client):
    with pytest.raises(RuntimeError):
        run(only_user_id="nonexistent", _client=fake_client, _anthropic_client=object())

    [quota_row] = fake_client.table("search_quota").rows
    assert quota_row["status"] == "failed"


def test_run_writes_aggregate_gate_rejections_run_metrics_row(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda client, user, jobs_pool, gate_rejections_out=None, **k: (
        gate_rejections_out.update({user.user_id: {"location": 2}}) if gate_rejections_out is not None else None,
        _prep(user.user_id, gate_survivor_count=0, job_cards=[]))[1])

    run(_client=fake_client, _anthropic_client=object())

    [row] = [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"]
    assert row["gate_rejections"] == {"u1": {"location": 2}}
    assert row["trigger"] == "scheduled"
    assert row["run_id"]


def test_run_skips_gate_rejections_row_when_nothing_rejected(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)

    run(_client=fake_client, _anthropic_client=object())

    assert [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"] == []


def test_run_survives_every_run_metrics_insert_failing(fake_client, monkeypatch):
    """The real incident this guards against: close_jobs() (and every other
    stage) writes its own run_metrics row via insert_run_metrics(), which
    never raises — a schema-drift failure there must not abort the run
    before enrich/matching ever gets a chance to run, the way a raw
    .insert().execute() call used to."""
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda client, user, jobs_pool, gate_rejections_out=None, **k: (
        gate_rejections_out.update({user.user_id: {"location": 2}}) if gate_rejections_out is not None else None,
        _prep(user.user_id, gate_survivor_count=0, job_cards=[]))[1])
    fake_client.table("run_metrics").raise_on_execute["insert"] = Exception("PGRST204: boom")

    result = run(_client=fake_client, _anthropic_client=object())  # must not raise

    assert fake_client.table("run_metrics").rows == []
    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "complete"
    assert result["users"]


# ---------------------------------------------------------------------------
# Enrichment 'pending' gate — enrich_jobs() returned status='pending' (its
# Batches API poll budget was exhausted, not an error). run() must stop
# cleanly before profile-embedding refresh and the per-user matching loop —
# never match on a partially-enriched jobs pool.
# ---------------------------------------------------------------------------

def test_run_stops_before_per_user_loop_when_enrichment_pending(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    pending_result = {"status": "pending", "batch_id": "batch_x", "request_counts": {"processing": 3}}
    _patch_full_pipeline_noops(monkeypatch, enrich_result=pending_result)

    def _should_not_run(*a, **k):
        raise AssertionError("must not reach per-user matching when enrichment is pending")

    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", _should_not_run)

    result = run(_client=fake_client, _anthropic_client=object())

    assert result["users"] == []
    assert result["enrich"] == pending_result
    assert result["status"] == "incomplete"


def test_run_pending_closes_pipeline_runs_as_incomplete_not_failed(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch, enrich_result={"status": "pending", "batch_id": "batch_x"})
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    run(_client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "incomplete"
    assert row["status"] != "failed"
    assert row["completed_at"] is not None


def test_run_pending_does_not_write_gate_rejections_row(fake_client, monkeypatch):
    """The match-stage gate_rejections aggregate row only makes sense once
    the per-user loop actually ran — it must not exist at all when the run
    stopped before ever reaching that loop."""
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch, enrich_result={"status": "pending", "batch_id": "batch_x"})

    run(_client=fake_client, _anthropic_client=object())

    assert [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"] == []


def test_run_complete_enrichment_proceeds_normally_to_matching(fake_client, monkeypatch):
    """Sanity check on the gate's condition itself: a normal status='complete'
    enrich_result must NOT trip the pending gate."""
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch, enrich_result={
        "status": "complete", "selected": 0, "enriched": 0, "stale": 0, "requeued": 0,
        "embeddings_skipped": True, "deferred": 0,
    })

    result = run(_client=fake_client, _anthropic_client=object())

    assert len(result["users"]) == 1
    assert result.get("status") != "incomplete"


def test_run_dry_run_skips_tier1_and_tier2_but_still_gates(fake_client, monkeypatch):
    """dry_run still computes gate_v2/vector rerank (free — no LLM cost) but
    never reaches Tier 1/Tier 2 (the only steps with real $ cost here)."""
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep(gate_survivor_count=3))
    # _patch_full_pipeline_noops defaults _finish_user_matching to a no-op —
    # restore the real one so this test can see its actual dry_run behavior.
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", _finish_user_matching)

    def _should_not_run(*a, **k):
        raise AssertionError("dry_run must not call Tier 1/Tier 2")

    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "fetch_tier2_candidates", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", _should_not_run)

    result = run(dry_run=True, _client=fake_client, _anthropic_client=object())

    assert len(result["users"]) == 1
    assert result["users"][0]["gate_survivors"] == 3


# ---------------------------------------------------------------------------
# Tier 1 'pending' gate (Phase 2) — mirrors the enrichment pending gate: ONE
# Batches submission covers every user, so if it doesn't finish within
# budget, run() must stop before Tier 2/digest for EVERYONE, not just the
# user(s) whose chunks were in that batch.
# ---------------------------------------------------------------------------

def test_run_stops_before_tier2_when_tier1_pending(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch, tier1_result={
        "status": "pending", "batch_id": "batch_x", "request_counts": {"processing": 5},
    })

    def _should_not_run(*a, **k):
        raise AssertionError("must not reach Tier 2/digest when Tier 1 is pending")

    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", _should_not_run)

    result = run(_client=fake_client, _anthropic_client=object())

    assert result["users"] == []
    assert result["status"] == "incomplete"


def test_run_tier1_pending_closes_pipeline_runs_as_incomplete(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch, tier1_result={"status": "pending", "batch_id": "batch_x"})
    monkeypatch.setattr(pipeline_v2, "_finish_user_matching", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    run(_client=fake_client, _anthropic_client=object())

    [row] = fake_client.table("pipeline_runs").rows
    assert row["status"] == "incomplete"


def test_run_tier1_not_submitted_in_dry_run(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_prepare_tier1_request", lambda *a, **k: _prep())

    def _should_not_run(*a, **k):
        raise AssertionError("dry_run must never submit a Tier 1 batch")

    monkeypatch.setattr(pipeline_v2, "rerank_tier1_batch", _should_not_run)

    run(dry_run=True, _client=fake_client, _anthropic_client=object())
