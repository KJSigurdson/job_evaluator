"""Tests for src/pipeline_v2.py: orchestration/wiring for both the scheduled
full pipeline and the manual DB-only path, skip logic for incomplete
profiles, already-matched exclusion, digest payload mapping, and
search_quota bookkeeping.

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
from src.models.tier1_rerank import JobScore
from src.pipeline_v2 import _build_digest_payload, _match_and_notify_user, run


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
        raw_description="text", source_board="80000_hours",
        enrichment={"summary": "Lead data.", "seniority": ["senior"], "cause_area": ["global health"],
                    "skill_tags": ["SQL"]},
    )
    defaults.update(overrides)
    return defaults


def _seed_active_user(fake_client, *, profile_embedding=None, **profile_overrides):
    row = _profile_row(**profile_overrides)
    row["profile_embedding"] = profile_embedding
    fake_client.seed("profiles", [row])
    fake_client.seed("scoring_weights", [_weights_row()])


# ---------------------------------------------------------------------------
# _build_digest_payload — v2 matches_v2 rows -> v1's exact digest shape
# ---------------------------------------------------------------------------

def test_build_digest_payload_maps_v2_fields_to_v1_shape():
    matches_v2_rows = [{"job_id": "job-1", "tier1_score": 82, "tier2": {"why_fits": "- Strong fit"}}]
    jobs_by_id = {"job-1": _job()}

    payload = _build_digest_payload(matches_v2_rows, jobs_by_id)

    assert payload == [{
        "title": "Head of Data", "org": "GiveDirectly", "location": "Remote, Global",
        "fit_score": 82, "why_fits": "- Strong fit", "url": "https://example.org/job-1",
    }]


def test_build_digest_payload_skips_row_with_no_matching_job():
    payload = _build_digest_payload([{"job_id": "missing", "tier1_score": 80, "tier2": {}}], {})
    assert payload == []


# ---------------------------------------------------------------------------
# _match_and_notify_user — skip logic
# ---------------------------------------------------------------------------

def _user_context(profile: dict, rubric: dict | None = None):
    from src.schemas import UserContext
    # threshold: 0.65 mirrors _weights_row()'s old-style default and the "below 65"
    # comments on tests further down that pre-date per-user threshold wiring —
    # 0.65 * 100 == 65, so those tests' score=40/score=80 fixtures still mean what
    # their comments say without every one of them needing its own rubric override.
    return UserContext(user_id="u1", profile=profile, rubric=rubric or {"threshold": 0.65})


def test_match_and_notify_user_skips_empty_profile(fake_client):
    user = _user_context({"career_goals": None, "cause_priorities": None, "experience_inventory": None,
                           "skills": None, "values_notes": None, "hard_constraints": {}})

    result = _match_and_notify_user(fake_client, object(), user, [], dry_run=False)

    assert result == {"user_id": "u1", "skipped": "empty_profile"}


def test_match_and_notify_user_skips_no_profile_embedding(fake_client):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": None}])

    result = _match_and_notify_user(fake_client, object(), user, [], dry_run=False)

    assert result == {"user_id": "u1", "skipped": "no_profile_embedding"}


def test_match_and_notify_user_excludes_already_matched_jobs(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-already-matched"}])

    seen_jobs_arg = {}

    def fake_match_user(client, *, user_id, profile, profile_embedding, jobs, top_n):
        seen_jobs_arg["jobs"] = jobs
        return {"gate_survivor_count": 0, "top": []}

    monkeypatch.setattr(pipeline_v2, "match_user", fake_match_user)

    jobs_pool = [_job("job-already-matched"), _job("job-new")]
    _match_and_notify_user(fake_client, object(), user, jobs_pool, dry_run=False)

    job_ids_passed = [j["id"] for j in seen_jobs_arg["jobs"]]
    assert "job-already-matched" not in job_ids_passed
    assert "job-new" in job_ids_passed


def test_match_and_notify_user_no_top_returns_zero_counts(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 5, "top": []})

    result = _match_and_notify_user(fake_client, object(), user, [], dry_run=False)

    assert result == {"user_id": "u1", "gate_survivors": 5, "tier1_scored": 0, "tier2_enriched": 0, "notified": False}


def test_match_and_notify_user_dry_run_skips_tier1_tier2_entirely(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})

    def _should_not_run(*args, **kwargs):
        raise AssertionError("dry_run must not call Tier 1/Tier 2")

    monkeypatch.setattr(pipeline_v2, "rerank_tier1", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", _should_not_run)

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=True)

    assert result["tier1_scored"] == 0
    assert result["notified"] is False


# ---------------------------------------------------------------------------
# Per-user insert_threshold — scoring_weights.insert_threshold (0-1 in storage)
# scaled to tier1_score's 0-100 range, read fresh per user, no near-miss tier.
# ---------------------------------------------------------------------------

def test_match_and_notify_user_scales_threshold_from_0_1_to_0_100(fake_client, monkeypatch):
    user = _user_context(
        {"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": False},
        rubric={"threshold": 0.8},
    )
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=80, reason="x")]},
    })

    captured = {}
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: (
        captured.update(k), {"considered": 1, "enriched": 0, "skipped_below_threshold": 1, "failed": 0},
    )[1])

    _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)

    assert captured["threshold"] == 80.0  # 0.8 * 100, not the raw 0.8


def test_match_and_notify_user_uses_this_users_own_threshold_not_a_global_constant(fake_client, monkeypatch):
    """Same tier1_score (70) against two different per-user thresholds must
    qualify for one user and not the other — proves the threshold is read from
    this user's own rubric, not a shared/global constant."""
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=70, reason="x")]},
    })
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })
    digest_calls = []
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: digest_calls.append(user_id))
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1", "tier1_score": 70, "tier2": {}}])

    lenient_user = _user_context(
        {"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": True},
        rubric={"threshold": 0.6},  # 60 -> 70 qualifies
    )
    result_lenient = _match_and_notify_user(fake_client, object(), lenient_user, [_job()], dry_run=False)
    assert result_lenient["notified"] is True

    digest_calls.clear()
    strict_user = _user_context(
        {"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": True},
        rubric={"threshold": 0.85},  # 85 -> 70 does not qualify
    )
    result_strict = _match_and_notify_user(fake_client, object(), strict_user, [_job()], dry_run=False)
    assert result_strict["notified"] is False
    assert digest_calls == []


def test_match_and_notify_user_calls_tier1_then_tier2(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": False})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})

    calls = []
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: (
        calls.append("tier1"),
        {"u1": {"scores": [JobScore(job_id="job-1", score=80, reason="Great fit")]}},
    )[1])
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: (
        calls.append("tier2"), {"considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0},
    )[1])

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)

    assert calls == ["tier1", "tier2"]
    assert result["tier1_scored"] == 1
    assert result["tier2_enriched"] == 1


def test_match_and_notify_user_records_gate_rejections_into_out_param(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {
        "gate_survivor_count": 0, "top": [], "gate_rejection_counts": {"location": 3, "seniority": 1},
    })

    gate_rejections: dict = {}
    _match_and_notify_user(fake_client, object(), user, [], dry_run=False, gate_rejections_out=gate_rejections)

    assert gate_rejections == {"u1": {"location": 3, "seniority": 1}}


def test_match_and_notify_user_no_rejections_not_recorded(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 5, "top": []})

    gate_rejections: dict = {}
    _match_and_notify_user(fake_client, object(), user, [], dry_run=False, gate_rejections_out=gate_rejections)

    assert gate_rejections == {}


def test_match_and_notify_user_sends_digest_when_email_on_match_and_new_matches(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": True})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1", "tier1_score": 80,
                                      "tier2": {"why_fits": "- Great fit"}}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=80, reason="Great fit")]},
    })
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })

    digest_calls = []
    monkeypatch.setattr(pipeline_v2, "send_match_digest", lambda user_id, matches: digest_calls.append((user_id, matches)))

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)

    assert result["notified"] is True
    [(user_id, matches)] = digest_calls
    assert user_id == "u1"
    assert matches == [{"title": "Head of Data", "org": "GiveDirectly", "location": "Remote, Global",
                         "fit_score": 80, "why_fits": "- Great fit", "url": "https://example.org/job-1"}]


def test_match_and_notify_user_no_digest_when_email_on_match_false(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": False})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1", "tier1_score": 80, "tier2": {}}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=80, reason="x")]},
    })
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })

    def _should_not_run(*a, **k):
        raise AssertionError("must not send digest")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _should_not_run)

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)
    assert result["notified"] is False


def test_match_and_notify_user_no_digest_when_no_qualifying_matches(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": True})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=40, reason="Below threshold")]},  # below 65
    })
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 0, "skipped_below_threshold": 1, "failed": 0,
    })

    def _should_not_run(*a, **k):
        raise AssertionError("must not send digest")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _should_not_run)

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)
    assert result["notified"] is False


def test_match_and_notify_user_digest_failure_does_not_raise(fake_client, monkeypatch):
    user = _user_context({"career_goals": "x", "hard_constraints": {"location": {}, "seniority": {}}, "email_on_match": True})
    fake_client.seed("profiles", [{"user_id": "u1", "profile_embedding": "[0.1]"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1", "tier1_score": 80, "tier2": {}}])
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 1, "top": [_job()]})
    monkeypatch.setattr(pipeline_v2, "rerank_tier1", lambda *a, **k: {
        "u1": {"scores": [JobScore(job_id="job-1", score=80, reason="x")]},
    })
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", lambda *a, **k: {
        "considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0,
    })

    def _raise(*a, **k):
        raise RuntimeError("digest edge function down")

    monkeypatch.setattr(pipeline_v2, "send_match_digest", _raise)

    result = _match_and_notify_user(fake_client, object(), user, [_job()], dry_run=False)
    assert result["notified"] is False  # swallowed, not raised


# ---------------------------------------------------------------------------
# run() — scheduled (full) mode vs manual DB-only mode
# ---------------------------------------------------------------------------

class _NoopAdapter:
    def fetch(self):
        return []


def _patch_full_pipeline_noops(monkeypatch, *, ingest_result=None, enrich_result=None, embed_result=None):
    from src.ingest import IngestCounts
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_NoopAdapter, _NoopAdapter, _NoopAdapter])
    monkeypatch.setattr(pipeline_v2, "run_ingest", lambda postings, client, dry_run, **k: ingest_result or IngestCounts())
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", lambda **k: enrich_result or {
        "selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": True,
    })
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: embed_result or {
        "selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0,
    })


def test_run_scheduled_mode_calls_adapters_ingest_enrich_embed(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    calls = []

    class _TrackingAdapter:
        def fetch(self):
            calls.append("adapter")
            return []

    from src.ingest import IngestCounts
    monkeypatch.setattr(pipeline_v2, "_ADAPTERS", [_TrackingAdapter])
    monkeypatch.setattr(pipeline_v2, "run_ingest", lambda postings, client, dry_run, **k: (
        calls.append("ingest"), IngestCounts())[1])
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", lambda **k: (
        calls.append("enrich"), {"selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": True})[1])
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: (
        calls.append("embed"), {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0})[1])
    monkeypatch.setattr(pipeline_v2, "_match_and_notify_user", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    run(_client=fake_client, _anthropic_client=object())

    assert calls == ["adapter", "ingest", "enrich", "embed"]


def test_run_manual_mode_skips_adapters_ingest_enrich(fake_client, monkeypatch):
    _seed_active_user(fake_client)

    def _should_not_run(*a, **k):
        raise AssertionError("manual mode must not scrape/ingest/enrich")

    monkeypatch.setattr(pipeline_v2, "run_ingest", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_enrich_jobs", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_match_and_notify_user", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

    result = run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    assert result["ingest"] is None
    assert result["enrich"] is None


def test_run_manual_mode_sets_quota_running_then_complete(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    monkeypatch.setattr(pipeline_v2, "backfill_profile_embeddings", lambda *a, **k: {
        "selected": 1, "embedded": 0, "skipped_unchanged": 1, "skipped_empty": 0,
    })
    monkeypatch.setattr(pipeline_v2, "_match_and_notify_user", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

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

    monkeypatch.setattr(pipeline_v2, "_match_and_notify_user", _boom)

    with pytest.raises(RuntimeError):
        run(only_user_id="u1", _client=fake_client, _anthropic_client=object())

    [quota_row] = fake_client.table("search_quota").rows
    assert quota_row["status"] == "failed"


def test_run_scheduled_mode_never_touches_quota(fake_client, monkeypatch):
    _seed_active_user(fake_client)
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "_match_and_notify_user", lambda *a, **k: {"user_id": "u1", "skipped": "test"})

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
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {
        "gate_survivor_count": 0, "top": [], "gate_rejection_counts": {"location": 2},
    })

    run(_client=fake_client, _anthropic_client=object())

    [row] = [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"]
    assert row["gate_rejections"] == {"u1": {"location": 2}}
    assert row["trigger"] == "scheduled"
    assert row["run_id"]


def test_run_skips_gate_rejections_row_when_nothing_rejected(fake_client, monkeypatch):
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 5, "top": []})

    run(_client=fake_client, _anthropic_client=object())

    assert [r for r in fake_client.table("run_metrics").rows if r.get("stage") == "match"] == []


def test_run_dry_run_skips_tier1_and_tier2_but_still_gates(fake_client, monkeypatch):
    """dry_run still computes gate_v2/vector rerank (free — no LLM cost) but
    never reaches Tier 1/Tier 2 (the only steps with real $ cost here)."""
    _seed_active_user(fake_client, profile_embedding="[0.1]")
    _patch_full_pipeline_noops(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "match_user", lambda *a, **k: {"gate_survivor_count": 3, "top": [_job()]})

    def _should_not_run(*a, **k):
        raise AssertionError("dry_run must not call Tier 1/Tier 2")

    monkeypatch.setattr(pipeline_v2, "rerank_tier1", _should_not_run)
    monkeypatch.setattr(pipeline_v2, "run_tier2_enrich", _should_not_run)

    result = run(dry_run=True, _client=fake_client, _anthropic_client=object())

    assert len(result["users"]) == 1
    assert result["users"][0]["gate_survivors"] == 3
    assert result["users"][0]["tier1_scored"] == 0
