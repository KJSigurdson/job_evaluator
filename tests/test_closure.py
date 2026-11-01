"""Tests for src/closure.py: plan_closures()/close_jobs() — jobs.status='closed'
for postings no longer seen, per the two rules described in that module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.closure import close_jobs, plan_closures


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _job(job_id: str, *, source_board: str, last_seen_days_ago: float, status: str = "enriched") -> dict:
    return {
        "id": job_id, "canonical_url": f"https://example.org/{job_id}", "source_board": source_board,
        "content_hash": "irrelevant", "status": status, "last_seen": _iso(last_seen_days_ago),
    }


# ---------------------------------------------------------------------------
# Rule (a): source's fetch THIS RUN (complete_sources, caller-supplied — see
# src/pipeline_v2.py's call site) was complete=true + job stale >3 days
# ---------------------------------------------------------------------------

def test_plan_closures_closes_stale_job_from_this_runs_complete_source(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])

    plan = plan_closures(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert plan["by_rule"]["complete_source_stale"] == ["job-1"]
    assert plan["by_rule"]["safety_ceiling"] == []
    assert plan["by_source"] == {"80000_hours": 1}
    assert plan["total"] == 1


def test_plan_closures_does_not_close_fresh_job_from_complete_source(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=1)])

    plan = plan_closures(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert plan["total"] == 0


def test_plan_closures_does_not_close_under_rule_a_when_source_not_in_complete_sources(fake_client):
    """A source NOT passed in complete_sources (its fetch this run was
    incomplete, or this run didn't even include it) must not have its
    missing postings closed under rule (a) — the fetch itself may have
    missed them, not the source."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=10)])

    plan = plan_closures(fake_client, complete_sources=frozenset())

    assert plan["by_rule"]["complete_source_stale"] == []
    # Still caught by the safety ceiling once it's old enough — just not THIS rule.
    assert plan["by_rule"]["safety_ceiling"] == []  # 10 days < 30-day ceiling


def test_plan_closures_defaults_to_no_complete_sources(fake_client):
    """complete_sources defaults to empty — a caller with no "this run" to
    speak of (an ad-hoc review script) gets rule (b) only, never rule (a)."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=10)])

    plan = plan_closures(fake_client)

    assert plan["total"] == 0


# ---------------------------------------------------------------------------
# Rule (b): safety ceiling, regardless of source
# ---------------------------------------------------------------------------

def test_plan_closures_safety_ceiling_closes_old_job_regardless_of_complete_sources(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="iap", last_seen_days_ago=31)])

    plan = plan_closures(fake_client, complete_sources=frozenset())

    assert plan["by_rule"]["safety_ceiling"] == ["job-1"]
    assert plan["by_rule"]["complete_source_stale"] == []
    assert plan["total"] == 1


def test_plan_closures_safety_ceiling_does_not_double_count_rule_a_job(fake_client):
    """A job stale enough to qualify under BOTH rules is counted once, under
    rule (a) — rule (a) already fully explains it."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=40)])

    plan = plan_closures(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert plan["by_rule"]["complete_source_stale"] == ["job-1"]
    assert plan["by_rule"]["safety_ceiling"] == []
    assert plan["total"] == 1


def test_plan_closures_mixed_sources_and_rules(fake_client):
    fake_client.seed("jobs", [
        _job("fresh", source_board="80000_hours", last_seen_days_ago=1),
        _job("stale-complete-source", source_board="80000_hours", last_seen_days_ago=4),
        _job("stale-no-source-signal", source_board="probably_good", last_seen_days_ago=10),
        _job("ancient-no-source-signal", source_board="iap", last_seen_days_ago=35),
    ])

    plan = plan_closures(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert set(plan["by_rule"]["complete_source_stale"]) == {"stale-complete-source"}
    assert set(plan["by_rule"]["safety_ceiling"]) == {"ancient-no-source-signal"}
    assert plan["total"] == 2
    assert plan["by_source"] == {"80000_hours": 1, "iap": 1}


def test_plan_closures_ignores_already_closed_and_stale_jobs(fake_client):
    fake_client.seed("jobs", [
        _job("already-closed", source_board="80000_hours", last_seen_days_ago=100, status="closed"),
        _job("already-stale", source_board="80000_hours", last_seen_days_ago=100, status="stale"),
    ])

    plan = plan_closures(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert plan["total"] == 0


# ---------------------------------------------------------------------------
# close_jobs() — applies the plan for real, writes a run_metrics row
# ---------------------------------------------------------------------------

def test_close_jobs_writes_closed_status(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])

    result = close_jobs(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert result == {"closed": 1, "by_source": {"80000_hours": 1}, "by_rule": {"complete_source_stale": 1, "safety_ceiling": 0}}
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "closed"


def test_close_jobs_noop_when_nothing_qualifies(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=1)])

    result = close_jobs(fake_client, complete_sources=frozenset({"80000_hours"}))

    assert result == {"closed": 0, "by_source": {}, "by_rule": {"complete_source_stale": 0, "safety_ceiling": 0}}
    assert fake_client.table("jobs").rows[0]["status"] == "enriched"


def test_close_jobs_upsert_payload_carries_not_null_columns(fake_client):
    """canonical_url/source_board/content_hash must be carried through the
    upsert payload even though this write only means to touch `status` — same
    upsert-on-INSERT-branch NOT NULL quirk as every other chunked_upsert call
    site in this codebase."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])

    close_jobs(fake_client, complete_sources=frozenset({"80000_hours"}))

    [payload] = fake_client.table("jobs").upsert_calls
    assert set(payload[0].keys()) == {"id", "canonical_url", "source_board", "content_hash", "status"}


def test_close_jobs_writes_run_metrics_closure_summary(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])

    close_jobs(fake_client, complete_sources=frozenset({"80000_hours"}), run_id="run-abc", trigger="scheduled")

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["stage"] == "closure"
    assert metrics["run_id"] == "run-abc"
    assert metrics["trigger"] == "scheduled"
    assert metrics["closure_summary"] == {
        "closed": 1, "by_source": {"80000_hours": 1},
        "by_rule": {"complete_source_stale": 1, "safety_ceiling": 0},
    }


def test_close_jobs_writes_run_metrics_even_when_nothing_closed(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=1)])

    close_jobs(fake_client, complete_sources=frozenset({"80000_hours"}))

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["closure_summary"]["closed"] == 0
