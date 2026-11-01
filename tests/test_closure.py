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


def _fetch_log(source_board: str, *, complete: bool, hours_ago: float = 1) -> dict:
    return {
        "source_board": source_board, "complete": complete,
        "fetched_at": (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(),
        "fetched": 100, "nb_hits": 100,
    }


# ---------------------------------------------------------------------------
# Rule (a): source confirmed complete recently + job stale >3 days
# ---------------------------------------------------------------------------

def test_plan_closures_closes_stale_job_from_recently_complete_source(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    plan = plan_closures(fake_client)

    assert plan["by_rule"]["complete_source_stale"] == ["job-1"]
    assert plan["by_rule"]["safety_ceiling"] == []
    assert plan["by_source"] == {"80000_hours": 1}
    assert plan["total"] == 1


def test_plan_closures_does_not_close_fresh_job_from_complete_source(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=1)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    plan = plan_closures(fake_client)

    assert plan["total"] == 0


def test_plan_closures_does_not_close_under_rule_a_when_source_last_fetch_incomplete(fake_client):
    """A source whose most recent fetch was incomplete must not have its
    missing postings closed under rule (a) — the fetch itself may have
    missed them, not the source."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=10)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=False)])

    plan = plan_closures(fake_client)

    assert plan["by_rule"]["complete_source_stale"] == []
    # Still caught by the safety ceiling once it's old enough — just not THIS rule.
    assert plan["by_rule"]["safety_ceiling"] == []  # 10 days < 30-day ceiling


def test_plan_closures_ignores_stale_complete_signal_outside_window(fake_client):
    """A complete=True fetch_log row from weeks ago doesn't permanently
    green-light rule (a) for that source — only a RECENT complete signal
    counts (_COMPLETE_SIGNAL_WINDOW_HOURS)."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=10)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True, hours_ago=24 * 10)])

    plan = plan_closures(fake_client)

    assert plan["by_rule"]["complete_source_stale"] == []


# ---------------------------------------------------------------------------
# Rule (b): safety ceiling, regardless of source
# ---------------------------------------------------------------------------

def test_plan_closures_safety_ceiling_closes_old_job_even_without_fetch_log(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="iap", last_seen_days_ago=31)])
    # No source_fetch_log rows at all for "iap" (e.g. a disabled adapter).

    plan = plan_closures(fake_client)

    assert plan["by_rule"]["safety_ceiling"] == ["job-1"]
    assert plan["by_rule"]["complete_source_stale"] == []
    assert plan["total"] == 1


def test_plan_closures_safety_ceiling_does_not_double_count_rule_a_job(fake_client):
    """A job stale enough to qualify under BOTH rules is counted once, under
    rule (a) — rule (a) already fully explains it."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=40)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    plan = plan_closures(fake_client)

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
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    plan = plan_closures(fake_client)

    assert set(plan["by_rule"]["complete_source_stale"]) == {"stale-complete-source"}
    assert set(plan["by_rule"]["safety_ceiling"]) == {"ancient-no-source-signal"}
    assert plan["total"] == 2
    assert plan["by_source"] == {"80000_hours": 1, "iap": 1}


def test_plan_closures_ignores_already_closed_and_stale_jobs(fake_client):
    fake_client.seed("jobs", [
        _job("already-closed", source_board="80000_hours", last_seen_days_ago=100, status="closed"),
        _job("already-stale", source_board="80000_hours", last_seen_days_ago=100, status="stale"),
    ])

    plan = plan_closures(fake_client)

    assert plan["total"] == 0


# ---------------------------------------------------------------------------
# close_jobs() — applies the plan for real
# ---------------------------------------------------------------------------

def test_close_jobs_writes_closed_status(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    result = close_jobs(fake_client)

    assert result == {"closed": 1, "by_source": {"80000_hours": 1}, "by_rule": {"complete_source_stale": 1, "safety_ceiling": 0}}
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "closed"


def test_close_jobs_noop_when_nothing_qualifies(fake_client):
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=1)])

    result = close_jobs(fake_client)

    assert result == {"closed": 0, "by_source": {}, "by_rule": {"complete_source_stale": 0, "safety_ceiling": 0}}
    assert fake_client.table("jobs").rows[0]["status"] == "enriched"


def test_close_jobs_upsert_payload_carries_not_null_columns(fake_client):
    """canonical_url/source_board/content_hash must be carried through the
    upsert payload even though this write only means to touch `status` — same
    upsert-on-INSERT-branch NOT NULL quirk as every other chunked_upsert call
    site in this codebase."""
    fake_client.seed("jobs", [_job("job-1", source_board="80000_hours", last_seen_days_ago=4)])
    fake_client.seed("source_fetch_log", [_fetch_log("80000_hours", complete=True)])

    close_jobs(fake_client)

    [payload] = fake_client.table("jobs").upsert_calls
    assert set(payload[0].keys()) == {"id", "canonical_url", "source_board", "content_hash", "status"}
