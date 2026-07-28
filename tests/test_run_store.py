"""Tests for run_store.py: run-history parent/child writes and — the core rule —
telemetry-failure isolation (a Supabase error here must never propagate)."""
from __future__ import annotations

import pytest

from src.run_store import complete_run, start_run
from src.schemas import UserRunResult


def _result(user_id: str, **kwargs) -> UserRunResult:
    defaults = dict(
        new_after_dedup=0, gated_out=0, scored=0, inserted=0,
        near_misses=0, parse_failures=0, gate_breakdown={}, notified=False,
    )
    defaults.update(kwargs)
    return UserRunResult(user_id=user_id, **defaults)


class _RaisingClient:
    """A client whose every write raises — to prove telemetry writes are isolated."""

    class _RaisingTable:
        def upsert(self, *a, **k):
            raise RuntimeError("supabase down")

        def update(self, *a, **k):
            raise RuntimeError("supabase down")

    def table(self, _name):
        return self._RaisingTable()


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------

def test_start_run_writes_one_running_row_and_returns_id(fake_client):
    run_id = start_run(fake_client, trigger="schedule")

    assert run_id is not None
    rows = fake_client.table("pipeline_runs").rows
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "running"
    assert rows[0]["trigger"] == "schedule"
    assert "started_at" in rows[0]


def test_start_run_dry_run_writes_nothing(fake_client):
    assert start_run(fake_client, trigger="x", dry_run=True) is None
    assert fake_client.table("pipeline_runs").rows == []


def test_start_run_write_failure_is_isolated():
    # Must not raise; returns None so complete_run later no-ops.
    assert start_run(_RaisingClient(), trigger="schedule") is None


# ---------------------------------------------------------------------------
# complete_run — parent totals + child rows
# ---------------------------------------------------------------------------

def test_complete_run_updates_parent_totals(fake_client):
    run_id = start_run(fake_client, trigger="schedule")
    results = [
        _result("u1", new_after_dedup=10, gated_out=6, scored=4, inserted=2, near_misses=1, parse_failures=1),
        _result("u2", new_after_dedup=3, gated_out=1, scored=2, inserted=0, near_misses=0, parse_failures=0),
    ]

    complete_run(fake_client, run_id, status="complete", results=results, scraped=100, shared_pool=50)

    parent = fake_client.table("pipeline_runs").rows[0]
    assert parent["status"] == "complete"
    assert "completed_at" in parent
    assert parent["scraped"] == 100
    assert parent["shared_pool"] == 50
    assert parent["user_count"] == 2
    assert parent["total_gated_out"] == 7
    assert parent["total_scored"] == 6
    assert parent["total_inserted"] == 2
    assert parent["total_parse_failures"] == 1


def test_complete_run_writes_child_rows_with_correct_field_mapping(fake_client):
    run_id = start_run(fake_client, trigger="workflow_dispatch")
    results = [
        _result("u1", new_after_dedup=10, gated_out=6, scored=4, inserted=2, near_misses=1,
                parse_failures=1, gate_breakdown={"location": 4, "seniority": 2}, notified=True),
        _result("u2", new_after_dedup=3, gated_out=1, scored=2),
    ]

    complete_run(fake_client, run_id, status="complete", results=results, scraped=9, shared_pool=None)

    children = fake_client.table("pipeline_run_users").rows
    assert len(children) == 2
    by_user = {c["user_id"]: c for c in children}

    u1 = by_user["u1"]
    assert u1["run_id"] == run_id
    assert u1["after_seen_skip"] == 10          # <- new_after_dedup
    assert u1["gated_out"] == 6
    assert u1["gate_breakdown"] == {"location": 4, "seniority": 2}
    assert u1["scored"] == 4
    assert u1["parse_failures"] == 1
    assert u1["inserted"] == 2
    assert u1["near_misses"] == 1
    assert u1["notified"] is True

    assert by_user["u2"]["after_seen_skip"] == 3
    assert by_user["u2"]["notified"] is False


def test_complete_run_no_children_when_results_empty(fake_client):
    run_id = start_run(fake_client, trigger=None)
    complete_run(fake_client, run_id, status="complete", results=[], scraped=0, shared_pool=None)

    assert fake_client.table("pipeline_runs").rows[0]["user_count"] == 0
    assert fake_client.table("pipeline_run_users").rows == []


def test_complete_run_noop_when_run_id_none(fake_client):
    complete_run(fake_client, None, status="failed", results=[_result("u1")], scraped=5, shared_pool=5)
    # start_run never ran, so no parent row was opened, and no child rows are written.
    assert fake_client.table("pipeline_runs").rows == []
    assert fake_client.table("pipeline_run_users").rows == []


def test_complete_run_dry_run_writes_nothing(fake_client):
    run_id = start_run(fake_client, trigger="x")  # note: not dry-run, so a row exists
    fake_client.table("pipeline_runs").update_calls.clear()

    complete_run(fake_client, run_id, status="complete", results=[_result("u1")],
                 scraped=1, shared_pool=1, dry_run=True)

    assert fake_client.table("pipeline_runs").update_calls == []
    assert fake_client.table("pipeline_run_users").rows == []


def test_complete_run_write_failure_is_isolated():
    # A real run_id + a client that raises on write → must be swallowed, not raised.
    complete_run(_RaisingClient(), "run-123", status="complete",
                 results=[_result("u1", inserted=1)], scraped=10, shared_pool=5)
