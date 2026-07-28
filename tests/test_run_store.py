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


class _ChildFailingClient:
    """Wraps a fake client but makes ONLY pipeline_run_users writes raise — isolates
    the "child insert fails, parent must still finalize" case from a total outage."""

    class _RaisingTable:
        def upsert(self, *a, **k):
            raise RuntimeError("pipeline_run_users is unavailable")

    def __init__(self, inner):
        self._inner = inner

    def table(self, name):
        if name == "pipeline_run_users":
            return self._RaisingTable()
        return self._inner.table(name)


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
    assert parent["children_written"] is True


def test_complete_run_writes_child_rows_with_correct_field_mapping(fake_client):
    """Uses real UserRunResult objects, exactly as pipeline.py's call site passes
    (results=user_results, the raw per-user objects appended during the run loop) —
    not pre-converted dicts. This is the fixture-vs-reality gap to guard against: the
    test input must match what the real caller actually hands complete_run."""
    run_id = start_run(fake_client, trigger="workflow_dispatch")
    results = [
        _result("u1", new_after_dedup=10, gated_out=6, scored=4, inserted=2, near_misses=1,
                parse_failures=1, gate_breakdown={"location": 4, "seniority": 2}, notified=True),
        _result("u2", new_after_dedup=3, gated_out=1, scored=2),
    ]
    assert all(isinstance(r, UserRunResult) for r in results)  # locks in the real call shape

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


class _ParentFailingClient:
    """Wraps a fake client but makes ONLY pipeline_runs writes raise — isolates the
    "parent update fails, must not stay stuck at running" case."""

    class _RaisingTable:
        def upsert(self, *a, **k):
            raise RuntimeError("pipeline_runs is unavailable")

        def update(self, *a, **k):
            raise RuntimeError("pipeline_runs is unavailable")

    def __init__(self, inner):
        self._inner = inner

    def table(self, name):
        if name == "pipeline_runs":
            return self._RaisingTable()
        return self._inner.table(name)


# ---------------------------------------------------------------------------
# gate_breakdown coercion — must always end up str-keyed/int-valued regardless of
# what shape landed in the Counter, since JSONB has no notion of a non-string key.
# ---------------------------------------------------------------------------

def test_complete_run_coerces_non_str_gate_breakdown_keys(fake_client):
    run_id = start_run(fake_client, trigger="schedule")
    result = _result("u1")
    # UserRunResult's gate_breakdown: dict[str, int] validates at CONSTRUCTION time,
    # but the model has no validate_assignment=True config, so a post-construction
    # attribute assignment (exactly how pipeline.py sets it: result.gate_breakdown =
    # dict(gate_rejection_reasons)) bypasses that validation entirely. Assigning
    # directly here — deliberately adversarial, non-str key and non-int value — is
    # how a malformed value could realistically reach complete_run in production,
    # and is exactly what the coercion in _build_child_row exists to catch.
    result.gate_breakdown = {"location": 3, 7: "2"}
    results = [result]

    complete_run(fake_client, run_id, status="complete", results=results, scraped=1, shared_pool=1)

    children = fake_client.table("pipeline_run_users").rows
    assert len(children) == 1
    gb = children[0]["gate_breakdown"]
    assert gb == {"location": 3, "7": 2}
    assert all(isinstance(k, str) for k in gb)
    assert all(isinstance(v, int) for v in gb.values())


# ---------------------------------------------------------------------------
# children_written / complete_partial — parent row must still finalize (with a
# terminal status) when only the child-row write fails, and must record that fact
# rather than silently looking like a healthy run with an empty pipeline_run_users
# table.
# ---------------------------------------------------------------------------

def test_complete_run_child_failure_downgrades_status_to_complete_partial(fake_client, caplog):
    run_id = start_run(fake_client, trigger="schedule")
    failing_client = _ChildFailingClient(fake_client)

    with caplog.at_level("ERROR"):
        complete_run(failing_client, run_id, status="complete",
                     results=[_result("u1", inserted=1)], scraped=5, shared_pool=5)

    # Parent row still finalizes — a child-write failure must not also swallow the
    # parent write — but status is downgraded so a healthy-looking 'complete' run
    # doesn't mask the fact that its per-user telemetry is missing.
    parent = fake_client.table("pipeline_runs").rows[0]
    assert parent["status"] == "complete_partial"
    assert parent["total_inserted"] == 1
    assert parent["children_written"] is False
    assert fake_client.table("pipeline_run_users").rows == []

    # The failure is attributable: ERROR level, names the run, includes exc repr
    # and a sample payload — not swallowed at debug/warning.
    assert any(
        r.levelname == "ERROR" and "child-row write FAILED" in r.message and run_id in r.message
        for r in caplog.records
    )


def test_complete_run_does_not_downgrade_a_genuinely_failed_run(fake_client):
    """status='failed' (the run itself crashed) must never be overwritten to
    'complete_partial' just because children also failed to write — that would hide
    a real failure behind a misleadingly reassuring label."""
    run_id = start_run(fake_client, trigger="schedule")
    failing_client = _ChildFailingClient(fake_client)

    complete_run(failing_client, run_id, status="failed",
                 results=[_result("u1")], scraped=5, shared_pool=5)

    assert fake_client.table("pipeline_runs").rows[0]["status"] == "failed"


def test_complete_run_children_written_true_when_results_empty(fake_client):
    """No users to write children for isn't a failure — children_written stays True
    and status is not downgraded."""
    run_id = start_run(fake_client, trigger="schedule")
    complete_run(fake_client, run_id, status="complete", results=[], scraped=0, shared_pool=None)

    parent = fake_client.table("pipeline_runs").rows[0]
    assert parent["children_written"] is True
    assert parent["status"] == "complete"


# ---------------------------------------------------------------------------
# Parent write failure — must not leave the row stuck at 'running' forever
# ---------------------------------------------------------------------------

def test_complete_run_parent_failure_triggers_fallback_status_failed(fake_client, caplog):
    run_id = start_run(fake_client, trigger="schedule")
    failing_client = _ParentFailingClient(fake_client)

    with caplog.at_level("ERROR"):
        complete_run(failing_client, run_id, status="complete",
                     results=[_result("u1")], scraped=1, shared_pool=1)

    # Both the primary update AND the fallback go through the wrapped table, so
    # both raise in this test — but the fallback attempt itself must not propagate,
    # and the failure must be logged at ERROR, not silently dropped.
    assert any(r.levelname == "ERROR" and "parent-row update FAILED" in r.message for r in caplog.records)
    assert any(r.levelname == "ERROR" and "fallback status='failed'" in r.message for r in caplog.records)


def test_complete_run_parent_failure_fallback_succeeds_when_table_recovers(fake_client):
    """A more realistic partial-failure case: the first .update() call raises (e.g. a
    payload/serialization issue), but a plain status-only .update() can still get
    through — the row must end up 'failed', not stuck at 'running'."""

    class _OnceFailingTable:
        def __init__(self, real_table):
            self._real = real_table
            self._calls = 0

        def update(self, payload):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("first update payload rejected")
            return self._real.update(payload)

    class _Client:
        def __init__(self, inner):
            self._inner = inner
            self._wrapped = _OnceFailingTable(inner.table("pipeline_runs"))

        def table(self, name):
            if name == "pipeline_runs":
                return self._wrapped
            return self._inner.table(name)

    run_id = start_run(fake_client, trigger="schedule")
    client = _Client(fake_client)

    complete_run(client, run_id, status="complete", results=[_result("u1")], scraped=1, shared_pool=1)

    assert fake_client.table("pipeline_runs").rows[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# Success log ordering — must fire only after BOTH writes have actually succeeded
# ---------------------------------------------------------------------------

def test_complete_run_success_log_fires_only_after_both_writes_succeed(fake_client, caplog):
    run_id = start_run(fake_client, trigger="schedule")

    with caplog.at_level("INFO"):
        complete_run(fake_client, run_id, status="complete",
                     results=[_result("u1", inserted=1)], scraped=1, shared_pool=1)

    assert any("completed run" in r.message for r in caplog.records if r.levelname == "INFO")


def test_complete_run_success_log_absent_when_child_write_fails(fake_client, caplog):
    run_id = start_run(fake_client, trigger="schedule")
    failing_client = _ChildFailingClient(fake_client)

    with caplog.at_level("INFO"):
        complete_run(failing_client, run_id, status="complete",
                     results=[_result("u1", inserted=1)], scraped=1, shared_pool=1)

    assert not any("completed run" in r.message for r in caplog.records if r.levelname == "INFO")


def test_complete_run_success_log_absent_when_parent_write_fails(fake_client, caplog):
    run_id = start_run(fake_client, trigger="schedule")
    failing_client = _ParentFailingClient(fake_client)

    with caplog.at_level("INFO"):
        complete_run(failing_client, run_id, status="complete",
                     results=[_result("u1", inserted=1)], scraped=1, shared_pool=1)

    assert not any("completed run" in r.message for r in caplog.records if r.levelname == "INFO")
