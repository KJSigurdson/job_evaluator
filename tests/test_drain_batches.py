"""Tests for src/drain_batches.py: wiring/orchestration only — the actual
poll/claim mechanics are src.batch_poll's own tests (test_batch_poll.py) and
the classify/apply mechanics are each module's own apply_drained_batch
tests (test_enrich_jobs.py / test_tier1_rerank.py); these tests monkeypatch
resume_or_submit/the per-module apply functions at the call site and verify
drain() calls them correctly (right modules, right order, right args) and
never raises."""
from __future__ import annotations

from datetime import datetime, timezone

from src import drain_batches
from src.batch_poll import BatchOutcome

_ALL_MODULES = ["enrich_jobs", "enrich_jobs_retry", "tier1_rerank", "tier1_rerank_retry"]


def test_drain_checks_every_module(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    calls = []

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        calls.append(module)
        assert requests is None  # drain must NEVER decide what new work to submit
        return BatchOutcome(status="idle", batch_id=None)

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    drain_batches.drain(client=client, anthropic_client=object())

    assert calls == _ALL_MODULES


def test_drain_idle_applies_nothing(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    monkeypatch.setattr(drain_batches, "resume_or_submit", lambda *a, **k: BatchOutcome(status="idle", batch_id=None))

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"] == {}


def test_drain_pending_applies_nothing_this_cycle(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    monkeypatch.setattr(drain_batches, "resume_or_submit", lambda *a, **k: BatchOutcome(status="pending", batch_id="x"))
    monkeypatch.setattr(drain_batches, "_apply_enrich_jobs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not apply a pending outcome")))
    monkeypatch.setattr(drain_batches, "_apply_tier1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not apply a pending outcome")))

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"] == {}


def test_drain_ended_applies_enrich_jobs_via_its_own_apply_fn(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        if module == "enrich_jobs":
            return BatchOutcome(status="ended", batch_id="batch_1", results={"job-1": {"type": "succeeded"}})
        return BatchOutcome(status="idle", batch_id=None)

    applied_calls = []
    def _fake_apply(client_, results, *, batch_id, run_id):
        applied_calls.append((results, run_id))
        return {"enriched": 1, "stale": 0, "requeued": 0, "embeddings_skipped": True}

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    monkeypatch.setattr(drain_batches, "_apply_enrich_jobs", _fake_apply)

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"]["enrich_jobs"] == {"enriched": 1, "stale": 0, "requeued": 0, "embeddings_skipped": True}
    [(results, run_id)] = applied_calls
    assert results == {"job-1": {"type": "succeeded"}}
    assert run_id == result["run_id"]


def test_drain_ended_applies_tier1_via_its_own_apply_fn(monkeypatch):
    """tier1_rerank's drained batches must go through tier1_rerank's own
    apply function (writes verdicts ONLY — see module docstring), not
    enrich_jobs's."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        if module == "tier1_rerank":
            return BatchOutcome(status="ended", batch_id="batch_2", results={"u1_0": {"type": "succeeded"}})
        return BatchOutcome(status="idle", batch_id=None)

    applied_calls = []
    def _fake_apply(client_, results, *, batch_id, run_id):
        applied_calls.append((results, run_id))
        return {"scored": 1, "failed": 0}

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    monkeypatch.setattr(drain_batches, "_apply_tier1", _fake_apply)

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"]["tier1_rerank"] == {"scored": 1, "failed": 0}
    [(results, run_id)] = applied_calls
    assert results == {"u1_0": {"type": "succeeded"}}
    assert run_id == result["run_id"]


def test_drain_opens_and_closes_pipeline_runs_row(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    monkeypatch.setattr(drain_batches, "resume_or_submit", lambda *a, **k: BatchOutcome(status="idle", batch_id=None))

    result = drain_batches.drain(client=client, anthropic_client=object())

    [row] = client.table("pipeline_runs").rows
    assert row["run_id"] == result["run_id"]
    assert row["trigger"] == "drain"
    assert row["status"] == "complete"
    assert row["completed_at"] is not None


def test_drain_closes_pipeline_runs_as_failed_on_exception(monkeypatch):
    import pytest
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()

    def _boom(*a, **k):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(drain_batches, "resume_or_submit", _boom)

    with pytest.raises(RuntimeError):
        drain_batches.drain(client=client, anthropic_client=object())

    [row] = client.table("pipeline_runs").rows
    assert row["status"] == "failed"


def test_drain_never_submits_new_work(monkeypatch):
    """requests=None on every call — the structural guarantee that drain()
    can never decide what new work to submit (see module docstring: that
    stays src.enrich_jobs.enrich_jobs()'s/src.tier1_rerank.
    rerank_tier1_batch()'s job)."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    seen_requests = []

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        seen_requests.append(requests)
        return BatchOutcome(status="idle", batch_id=None)

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    drain_batches.drain(client=client, anthropic_client=object())

    assert seen_requests == [None] * len(_ALL_MODULES)


# ---------------------------------------------------------------------------
# Drainer-triggered continuation runs (2026-11-05)
# ---------------------------------------------------------------------------

class _FakeBatch:
    def __init__(self, processing_status):
        self.processing_status = processing_status


class _FakeBatchesAPI:
    def __init__(self, statuses_by_id: dict[str, str]):
        self._statuses_by_id = statuses_by_id
        self.retrieved_ids: list[str] = []

    def retrieve(self, batch_id):
        self.retrieved_ids.append(batch_id)
        return _FakeBatch(self._statuses_by_id.get(batch_id, "in_progress"))


class _FakeMessagesAPI:
    def __init__(self, batches_api):
        self.batches = batches_api


class _FakeAnthropicClient:
    def __init__(self, statuses_by_id: dict[str, str] | None = None):
        self.messages = _FakeMessagesAPI(_FakeBatchesAPI(statuses_by_id or {}))


def _daily_run_row(*, status, started_at, trigger="scheduled", is_continuation=False, run_id="r1"):
    return {"run_id": run_id, "status": status, "trigger": trigger, "started_at": started_at, "is_continuation": is_continuation}


def test_should_trigger_continuation_false_when_no_daily_run_exists(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is False
    assert "missing" in reason


def test_should_trigger_continuation_false_when_latest_is_complete(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="complete", started_at="2026-11-05T06:17:00+00:00")])

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is False
    assert "complete" in reason


def test_should_trigger_continuation_ignores_drains_own_bookkeeping_rows(monkeypatch):
    """drain()'s own trigger='drain' rows (_open_drain_run) must never be
    mistaken for 'the latest daily run' or 'a daily run is running' — a
    drain cycle's own just-opened 'running' row would otherwise poison
    every single decision, since _open_drain_run runs before this check."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [
        _daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00", run_id="daily-1"),
        _daily_run_row(status="running", started_at="2026-11-05T07:15:00+00:00", trigger="drain", run_id="drain-1"),
    ])

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is True, reason


def test_should_trigger_continuation_false_when_a_daily_run_is_running(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [
        _daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00", run_id="daily-1"),
        _daily_run_row(status="running", started_at="2026-11-05T07:00:00+00:00", trigger="manual", run_id="daily-2"),
    ])

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is False
    assert "running" in reason


def test_should_trigger_continuation_false_when_daily_cap_reached(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rows = [_daily_run_row(status="incomplete", started_at=today, run_id="daily-0")]
    for i in range(3):
        rows.append(_daily_run_row(status="complete", started_at=today, is_continuation=True, run_id=f"cont-{i}"))
    client.seed("pipeline_runs", rows)

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is False
    assert "cap" in reason


def test_should_trigger_continuation_under_cap_still_triggers(monkeypatch):
    """Sibling of the cap test above — 2 continuations today (under the
    cap of 3) must still allow triggering, proving the cap boundary
    itself (>=3, not >3) is right."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rows = [_daily_run_row(status="incomplete", started_at=today, run_id="daily-0")]
    for i in range(2):
        rows.append(_daily_run_row(status="complete", started_at=today, is_continuation=True, run_id=f"cont-{i}"))
    client.seed("pipeline_runs", rows)

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is True, reason


def test_should_trigger_continuation_true_when_drained_any(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])

    should, reason = drain_batches._should_trigger_continuation(client, _FakeAnthropicClient(), drained_any=True)

    assert should is True
    assert "retrieved" in reason


def test_should_trigger_continuation_true_when_tier2_batch_ended_status_only(monkeypatch):
    """Confirms this is a STATUS-ONLY lookup — only .retrieve() is called,
    never .results(), and nothing gets applied as a side effect of this
    check."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])
    client.seed("batch_submissions", [
        {"batch_id": "msgbatch_t2", "module": "tier2_enrich", "status": "submitted"},
    ])
    anthropic_client = _FakeAnthropicClient({"msgbatch_t2": "ended"})

    should, reason = drain_batches._should_trigger_continuation(client, anthropic_client, drained_any=False)

    assert should is True
    assert "tier2" in reason
    assert anthropic_client.messages.batches.retrieved_ids == ["msgbatch_t2"]


def test_should_trigger_continuation_false_when_tier2_batch_still_in_progress(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])
    client.seed("batch_submissions", [
        {"batch_id": "msgbatch_t2", "module": "tier2_enrich", "status": "submitted"},
    ])
    anthropic_client = _FakeAnthropicClient({"msgbatch_t2": "in_progress"})

    should, reason = drain_batches._should_trigger_continuation(client, anthropic_client, drained_any=False)

    assert should is False
    assert "nothing" in reason


def test_should_trigger_continuation_checks_tier2_retry_module_too(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])
    client.seed("batch_submissions", [
        {"batch_id": "msgbatch_t2r", "module": "tier2_enrich_retry", "status": "claimed"},
    ])
    anthropic_client = _FakeAnthropicClient({"msgbatch_t2r": "ended"})

    should, reason = drain_batches._should_trigger_continuation(client, anthropic_client, drained_any=False)

    assert should is True


def test_drain_triggers_continuation_run_when_something_applied(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        if module == "enrich_jobs":
            return BatchOutcome(status="ended", batch_id="batch_1", results={"job-1": {"type": "succeeded"}})
        return BatchOutcome(status="idle", batch_id=None)

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    monkeypatch.setattr(drain_batches, "_apply_enrich_jobs", lambda *a, **k: {"enriched": 1})
    triggered_calls = []
    monkeypatch.setattr(drain_batches, "_trigger_continuation_run", lambda: (triggered_calls.append(1), True)[1])

    drain_batches.drain(client=client, anthropic_client=_FakeAnthropicClient())

    assert triggered_calls == [1]


def test_drain_skips_continuation_trigger_when_latest_run_complete(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="complete", started_at="2026-11-05T06:17:00+00:00")])
    monkeypatch.setattr(drain_batches, "resume_or_submit", lambda *a, **k: BatchOutcome(status="idle", batch_id=None))

    def _should_not_trigger():
        raise AssertionError("must not trigger when the latest daily run is already complete")

    monkeypatch.setattr(drain_batches, "_trigger_continuation_run", _should_not_trigger)

    drain_batches.drain(client=client, anthropic_client=_FakeAnthropicClient())
    # no assertion needed beyond "did not raise" — _should_not_trigger would have


def test_drain_continuation_trigger_failure_does_not_fail_drain_cycle(monkeypatch):
    """A failure deciding/triggering continuation must be isolated — the
    drain cycle itself (which already succeeded) must still report
    'complete', not 'failed'."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    client.seed("pipeline_runs", [_daily_run_row(status="incomplete", started_at="2026-11-05T06:17:00+00:00")])
    monkeypatch.setattr(drain_batches, "resume_or_submit", lambda *a, **k: BatchOutcome(status="idle", batch_id=None))

    def _boom(*a, **k):
        raise RuntimeError("gh CLI exploded")

    monkeypatch.setattr(drain_batches, "_should_trigger_continuation", _boom)

    result = drain_batches.drain(client=client, anthropic_client=_FakeAnthropicClient())

    assert result["applied"] == {}
    [drain_row] = [r for r in client.table("pipeline_runs").rows if r["run_id"] == result["run_id"]]
    assert drain_row["status"] == "complete"  # the drain cycle itself, not the trigger decision


def test_trigger_continuation_run_invokes_gh_cli(monkeypatch):
    calls = []

    class _FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def _fake_run(args, **kwargs):
        calls.append(args)
        return _FakeCompletedProcess()

    monkeypatch.setattr(drain_batches.subprocess, "run", _fake_run)

    assert drain_batches._trigger_continuation_run() is True
    [args] = calls
    assert args == ["gh", "workflow", "run", "daily.yml", "-f", "mode=continue"]


def test_trigger_continuation_run_false_on_nonzero_exit(monkeypatch):
    class _FakeCompletedProcess:
        returncode = 1
        stderr = "HTTP 404: Not Found"

    monkeypatch.setattr(drain_batches.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())

    assert drain_batches._trigger_continuation_run() is False
