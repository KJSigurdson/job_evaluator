"""Tests for src/drain_batches.py: wiring/orchestration only — the actual
poll/claim mechanics are src.batch_poll's own tests (test_batch_poll.py) and
the classify/apply mechanics are src.enrich_jobs.apply_drained_batch's
(test_enrich_jobs.py); these tests monkeypatch resume_or_submit/
apply_drained_batch at the call site and verify drain() calls them correctly
(right modules, right order, right args) and never raises."""
from __future__ import annotations

from src import drain_batches
from src.batch_poll import BatchOutcome


def test_drain_checks_both_enrich_modules(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    calls = []

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        calls.append(module)
        assert requests is None  # drain must NEVER decide what new work to submit
        return BatchOutcome(status="idle", batch_id=None)

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    drain_batches.drain(client=client, anthropic_client=object())

    assert calls == ["enrich_jobs", "enrich_jobs_retry"]


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
    monkeypatch.setattr(drain_batches, "apply_drained_batch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not apply a pending outcome")))

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"] == {}


def test_drain_ended_applies_via_apply_drained_batch(monkeypatch):
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        if module == "enrich_jobs":
            return BatchOutcome(status="ended", batch_id="batch_1", results={"job-1": {"type": "succeeded"}})
        return BatchOutcome(status="idle", batch_id=None)

    applied_calls = []
    def _fake_apply(client_, results, *, run_id, trigger):
        applied_calls.append((results, run_id, trigger))
        return {"enriched": 1, "stale": 0, "requeued": 0, "embeddings_skipped": True}

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    monkeypatch.setattr(drain_batches, "apply_drained_batch", _fake_apply)

    result = drain_batches.drain(client=client, anthropic_client=object())

    assert result["applied"]["enrich_jobs"] == {"enriched": 1, "stale": 0, "requeued": 0, "embeddings_skipped": True}
    [(results, run_id, trigger)] = applied_calls
    assert results == {"job-1": {"type": "succeeded"}}
    assert trigger == "drain"
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
    can never decide what new enrichment work to submit (see module
    docstring: that stays src.enrich_jobs.enrich_jobs()'s job)."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    seen_requests = []

    def _fake_resume(anthropic_client, client_, *, module, requests, budget_seconds):
        seen_requests.append(requests)
        return BatchOutcome(status="idle", batch_id=None)

    monkeypatch.setattr(drain_batches, "resume_or_submit", _fake_resume)
    drain_batches.drain(client=client, anthropic_client=object())

    assert seen_requests == [None, None]
