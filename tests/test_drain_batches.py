"""Tests for src/drain_batches.py: wiring/orchestration only — the actual
poll/claim mechanics are src.batch_poll's own tests (test_batch_poll.py) and
the classify/apply mechanics are each module's own apply_drained_batch
tests (test_enrich_jobs.py / test_tier1_rerank.py); these tests monkeypatch
resume_or_submit/the per-module apply functions at the call site and verify
drain() calls them correctly (right modules, right order, right args) and
never raises."""
from __future__ import annotations

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
            return BatchOutcome(status="ended", batch_id="batch_2", results={"u1:0": {"type": "succeeded"}})
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
    assert results == {"u1:0": {"type": "succeeded"}}
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
