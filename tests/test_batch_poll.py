"""Tests for src/batch_poll.py: the shared non-blocking submit/resume/poll/
retrieve helper. Uses a fake Anthropic Batches client (processing_status can
vary call-to-call, unlike test_enrich_jobs.py's always-'ended' fake) and an
injectable fake clock (monotonic_fn/sleep_fn) so budget-exhaustion and the
10/30/60 backoff schedule are exercised deterministically, without any real
wall-clock delay — this is the sleep-injection pattern the whole test suite
previously lacked (time.sleep/time.monotonic were never mocked anywhere)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.batch_poll import (
    DEFAULT_POLL_BUDGET_SECONDS,
    DRAIN_POLL_BUDGET_SECONDS,
    SUBMISSION_EXPIRY_HOURS,
    BatchOutcome,
    resume_or_submit,
)

# ---------------------------------------------------------------------------
# Fake clock — sleep_fn advances the same counter monotonic_fn reads, so the
# poll loop's elapsed-time math is exercised deterministically.
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, start: float = 0.0):
        self._now = start
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += seconds


# ---------------------------------------------------------------------------
# Fake Anthropic Batches client — processing_status is a caller-supplied
# sequence (one value consumed per .retrieve() call, last value repeats),
# unlike test_enrich_jobs.py's fake which always reports 'ended' immediately.
# ---------------------------------------------------------------------------

class _FakeRequestCounts:
    def __init__(self, **kwargs):
        self._data = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, **kwargs}

    def model_dump(self):
        return dict(self._data)


class _FakeBatch:
    def __init__(self, id, processing_status, request_counts=None):  # noqa: A002
        self.id = id
        self.processing_status = processing_status
        self.request_counts = request_counts or _FakeRequestCounts()


class _FakeResultItem:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _FakeResult:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self, mode="json"):
        return self._data


class _FakeBatchesAPI:
    def __init__(self, *, statuses: list[str], results: list | None = None):
        """statuses: one entry consumed per .retrieve() call; the last entry
        repeats once exhausted. results: what .results() returns once
        'ended' is reached."""
        self._statuses = list(statuses)
        self._results = results or []
        self.created_batches: list[list] = []
        self.retrieve_calls = 0
        self.results_calls = 0
        self._batch_id = None

    def create(self, requests):
        self.created_batches.append(list(requests))
        self._batch_id = f"batch_{len(self.created_batches)}"
        return _FakeBatch(self._batch_id, self._statuses[0] if self._statuses else "in_progress")

    def retrieve(self, batch_id):
        self.retrieve_calls += 1
        idx = min(self.retrieve_calls - 1, len(self._statuses) - 1)
        status = self._statuses[idx]
        return _FakeBatch(batch_id, status)

    def results(self, batch_id):
        self.results_calls += 1
        return self._results


class _FakeMessagesAPI:
    def __init__(self, batches_api):
        self.batches = batches_api


class _FakeAnthropicClient:
    def __init__(self, *, statuses: list[str], results: list | None = None):
        self.messages = _FakeMessagesAPI(_FakeBatchesAPI(statuses=statuses, results=results))


# ---------------------------------------------------------------------------
# batch_submissions row helper — created_at/updated_at are NOT NULL with a
# default in the real schema, always present on a real row.
# ---------------------------------------------------------------------------

def _row(batch_id="batch_1", *, module="enrich_jobs", status="submitted",
         created_at=None, updated_at=None) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "batch_id": batch_id, "module": module, "status": status,
        "created_at": (created_at or now).isoformat(),
        "updated_at": (updated_at or now).isoformat(),
    }


def _requests(n=1) -> list[dict]:
    return [{"custom_id": f"job-{i}", "params": {}} for i in range(n)]


# ---------------------------------------------------------------------------
# Resume applies an already-ended batch, does not call .create()
# ---------------------------------------------------------------------------

def test_resume_applies_already_ended_batch_without_submitting(fake_client):
    fake_client.seed("batch_submissions", [_row("existing_batch")])
    anthropic_client = _FakeAnthropicClient(
        statuses=["ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    outcome = resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
    )

    assert outcome.status == "ended"
    assert outcome.batch_id == "existing_batch"
    assert outcome.results == {"job-1": {"type": "succeeded"}}
    assert anthropic_client.messages.batches.created_batches == []  # never submitted fresh work

    row = fake_client.table("batch_submissions").rows[0]
    assert row["status"] == "retrieved"


def test_resume_persists_results_to_batch_results(fake_client):
    fake_client.seed("batch_submissions", [_row("existing_batch")])
    anthropic_client = _FakeAnthropicClient(
        statuses=["ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    [stored] = fake_client.table("batch_results").rows
    assert stored["batch_id"] == "existing_batch"
    assert stored["custom_id"] == "job-1"


# ---------------------------------------------------------------------------
# Budget exhaustion returns pending, exits cleanly (no exception), leaves
# the row open (resumable)
# ---------------------------------------------------------------------------

def test_budget_exhaustion_returns_pending_never_raises(fake_client):
    clock = _FakeClock()
    anthropic_client = _FakeAnthropicClient(statuses=["in_progress"])  # never ends

    outcome = resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
        budget_seconds=100, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    assert outcome.status == "pending"
    assert outcome.batch_id is not None
    # No exception raised — this IS the point (see module docstring: exhausting
    # the budget is a normal outcome, never a TimeoutError).


def test_budget_exhaustion_leaves_row_resumable(fake_client):
    clock = _FakeClock()
    anthropic_client = _FakeAnthropicClient(statuses=["in_progress"])

    resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
        budget_seconds=100, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    row = fake_client.table("batch_submissions").rows[0]
    assert row["status"] == "submitted"  # released back, not stuck at 'claimed'


def test_budget_exhaustion_does_not_write_batch_results(fake_client):
    clock = _FakeClock()
    anthropic_client = _FakeAnthropicClient(statuses=["in_progress"])

    resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
        budget_seconds=50, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    assert fake_client.table("batch_results").rows == []


def test_drain_budget_is_a_single_check_no_sleep(fake_client):
    """DRAIN_POLL_BUDGET_SECONDS=0 — the drainer never sleeps, just one
    retrieve() and, if not ended, an immediate pending return."""
    clock = _FakeClock()
    fake_client.seed("batch_submissions", [_row("existing_batch")])
    anthropic_client = _FakeAnthropicClient(statuses=["in_progress"])

    outcome = resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=None,
        budget_seconds=DRAIN_POLL_BUDGET_SECONDS, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    assert outcome.status == "pending"
    assert clock.sleep_calls == []
    assert anthropic_client.messages.batches.retrieve_calls == 1


# ---------------------------------------------------------------------------
# Backoff schedule is 10/30/60 — no real wall-clock delay
# ---------------------------------------------------------------------------

def test_backoff_schedule_is_10_30_60_steady(fake_client):
    clock = _FakeClock()
    # 5 in_progress checks then ended — enough to observe the full 10/30/60/60 pattern.
    anthropic_client = _FakeAnthropicClient(
        statuses=["in_progress", "in_progress", "in_progress", "in_progress", "ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    outcome = resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
        budget_seconds=DEFAULT_POLL_BUDGET_SECONDS, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    assert outcome.status == "ended"
    assert clock.sleep_calls == [10, 30, 60, 60]


def test_backoff_never_exceeds_60s_even_after_many_polls(fake_client):
    clock = _FakeClock()
    statuses = ["in_progress"] * 8 + ["ended"]
    anthropic_client = _FakeAnthropicClient(
        statuses=statuses,
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
        budget_seconds=DEFAULT_POLL_BUDGET_SECONDS, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
    )

    assert clock.sleep_calls[:3] == [10, 30, 60]
    assert all(delay == 60 for delay in clock.sleep_calls[3:])


# ---------------------------------------------------------------------------
# An expired submission unblocks and allows resubmission
# ---------------------------------------------------------------------------

def test_expired_submission_marked_expired_and_resubmitted(fake_client):
    old = datetime.now(timezone.utc) - timedelta(hours=SUBMISSION_EXPIRY_HOURS + 1)
    fake_client.seed("batch_submissions", [_row("stuck_batch", created_at=old, updated_at=old)])
    anthropic_client = _FakeAnthropicClient(
        statuses=["ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    outcome = resume_or_submit(
        anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
    )

    # The stuck row is marked expired...
    rows_by_batch_id = {r["batch_id"]: r for r in fake_client.table("batch_submissions").rows}
    assert rows_by_batch_id["stuck_batch"]["status"] == "expired"
    # ...and fresh work WAS submitted instead of being blocked forever.
    assert len(anthropic_client.messages.batches.created_batches) == 1
    assert outcome.status == "ended"


def test_expired_submission_with_no_new_requests_just_expires_idle(fake_client):
    old = datetime.now(timezone.utc) - timedelta(hours=SUBMISSION_EXPIRY_HOURS + 1)
    fake_client.seed("batch_submissions", [_row("stuck_batch", created_at=old, updated_at=old)])
    anthropic_client = _FakeAnthropicClient(statuses=[])

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    assert outcome.status == "idle"
    assert fake_client.table("batch_submissions").rows[0]["status"] == "expired"


def test_non_expired_submission_not_touched(fake_client):
    """A submission well within the 24h window must NOT be marked expired —
    only genuinely stale rows are swept."""
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    fake_client.seed("batch_submissions", [_row("recent_batch", created_at=recent, updated_at=recent)])
    anthropic_client = _FakeAnthropicClient(
        statuses=["ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    assert fake_client.table("batch_submissions").rows[0]["status"] == "retrieved"
    assert anthropic_client.messages.batches.created_batches == []


# ---------------------------------------------------------------------------
# Idle — nothing outstanding, nothing to submit
# ---------------------------------------------------------------------------

def test_idle_when_nothing_outstanding_and_no_requests(fake_client):
    anthropic_client = _FakeAnthropicClient(statuses=[])

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    assert outcome == BatchOutcome(status="idle", batch_id=None)
    assert anthropic_client.messages.batches.created_batches == []


def test_idle_when_requests_is_empty_list(fake_client):
    anthropic_client = _FakeAnthropicClient(statuses=[])

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=[])

    assert outcome.status == "idle"


# ---------------------------------------------------------------------------
# Concurrency guard — claimed-by-another-process is respected, never
# double-polled/double-submitted
# ---------------------------------------------------------------------------

def test_actively_claimed_batch_is_not_touched(fake_client):
    """A 'claimed' row updated recently (not stale) means another process —
    very likely the drainer — is working it right now. Must not poll or
    submit fresh work."""
    fake_client.seed("batch_submissions", [_row("claimed_batch", status="claimed")])
    anthropic_client = _FakeAnthropicClient(statuses=["ended"])

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1))

    assert outcome.status == "pending"
    assert outcome.batch_id == "claimed_batch"
    assert anthropic_client.messages.batches.retrieve_calls == 0
    assert anthropic_client.messages.batches.created_batches == []


def test_stale_claimed_batch_is_reclaimed(fake_client):
    """A 'claimed' row untouched well past the staleness window is presumed
    abandoned by a crashed process and becomes reclaimable."""
    stale_claim = datetime.now(timezone.utc) - timedelta(minutes=30)
    fake_client.seed("batch_submissions", [_row("crashed_batch", status="claimed", updated_at=stale_claim)])
    anthropic_client = _FakeAnthropicClient(
        statuses=["ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    assert outcome.status == "ended"
    assert fake_client.table("batch_submissions").rows[0]["status"] == "retrieved"


# ---------------------------------------------------------------------------
# 'canceling' distinct from 'in_progress' in logs
# ---------------------------------------------------------------------------

def test_canceling_status_logged_distinctly(fake_client, caplog):
    clock = _FakeClock()
    anthropic_client = _FakeAnthropicClient(
        statuses=["canceling", "ended"],
        results=[_FakeResultItem("job-1", _FakeResult({"type": "succeeded"}))],
    )

    with caplog.at_level("INFO"):
        resume_or_submit(
            anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1),
            sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
        )

    assert any("canceling" in r.message.lower() for r in caplog.records)
    assert any("cancellation requested" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# WARNING log includes the copy-paste unblock UPDATE when refusing to submit
# ---------------------------------------------------------------------------

def test_refusing_to_submit_logs_copy_paste_unblock_statement(fake_client, caplog):
    fake_client.seed("batch_submissions", [_row("stuck_batch", status="claimed")])
    anthropic_client = _FakeAnthropicClient(statuses=[])

    with caplog.at_level("WARNING"):
        resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=_requests(1))

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("stuck_batch" in w and "UPDATE batch_submissions" in w and "status = 'submitted'" in w for w in warnings)


# ---------------------------------------------------------------------------
# More than one open row for a module — anomaly, don't auto-act
# ---------------------------------------------------------------------------

def test_multiple_open_rows_for_module_is_not_auto_acted_on(fake_client):
    fake_client.seed("batch_submissions", [_row("batch_a"), _row("batch_b")])
    anthropic_client = _FakeAnthropicClient(statuses=[])

    outcome = resume_or_submit(anthropic_client, fake_client, module="enrich_jobs", requests=None)

    assert outcome.status == "idle"
    assert anthropic_client.messages.batches.retrieve_calls == 0
