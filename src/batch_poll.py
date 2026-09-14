"""Shared Anthropic Batches API submit/resume/poll/retrieve helper.

Extracted from src/enrich_jobs.py's old `_run_batch` (byte-identical logic
was duplicated across enrich_jobs.py/tier1_rerank.py/tier2_enrich.py — same
3600s/10s constants, same shape). Only src/enrich_jobs.py is wired to this
module for now; tier1_rerank.py/tier2_enrich.py keep their own local,
unmodified copies — rewiring them is a mechanical follow-up, not done here.

Non-blocking by design: `resume_or_submit()` NEVER raises TimeoutError.
Exhausting the poll budget is a normal outcome (status="pending"), not a
failure — the Anthropic Batches API only guarantees completion within 24h,
and blocking a CI runner on that is wrong at any timeout value. A caller
gets back a `BatchOutcome` and decides what to do with a pending result
(typically: stop cleanly, resume next run).

Concurrency safety (daily pipeline vs. the hourly drainer, src/drain_batches.py):
before polling or applying anything, this module atomically CLAIMS the
batch_submissions row via a compare-and-swap UPDATE (`status='submitted' ->
'claimed'`, `WHERE status='submitted'` — PostgREST/supabase-py's `.update()`
returns the modified rows, so an empty response means the WHERE clause
matched nothing and we lost the race). Whoever's UPDATE actually matches a
row owns that batch until it releases the claim (back to 'submitted' if
still pending, or to 'retrieved' once applied) or crashes — a 'claimed' row
left untouched for longer than `_CLAIM_STALE_MINUTES` is presumed abandoned
by a dead process and becomes reclaimable again. This is the "conditional
status transition" concurrency guard (chosen over a Postgres advisory lock —
poor fit here, since PostgREST opens a new connection per HTTP call and a
session-scoped lock wouldn't span this module's multi-request polling loop —
and over a coarser "skip if the daily pipeline is running" check, which was
judged less precise).

A submission untouched (still 'submitted' or 'claimed') for more than
`SUBMISSION_EXPIRY_HOURS` (24h, the Batches API's own completion guarantee)
is marked 'expired' and freed up for resubmission — otherwise a batch that
genuinely never finishes server-side would block this module's caller from
ever making progress on that job again.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.supabase_client import persist_batch_results, record_batch_submitted

log = logging.getLogger(__name__)

# The daily pipeline's default budget — comfortably covers this project's
# observed steady-state batch completion times (historically 2-15 minutes for
# enrich_jobs's typical small batches; see production batch_submissions
# history) while staying well short of a CI runner's timeout.
DEFAULT_POLL_BUDGET_SECONDS = 600

# The drainer's budget: a single retrieve() check, no sleeping at all — it
# runs hourly, so there is no reason for it to block waiting for a batch that
# isn't ready yet; the next hourly tick is the next check.
DRAIN_POLL_BUDGET_SECONDS = 0

_BACKOFF_SCHEDULE = (10, 30, 60)  # seconds; steady at the last value once exhausted
SUBMISSION_EXPIRY_HOURS = 24
_CLAIM_STALE_MINUTES = 15  # comfortably longer than DEFAULT_POLL_BUDGET_SECONDS


def _backoff_delay(poll_count: int) -> int:
    idx = min(poll_count, len(_BACKOFF_SCHEDULE) - 1)
    return _BACKOFF_SCHEDULE[idx]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BatchOutcome:
    """status: 'idle' (nothing outstanding, nothing to submit — the
    drainer's common no-op tick), 'ended' (results ready — see `results`),
    or 'pending' (still in flight, budget exhausted, or being handled by
    another process right now — try again later, nothing was lost)."""
    status: str
    batch_id: str | None
    results: dict | None = None
    elapsed_seconds: float = 0.0
    request_counts: dict | None = None


# ---------------------------------------------------------------------------
# batch_submissions row lookup / claim / release — the concurrency guard
# ---------------------------------------------------------------------------

def _find_candidate(client, module: str) -> dict | None:
    """The one open (submitted or claimed) batch_submissions row for
    *module*, if exactly one exists. Mirrors src.supabase_client.
    find_outstanding_batch's existing "more than one is an anomaly, don't
    auto-act on any of them" philosophy, widened to cover 'claimed' too."""
    rows = (
        client.table("batch_submissions").select("*")
        .eq("module", module).in_("status", ["submitted", "claimed"]).execute().data or []
    )
    if not rows:
        return None
    if len(rows) > 1:
        log.warning(
            "batch_poll: %d open batch_submissions rows for module %r (expected at most 1) — "
            "not auto-acting on any of them: %s",
            len(rows), module, [r["batch_id"] for r in rows],
        )
        return None
    return rows[0]


def _age(row: dict, key: str) -> timedelta:
    return datetime.now(timezone.utc) - datetime.fromisoformat(row[key])


def _unblock_hint(batch_id: str) -> str:
    return (
        f"UPDATE batch_submissions SET status = 'submitted', updated_at = now() "
        f"WHERE batch_id = '{batch_id}';"
    )


def _try_expire(client, row: dict) -> bool:
    """Compare-and-swap *row*'s current status -> 'expired'. Returns True iff
    THIS call won (i.e., actually performed the transition)."""
    resp = (
        client.table("batch_submissions")
        .update({"status": "expired", "updated_at": _now_iso()})
        .eq("batch_id", row["batch_id"]).eq("status", row["status"])
        .execute()
    )
    return bool(resp.data)


def _try_claim(client, batch_id: str, from_status: str) -> bool:
    """Compare-and-swap *from_status* -> 'claimed'. Returns True iff THIS
    call's UPDATE actually matched (and modified) a row — PostgREST/
    supabase-py's .update() returns the modified rows by default (Prefer:
    return=representation), so an empty response.data means the WHERE
    clause matched nothing: the row moved to a different status between our
    SELECT and this UPDATE (another process claimed/retrieved/expired it
    first). Either way, we lost the race and must not proceed as if we own
    this batch."""
    resp = (
        client.table("batch_submissions")
        .update({"status": "claimed", "updated_at": _now_iso()})
        .eq("batch_id", batch_id).eq("status", from_status)
        .execute()
    )
    return bool(resp.data)


def _release_claim(client, batch_id: str, *, new_status: str) -> None:
    """Move a 'claimed' row to its next real status: 'submitted' (still
    pending — budget exhausted, resumable later) or 'retrieved' (results
    applied). Not itself compare-and-swapped — by the time this is called,
    THIS process is the confirmed claim holder (see _try_claim), so an
    unconditional update is correct and simpler."""
    client.table("batch_submissions").update({
        "status": new_status, "updated_at": _now_iso(),
    }).eq("batch_id", batch_id).execute()


# ---------------------------------------------------------------------------
# Bounded poll of a batch THIS process already holds the claim on
# ---------------------------------------------------------------------------

def _poll_claimed_batch(
    anthropic_client, client, batch_id: str, *, budget_seconds: int, sleep_fn, monotonic_fn,
) -> BatchOutcome:
    start = monotonic_fn()
    deadline = start + budget_seconds
    poll_count = 0
    last_counts_obj = None

    while True:
        batch = anthropic_client.messages.batches.retrieve(batch_id)
        last_counts_obj = getattr(batch, "request_counts", None)

        if batch.processing_status == "ended":
            elapsed = monotonic_fn() - start
            results = {}
            for item in anthropic_client.messages.batches.results(batch_id):
                results[item.custom_id] = item.result.model_dump(mode="json")
            persist_batch_results(client, batch_id, results)
            _release_claim(client, batch_id, new_status="retrieved")
            log.info(
                "batch_poll: batch %s ended after %.0fs — %d result(s) retrieved and persisted",
                batch_id, elapsed, len(results),
            )
            return BatchOutcome(
                status="ended", batch_id=batch_id, results=results, elapsed_seconds=elapsed,
                request_counts=last_counts_obj.model_dump() if last_counts_obj else None,
            )

        now = monotonic_fn()
        elapsed = now - start
        counts_dict = last_counts_obj.model_dump() if last_counts_obj else None
        if now >= deadline:
            _release_claim(client, batch_id, new_status="submitted")
            log.info(
                "batch_poll: batch %s still %s after %.0fs (budget %ds exhausted) — "
                "counts=%s — resuming next run.",
                batch_id, batch.processing_status, elapsed, budget_seconds, counts_dict,
            )
            return BatchOutcome(status="pending", batch_id=batch_id, elapsed_seconds=elapsed, request_counts=counts_dict)

        delay = _backoff_delay(poll_count)
        if batch.processing_status == "canceling":
            log.info(
                "batch_poll: batch %s is canceling (cancellation requested externally) — "
                "elapsed %.0fs, checking again in %ds", batch_id, elapsed, delay,
            )
        else:
            log.info(
                "batch_poll: batch %s still %s — elapsed %.0fs, checking again in %ds",
                batch_id, batch.processing_status, elapsed, delay,
            )
        sleep_fn(delay)
        poll_count += 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resume_or_submit(
    anthropic_client, client, *, module: str, requests: list[dict] | None,
    budget_seconds: int = DEFAULT_POLL_BUDGET_SECONDS,
    sleep_fn=time.sleep, monotonic_fn=time.monotonic,
) -> BatchOutcome:
    """Submit *requests* as a new batch for *module* — or, if an outstanding
    submission already exists for *module*, resume/claim/poll THAT one
    instead (never submit while one is open, whether or not *requests* is
    given: submitting a duplicate means paying twice for the same work).

    requests=None or [] means "don't submit new work even if nothing's
    outstanding" — used by src/drain_batches.py, which only ever drains
    what's already in flight and must never decide what new work to submit
    (that stays src/enrich_jobs.py's job).

    NEVER raises on a slow batch — exhausting *budget_seconds* is a normal
    "pending" outcome. sleep_fn/monotonic_fn are injectable (default
    time.sleep/time.monotonic) so tests can exercise the backoff schedule
    and budget exhaustion without real wall-clock delay.
    """
    candidate = _find_candidate(client, module)

    if candidate is not None and _age(candidate, "created_at") > timedelta(hours=SUBMISSION_EXPIRY_HOURS):
        if _try_expire(client, candidate):
            log.warning(
                "batch_poll: batch %s (module=%r) exceeded the %dh completion guarantee "
                "without finishing — marked expired, eligible for resubmission.",
                candidate["batch_id"], module, SUBMISSION_EXPIRY_HOURS,
            )
        # Whether we won that race or another process did, it's no longer
        # 'submitted'/'claimed' — not ours to resume.
        candidate = None

    if candidate is not None and candidate["status"] == "claimed" and _age(candidate, "updated_at") <= timedelta(minutes=_CLAIM_STALE_MINUTES):
        # Actively held by another process right now (very likely the
        # drainer, or a concurrent daily run) — must not submit fresh work
        # (would duplicate already-paid-for requests) and must not poll it
        # ourselves (would race the actual claim holder's writes).
        log.warning(
            "batch_poll: batch %s (module=%r) is currently claimed by another process "
            "(claimed %s ago) — not submitting new work. If this is stuck (the claimant "
            "crashed), unblock with:\n  %s",
            candidate["batch_id"], module, _age(candidate, "updated_at"), _unblock_hint(candidate["batch_id"]),
        )
        return BatchOutcome(status="pending", batch_id=candidate["batch_id"])

    if candidate is not None:
        # 'submitted', or a stale/crashed 'claimed' row eligible for reclaim.
        batch_id = candidate["batch_id"]
        if not _try_claim(client, batch_id, candidate["status"]):
            log.warning(
                "batch_poll: lost the claim race on batch %s (module=%r) to another "
                "process — not submitting new work. If this is stuck, unblock with:\n  %s",
                batch_id, module, _unblock_hint(batch_id),
            )
            return BatchOutcome(status="pending", batch_id=batch_id)
        log.warning(
            "batch_poll: resuming outstanding batch %s for module %r instead of "
            "submitting new work (a prior run likely didn't finish polling it). If this "
            "needs a manual reset:\n  %s",
            batch_id, module, _unblock_hint(batch_id),
        )
    elif requests:
        batch = anthropic_client.messages.batches.create(requests=requests)
        batch_id = batch.id
        record_batch_submitted(client, batch_id, module)
        if not _try_claim(client, batch_id, "submitted"):
            # Vanishingly unlikely (nobody else can know this batch_id until
            # the INSERT above commits) but handled uniformly rather than
            # assumed impossible — see this module's docstring on the CAS
            # guard applying evenly to fresh submissions too.
            log.warning(
                "batch_poll: lost the claim race on batch %s (module=%r) immediately "
                "after submitting it — another process claimed it first.", batch_id, module,
            )
            return BatchOutcome(status="pending", batch_id=batch_id)
        log.info("batch_poll: submitted batch %s with %d request(s)", batch_id, len(requests))
    else:
        return BatchOutcome(status="idle", batch_id=None)

    return _poll_claimed_batch(
        anthropic_client, client, batch_id,
        budget_seconds=budget_seconds, sleep_fn=sleep_fn, monotonic_fn=monotonic_fn,
    )
