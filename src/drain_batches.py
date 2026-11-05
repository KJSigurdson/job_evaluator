"""Hourly batch-drain worker — mops up Anthropic Batches API work left
mid-flight by src/pipeline_v2.py's daily run (its own poll budget exhausted,
batch not 'ended' yet) or by a crashed process. Does NOT scrape, match, or
send digest email — its only job is: if an enrich_jobs/enrich_jobs_retry or
tier1_rerank/tier1_rerank_retry batch has finished server-side since the
daily run gave up on it, retrieve and apply it now instead of waiting for
tomorrow's run.

tier1_rerank's drained batches write ONLY to tier1_verdicts (see
src.tier1_rerank.apply_drained_batch) — never matches_v2, never a digest.
That's deliberate, not an oversight: this module's whole contract is "finish
applying work someone else already decided to submit", and Tier 2/digest are
exactly the kind of NEW decision draining must never make on its own.

2026-11-05: rather than waiting for "the NEXT scheduled run" (tomorrow),
after a drain pass this module now ALSO decides whether to trigger a
CONTINUATION run of src/pipeline_v2.py today — `gh workflow run daily.yml
-f mode=continue`, which skips adapters/ingest/close_jobs but still applies
whatever was just retrieved and runs the normal match -> Tier 1 -> Tier 2 ->
digest path for every active user (see pipeline_v2.run()'s skip_fetch param).
See _should_trigger_continuation for the exact conditions (latest daily run
'incomplete', none currently 'running', under the daily continuation cap,
and either something was actually retrieved this pass or an outstanding
Tier 2 batch has already ended server-side — status-only lookup, still never
retrieved/applied here, tier2_enrich stays deliberately absent from
_DRAIN_MODULES below). This decision is isolated in its own try/except
inside drain() — a failure to trigger a continuation is logged and
swallowed, never turned into a failed drain cycle, since it's a best-effort
nudge: the next scheduled run always picks up the same state regardless.

Deliberately narrow in scope, matching this module's own concurrency
contract (see src.batch_poll's module docstring): it never decides what NEW
work to submit (requests=None on every resume_or_submit call — submitting is
src.enrich_jobs.enrich_jobs()'s/src.tier1_rerank.rerank_tier1_batch()'s job,
next time the daily pipeline runs), and a parse/validation failure
encountered while draining goes straight to a terminal/dropped outcome with
no retry batch of its own — see each module's own apply_drained_batch
docstring for why.

Concurrency-safe with a simultaneously-running daily pipeline: both go
through src.batch_poll.resume_or_submit's atomic claim (compare-and-swap on
batch_submissions.status), so at most one of them is ever actually polling/
applying a given batch at a time.

Usage:
    python -m src.drain_batches
"""
from __future__ import annotations

import logging
import subprocess
import uuid
from datetime import datetime, timezone

import anthropic

from src.batch_poll import DRAIN_POLL_BUDGET_SECONDS, resume_or_submit
from src.enrich_jobs import _MODULE as _ENRICH_MODULE
from src.enrich_jobs import apply_drained_batch as _apply_enrich_jobs_batch
from src.supabase_client import get_client
from src.tier1_rerank import _MODULE as _TIER1_MODULE
from src.tier1_rerank import apply_drained_batch as _apply_tier1_batch
from src.tier2_enrich import _MODULE as _TIER2_MODULE

log = logging.getLogger(__name__)


def _apply_enrich_jobs(client, results: dict, *, batch_id: str, run_id: str) -> dict:
    return _apply_enrich_jobs_batch(client, results, batch_id=batch_id, run_id=run_id, trigger="drain")


def _apply_tier1(client, results: dict, *, batch_id: str, run_id: str) -> dict:
    # tier1_rerank.apply_drained_batch takes no run_id/trigger — it writes
    # only to tier1_verdicts, which has no run_id column of its own (unlike
    # run_metrics/jobs — see that table's migration).
    return _apply_tier1_batch(client, results, batch_id=batch_id)


# enrich_jobs's round-1 module name plus its round-2 retry module, same for
# tier1_rerank — kept as a tuple built from each module's own _MODULE
# constant, not hardcoded strings, so a rename there can't silently desync
# this.
_DRAIN_MODULES = (_ENRICH_MODULE, f"{_ENRICH_MODULE}_retry", _TIER1_MODULE, f"{_TIER1_MODULE}_retry")


def _apply_for_module(module: str, client, results: dict, *, batch_id: str, run_id: str) -> dict:
    """Dispatch to the apply function that knows how to handle a batch from
    *module*. References _apply_enrich_jobs/_apply_tier1 by bare name (not
    via a dict built at import time) specifically so tests monkeypatching
    drain_batches._apply_enrich_jobs/_apply_tier1 are actually honored — a
    dict of function OBJECTS captured at module-load time would freeze in
    the original (real) functions regardless of any later monkeypatch.
    """
    if module in (_ENRICH_MODULE, f"{_ENRICH_MODULE}_retry"):
        return _apply_enrich_jobs(client, results, batch_id=batch_id, run_id=run_id)
    if module in (_TIER1_MODULE, f"{_TIER1_MODULE}_retry"):
        return _apply_tier1(client, results, batch_id=batch_id, run_id=run_id)
    raise ValueError(f"no apply function registered for drain module {module!r}")


def _open_drain_run(client, run_id: str) -> None:
    """Same shape as pipeline_v2._open_pipeline_run — duplicated, not
    imported, since it's ~10 lines and this script's only reason to need a
    pipeline_runs row at all is satisfying run_metrics.run_id's FK to
    pipeline_runs.run_id on the (common) path where draining actually
    applies something. Matches this codebase's established convention of
    duplicating small, mechanical plumbing rather than importing a private
    helper across module boundaries (see enrich_jobs.py's
    _merge_tool_use_blocks docstring for the same precedent)."""
    client.table("pipeline_runs").insert({
        "run_id": run_id, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "trigger": "drain",
    }).execute()


def _close_drain_run(client, run_id: str, *, status: str) -> None:
    try:
        client.table("pipeline_runs").update({
            "status": status, "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("run_id", run_id).execute()
    except Exception as exc:
        log.warning(
            "pipeline_runs close (drain run %s, status=%s) failed — telemetry only, ignored: %r",
            run_id, status, exc,
        )


# ---------------------------------------------------------------------------
# Drainer-triggered continuation runs (2026-11-05) — after retrieving at
# least one batch (or confirming a Tier 2 batch has finished server-side,
# status-only), kick off a continuation daily.yml run (mode=continue) so
# matching/Tier 2/digest for whoever was left mid-flight doesn't wait for
# tomorrow's scheduled run. Every condition in _should_trigger_continuation
# must hold; see drain()'s own try/except around this block for why a
# failure here is isolated from (and never turns into) a drain-cycle
# failure — this is a best-effort nudge, not a correctness requirement
# (the next scheduled run always picks up the same state regardless).
# ---------------------------------------------------------------------------

# Excludes THIS module's own trigger='drain' bookkeeping rows
# (_open_drain_run/_close_drain_run, above) — those are not the daily
# pipeline's own run state and must never be mistaken for "a daily run is
# running" or "the latest daily run".
_DAILY_RUN_TRIGGERS = ("scheduled", "manual")
_CONTINUATION_DAILY_CAP = 3
_TIER2_DRAIN_CHECK_MODULES = (_TIER2_MODULE, f"{_TIER2_MODULE}_retry")


def _latest_daily_run(client) -> dict | None:
    rows = (
        client.table("pipeline_runs").select("run_id,status,started_at,is_continuation")
        .in_("trigger", _DAILY_RUN_TRIGGERS).order("started_at", desc=True).limit(1).execute().data
    )
    return rows[0] if rows else None


def _any_daily_run_running(client) -> bool:
    rows = (
        client.table("pipeline_runs").select("run_id")
        .in_("trigger", _DAILY_RUN_TRIGGERS).eq("status", "running").limit(1).execute().data
    )
    return bool(rows)


def _continuation_runs_today(client) -> int:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = (
        client.table("pipeline_runs").select("run_id")
        .eq("is_continuation", True).gte("started_at", today_start).execute().data
    )
    return len(rows)


def _any_tier2_batch_ended(client, anthropic_client) -> bool:
    """Status-only lookup — never retrieves results, never applies
    anything. There is no tier2 apply_drained_batch (deliberately, see
    module docstring); the continuation run itself resumes Tier 2 the same
    way any other run would, via resume_or_submit."""
    rows = (
        client.table("batch_submissions").select("batch_id")
        .in_("module", _TIER2_DRAIN_CHECK_MODULES).in_("status", ["submitted", "claimed"]).execute().data
    )
    for row in rows:
        batch = anthropic_client.messages.batches.retrieve(row["batch_id"])
        if batch.processing_status == "ended":
            return True
    return False


def _should_trigger_continuation(client, anthropic_client, *, drained_any: bool) -> tuple[bool, str]:
    """All of: latest daily run is 'incomplete'; no daily run is currently
    'running'; fewer than _CONTINUATION_DAILY_CAP continuation runs
    started today (UTC); AND (this pass drained >=1 batch OR an
    outstanding Tier 2 batch has already ended server-side). Returns
    (should_trigger, human-readable reason) — the reason is logged either
    way by the caller, never silently decided."""
    latest = _latest_daily_run(client)
    if latest is None or latest["status"] != "incomplete":
        status_repr = repr(latest["status"]) if latest else "missing"
        return False, f"latest daily pipeline_runs row is {status_repr}, not 'incomplete'"

    if _any_daily_run_running(client):
        return False, "a daily pipeline_runs row is currently 'running'"

    today_count = _continuation_runs_today(client)
    if today_count >= _CONTINUATION_DAILY_CAP:
        return False, f"{today_count} continuation run(s) already started today (cap {_CONTINUATION_DAILY_CAP})"

    if drained_any:
        return True, "this drain pass retrieved >=1 enrich/tier1 batch"

    if _any_tier2_batch_ended(client, anthropic_client):
        return True, "an outstanding tier2 batch has processing_status='ended' at Anthropic"

    return False, "nothing retrieved this pass and no outstanding tier2 batch has ended yet"


def _trigger_continuation_run() -> bool:
    """Shells out to `gh workflow run daily.yml -f mode=continue` — the gh
    CLI reads GH_TOKEN/GITHUB_TOKEN from the environment itself (set in
    drain_batches.yml's own step env), so this needs no explicit auth
    plumbing here. Never raises: a failure to trigger is logged and
    swallowed by the caller — see this module's continuation-trigger
    section docstring on why that's safe."""
    result = subprocess.run(
        ["gh", "workflow", "run", "daily.yml", "-f", "mode=continue"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.error("drain: `gh workflow run` exited %d: %s", result.returncode, result.stderr.strip())
        return False
    return True


def drain(*, client=None, anthropic_client=None) -> dict:
    """One drain cycle: a single brief check per module in _DRAIN_MODULES.
    Never raises — always exits cleanly (process exit 0), whether there was
    nothing to drain, something still in flight, or something successfully
    applied. Returns a summary dict for logging/testing.
    """
    client = client if client is not None else get_client()
    anthropic_client = anthropic_client if anthropic_client is not None else anthropic.Anthropic()

    run_id = str(uuid.uuid4())
    _open_drain_run(client, run_id)
    applied: dict[str, dict] = {}
    outcome_status = "complete"

    try:
        for module in _DRAIN_MODULES:
            outcome = resume_or_submit(
                anthropic_client, client, module=module, requests=None,
                budget_seconds=DRAIN_POLL_BUDGET_SECONDS,
            )
            if outcome.status == "idle":
                log.info("drain: nothing outstanding for module %r", module)
                continue
            if outcome.status == "pending":
                log.info(
                    "drain: module %r still in flight (batch %s) — will check again next hour",
                    module, outcome.batch_id,
                )
                continue
            # outcome.status == "ended"
            summary = _apply_for_module(module, client, outcome.results, batch_id=outcome.batch_id, run_id=run_id)
            applied[module] = summary
            log.info("drain: applied module %r batch %s -> %s", module, outcome.batch_id, summary)

        # Isolated in its own try/except, deliberately NOT allowed to affect
        # outcome_status or propagate — this is a best-effort nudge (see this
        # section's module-level docstring), and a failure deciding whether
        # to trigger (or in the trigger itself) must never make a drain
        # cycle that otherwise worked report 'failed'.
        try:
            drained_any = bool(applied)
            should_trigger, reason = _should_trigger_continuation(client, anthropic_client, drained_any=drained_any)
            if should_trigger:
                triggered = _trigger_continuation_run()
                log.info(
                    "drain: continuation run %s (%s)",
                    "triggered" if triggered else "trigger attempt failed", reason,
                )
            else:
                log.info("drain: continuation run skipped (%s)", reason)
        except Exception:
            log.exception("drain: continuation-trigger decision failed — drain cycle itself still succeeded")

        return {"run_id": run_id, "applied": applied}
    except Exception:
        outcome_status = "failed"
        raise
    finally:
        _close_drain_run(client, run_id, status=outcome_status)


if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    result = drain()
    print(json.dumps(result, indent=2, default=str))
