"""Hourly batch-drain worker — mops up Anthropic Batches API work left
mid-flight by src/pipeline_v2.py's daily run (its own poll budget exhausted,
batch not 'ended' yet) or by a crashed process. Does NOT scrape, match, or
send digest email — its only job is: if an enrich_jobs (or enrich_jobs_retry)
batch has finished server-side since the daily run gave up on it, retrieve
and apply it now instead of waiting for tomorrow's run.

Deliberately narrow in scope, matching this module's own concurrency
contract (see src.batch_poll's module docstring): it never decides what NEW
work to submit (requests=None on every resume_or_submit call — submitting is
src.enrich_jobs.enrich_jobs()'s job, next time the daily pipeline runs), and
a parse/validation failure encountered while draining goes straight to
'stale' with no retry batch of its own — see
src.enrich_jobs.apply_drained_batch's docstring for why.

Concurrency-safe with a simultaneously-running daily pipeline: both go
through src.batch_poll.resume_or_submit's atomic claim (compare-and-swap on
batch_submissions.status), so at most one of them is ever actually polling/
applying a given batch at a time.

Usage:
    python -m src.drain_batches
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import anthropic

from src.batch_poll import DRAIN_POLL_BUDGET_SECONDS, resume_or_submit
from src.enrich_jobs import _MODULE as _ENRICH_MODULE
from src.enrich_jobs import apply_drained_batch
from src.supabase_client import get_client

log = logging.getLogger(__name__)

# enrich_jobs's round-1 module name, plus its round-2 retry module — both are
# drained the same way (an ended batch from either is applied identically;
# see apply_drained_batch). Kept as a tuple, not hardcoded strings, so a
# rename of enrich_jobs.py's own _MODULE constant can't silently desync this.
_DRAIN_MODULES = (_ENRICH_MODULE, f"{_ENRICH_MODULE}_retry")


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
            summary = apply_drained_batch(client, outcome.results, run_id=run_id, trigger="drain")
            applied[module] = summary
            log.info("drain: applied module %r batch %s -> %s", module, outcome.batch_id, summary)
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
