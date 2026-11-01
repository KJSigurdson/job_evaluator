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
exactly the kind of NEW decision draining must never make on its own. A user
whose Tier 1 verdicts land via the drainer gets picked up by Tier 2 on the
NEXT scheduled run (src.tier2_enrich.fetch_candidates reads tier1_verdicts
directly, so it has no way to tell a drained verdict from a live one, by
design).

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
import uuid
from datetime import datetime, timezone

import anthropic

from src.batch_poll import DRAIN_POLL_BUDGET_SECONDS, resume_or_submit
from src.enrich_jobs import _MODULE as _ENRICH_MODULE
from src.enrich_jobs import apply_drained_batch as _apply_enrich_jobs_batch
from src.supabase_client import get_client
from src.tier1_rerank import _MODULE as _TIER1_MODULE
from src.tier1_rerank import apply_drained_batch as _apply_tier1_batch

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
