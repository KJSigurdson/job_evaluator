"""Run-history telemetry, backed by Supabase `pipeline_runs` (parent, one row per
run) + `pipeline_run_users` (child, one row per user per run). Powers the admin
page's run history.

Mirrors quota_store.py's client/pattern: same Supabase client, upsert-style writes,
INFO logging. The one hard rule here is the core reliability invariant — this is
MONITORING, so a telemetry write failure must NEVER crash or abort a real pipeline
run. Every write is wrapped: on any error it logs a warning and returns, never
raising into pipeline.run(). start_run failing (or dry_run) simply yields run_id=None,
which complete_run treats as "nothing to finalize".
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from src.schemas import UserRunResult

log = logging.getLogger(__name__)


def start_run(client, *, trigger: str | None, dry_run: bool = False) -> str | None:
    """Write one `pipeline_runs` row with status='running' and return its run_id.

    dry_run → log and return None (no write). If the write itself fails, log a warning
    and return None — run history must never break an actual run. Uses upsert on
    run_id (mirrors quota_store's upsert style); run_id is a fresh uuid so this is
    always an insert in practice.
    """
    if dry_run:
        log.info("DRY RUN — would open pipeline_runs row (trigger=%s)", trigger)
        return None

    run_id = str(uuid.uuid4())
    payload = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
    }
    try:
        client.table("pipeline_runs").upsert(payload, on_conflict="run_id").execute()
        log.info("pipeline_runs: opened run %s (trigger=%s)", run_id, trigger)
        return run_id
    except Exception as exc:
        log.warning("pipeline_runs start_run write failed (telemetry only, ignored): %s", exc)
        return None


def complete_run(
    client,
    run_id: str | None,
    *,
    status: str,
    results: list[UserRunResult],
    scraped: int,
    shared_pool: int | None,
    dry_run: bool = False,
) -> None:
    """Finalize the `pipeline_runs` row and write per-user `pipeline_run_users` children.

    No-op (log only) when run_id is None or dry_run — either means start_run never
    opened a row. Updates the parent with terminal status/completed_at, the run-wide
    scraped/shared_pool/user_count, totals summed across *results*, and
    children_written (see below); then bulk-upserts one child row per UserRunResult.

    Child-row writes are attempted and caught in their OWN try/except, separate from
    the parent update: a child-write failure must not also skip the parent's
    finalization or its "completed run" log line — those are two independent Supabase
    calls and a failure in the second should not erase evidence that the first
    succeeded. children_written on the parent row records which case happened, so the
    admin page can distinguish "healthy run with per-user drill-down" from "run
    finished but per-user telemetry failed" instead of silently showing an empty
    child table as if that were normal. Requires a `children_written boolean` column
    on `pipeline_runs` (not yet in any DDL this repo tracks — add it before relying on
    this field). Every write here is wrapped: a failure logs a WARNING (with the
    exception repr, so it's actually greppable in the Actions log) and returns/continues
    — never raises into run().
    """
    if run_id is None or dry_run:
        log.info(
            "pipeline_runs: skipping completion (run_id=%s, dry_run=%s, status=%s)",
            run_id, dry_run, status,
        )
        return

    children_written = True
    if results:
        try:
            child_rows = [
                {
                    "run_id": run_id,
                    "user_id": r.user_id,
                    "after_seen_skip": r.new_after_dedup,
                    "gated_out": r.gated_out,
                    "gate_breakdown": dict(r.gate_breakdown),
                    "scored": r.scored,
                    "parse_failures": r.parse_failures,
                    "inserted": r.inserted,
                    "near_misses": r.near_misses,
                    "notified": r.notified,
                }
                for r in results
            ]
            client.table("pipeline_run_users").upsert(
                child_rows, on_conflict="run_id,user_id"
            ).execute()
        except Exception as exc:
            children_written = False
            log.warning(
                "pipeline_run_users child-row write failed for run %s (telemetry only, ignored): %r",
                run_id, exc,
            )

    try:
        parent = {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "scraped": scraped,
            "shared_pool": shared_pool,
            "user_count": len(results),
            "total_gated_out": sum(r.gated_out for r in results),
            "total_scored": sum(r.scored for r in results),
            "total_inserted": sum(r.inserted for r in results),
            "total_parse_failures": sum(r.parse_failures for r in results),
            "children_written": children_written,
        }
        client.table("pipeline_runs").update(parent).eq("run_id", run_id).execute()
        log.info(
            "pipeline_runs: completed run %s status=%s users=%d inserted=%d children_written=%s",
            run_id, status, len(results), parent["total_inserted"], children_written,
        )
    except Exception as exc:
        log.warning(
            "pipeline_runs parent-row write failed for run %s (telemetry only, ignored): %r",
            run_id, exc,
        )
