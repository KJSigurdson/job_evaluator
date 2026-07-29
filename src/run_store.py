"""Run-history telemetry, backed by Supabase `pipeline_runs` (parent, one row per
run) + `pipeline_run_users` (child, one row per user per run). Powers the admin
page's run history.

Mirrors quota_store.py's client/pattern: same Supabase client, upsert-style writes,
INFO logging. The one hard rule here is the core reliability invariant — this is
MONITORING, so a telemetry write failure must NEVER crash or abort a real pipeline
run. Every write is wrapped: on any error it logs a warning and returns, never
raising into pipeline.run(). start_run failing (or dry_run) simply yields run_id=None,
which complete_run treats as "nothing to finalize".

record_token_usage() writes tier1_*/tier2_* token totals onto the pipeline_runs
parent row. Those columns are NOT in any DDL this repo tracks yet — run the
following in Supabase before relying on this data (nullable, so old rows are simply
NULL there, and the write is defensively isolated from complete_run's already-
verified parent write, so a missing column here can't take that down too — see
record_token_usage's docstring):

    ALTER TABLE pipeline_runs
      ADD COLUMN IF NOT EXISTS tier1_calls integer,
      ADD COLUMN IF NOT EXISTS tier1_input_tokens integer,
      ADD COLUMN IF NOT EXISTS tier1_output_tokens integer,
      ADD COLUMN IF NOT EXISTS tier1_cache_read_tokens integer,
      ADD COLUMN IF NOT EXISTS tier1_cache_creation_tokens integer,
      ADD COLUMN IF NOT EXISTS tier2_calls integer,
      ADD COLUMN IF NOT EXISTS tier2_input_tokens integer,
      ADD COLUMN IF NOT EXISTS tier2_output_tokens integer,
      ADD COLUMN IF NOT EXISTS tier2_cache_read_tokens integer,
      ADD COLUMN IF NOT EXISTS tier2_cache_creation_tokens integer;
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from src.schemas import TokenUsage, UserRunResult

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


def _build_child_row(run_id: str, r: UserRunResult) -> dict:
    """One pipeline_run_users row for *r*. gate_breakdown is coerced to a plain
    str-keyed, int-valued dict before it goes anywhere near JSON — defensive even
    though today's only producer (pipeline.py's gate loop, via gate.py's
    rejection_reason) can only ever put "location"/"seniority"/"hard_constraints"
    string keys in there; this stops that guarantee from becoming a silent landmine
    if that ever changes."""
    gate_breakdown = {str(k): int(v) for k, v in (r.gate_breakdown or {}).items()}
    return {
        "run_id": run_id,
        "user_id": r.user_id,
        "after_seen_skip": r.new_after_dedup,
        "gated_out": r.gated_out,
        "gate_breakdown": gate_breakdown,
        "scored": r.scored,
        "parse_failures": r.parse_failures,
        "inserted": r.inserted,
        "near_misses": r.near_misses,
        "notified": r.notified,
    }


def _write_children(client, run_id: str, results: list[UserRunResult]) -> bool:
    """Bulk-upsert one pipeline_run_users row per result. Returns True on success
    (including the trivial case of zero results — nothing to write is not a
    failure). Row construction happens INSIDE this try, not just the network call:
    a bad field on a UserRunResult would throw during _build_child_row, not during
    .execute(), and that must be caught and attributed here too.

    On failure: logs at ERROR (not WARNING — a swallowed telemetry failure that
    isn't at least ERROR-level is functionally invisible) with the exception repr
    AND, best-effort, the first row's payload — rebuilt in isolation so a
    genuinely-malformed results[0] doesn't also hide the sample. Never raises.
    """
    if not results:
        return True

    try:
        child_rows = [_build_child_row(run_id, r) for r in results]
        client.table("pipeline_run_users").upsert(child_rows, on_conflict="run_id,user_id").execute()
        return True
    except Exception as exc:
        try:
            first_payload = _build_child_row(run_id, results[0])
        except Exception:
            first_payload = None  # even the isolated rebuild failed; log without a sample
        log.error(
            "pipeline_run_users child-row write FAILED for run %s (telemetry only, ignored): "
            "%r  first_payload=%r",
            run_id, exc, first_payload,
        )
        return False


def _write_parent(
    client, run_id: str, *, status: str, results: list[UserRunResult],
    scraped: int, shared_pool: int | None, children_written: bool,
) -> bool:
    """Update the pipeline_runs parent row. Returns True on success.

    On failure: logs at ERROR with the exception repr, then makes a best-effort
    second attempt to at least flip status to 'failed' — so this run_id doesn't sit
    stuck at 'running' with a null completed_at forever, which is a worse outcome
    than a correctly-labeled failure. That fallback write is itself wrapped and
    logged-not-raised; if it also fails there's nothing further telemetry can do.
    """
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
        return True
    except Exception as exc:
        log.error(
            "pipeline_runs parent-row update FAILED for run %s — row may be stuck at "
            "its previous status without this fallback (telemetry only, ignored): %r",
            run_id, exc,
        )
        try:
            client.table("pipeline_runs").update({"status": "failed"}).eq("run_id", run_id).execute()
        except Exception as exc2:
            log.error(
                "pipeline_runs fallback status='failed' write ALSO failed for run %s: %r",
                run_id, exc2,
            )
        return False


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
    opened a row.

    Order: children are written first (_write_children), then the parent
    (_write_parent) — folding children_written, and a status downgrade to
    'complete_partial' when the run itself succeeded but its child rows didn't, into
    the SAME parent update rather than a separate follow-up write. A run that
    genuinely failed (status='failed' from the caller) is never overwritten to
    'complete_partial' just because children also failed — that would hide a real
    failure behind a "partial success" label. Requires a 'complete_partial' status
    value to be a valid string in `pipeline_runs.status` (plain text column already,
    no enum/DDL change needed) and a `children_written boolean` column (not yet in
    any DDL this repo tracks — add before an admin panel reads it).

    The "run complete" INFO log only fires once BOTH writes have actually succeeded
    — never before, and never if either failed silently, since a failure already got
    its own ERROR log above. Every write in this function is isolated: nothing here
    ever raises into pipeline.run().
    """
    if run_id is None or dry_run:
        log.info(
            "pipeline_runs: skipping completion (run_id=%s, dry_run=%s, status=%s)",
            run_id, dry_run, status,
        )
        return

    children_ok = _write_children(client, run_id, results)

    effective_status = status
    if not children_ok and status == "complete":
        effective_status = "complete_partial"

    parent_ok = _write_parent(
        client, run_id, status=effective_status, results=results,
        scraped=scraped, shared_pool=shared_pool, children_written=children_ok,
    )

    if parent_ok and children_ok:
        log.info(
            "pipeline_runs: completed run %s status=%s users=%d inserted=%d",
            run_id, effective_status, len(results), sum(r.inserted for r in results),
        )


def record_token_usage(
    client, run_id: str | None, *, tier1_usage: TokenUsage, tier2_usage: TokenUsage,
    dry_run: bool = False,
) -> None:
    """Best-effort write of aggregated token-usage totals onto the pipeline_runs
    parent row. Deliberately a SEPARATE .update() call from complete_run/_write_parent
    rather than folded into that payload: tier1_*/tier2_* are new columns that may
    not exist in the live schema yet (see this module's docstring for the ALTER
    TABLE statements), and a missing-column failure here must not be able to take
    down the already-working status/totals/children_written write the way bundling
    it into one shared payload would.

    No-op (log only) when run_id is None or dry_run — same guard as complete_run.
    Never raises into pipeline.run(); a failure here is logged and swallowed.
    """
    if run_id is None or dry_run:
        log.info(
            "pipeline_runs: skipping token-usage write (run_id=%s, dry_run=%s)",
            run_id, dry_run,
        )
        return

    payload = {
        "tier1_calls": tier1_usage.calls,
        "tier1_input_tokens": tier1_usage.input_tokens,
        "tier1_output_tokens": tier1_usage.output_tokens,
        "tier1_cache_read_tokens": tier1_usage.cache_read_tokens,
        "tier1_cache_creation_tokens": tier1_usage.cache_creation_tokens,
        "tier2_calls": tier2_usage.calls,
        "tier2_input_tokens": tier2_usage.input_tokens,
        "tier2_output_tokens": tier2_usage.output_tokens,
        "tier2_cache_read_tokens": tier2_usage.cache_read_tokens,
        "tier2_cache_creation_tokens": tier2_usage.cache_creation_tokens,
    }
    try:
        client.table("pipeline_runs").update(payload).eq("run_id", run_id).execute()
        log.info(
            "pipeline_runs: recorded token usage for run %s (tier1 calls=%d, tier2 calls=%d)",
            run_id, tier1_usage.calls, tier2_usage.calls,
        )
    except Exception as exc:
        log.warning(
            "pipeline_runs token-usage write failed for run %s — requires tier1_*/tier2_* "
            "columns on pipeline_runs (see this module's docstring for ALTER TABLE); "
            "telemetry only, ignored: %r",
            run_id, exc,
        )
