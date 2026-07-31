"""Supabase client construction. Service-role key bypasses RLS — server-side use only."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_PAGE_SIZE = 1000
_WRITE_CHUNK_SIZE = 500


def get_client():
    """Build a Supabase client from SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY."""
    from supabase import Client, create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def fetch_all(query, *, page_size: int = _PAGE_SIZE) -> list[dict]:
    """Execute *query* (an already-filtered, not-yet-.execute()'d PostgREST query
    builder) with .range() pagination, collecting every matching row.

    PostgREST/Supabase caps a single response at db.max_rows (1000 by default,
    this project's real setting) — a plain query.execute() on a query matching
    MORE rows than that silently truncates to the first page, no error, no
    warning. Confirmed for real once True North v2's `jobs` table crossed 1000
    rows: an unpaginated select saw 1000 of 1697 rows with no indication
    anything was missing. Any select that isn't already guaranteed to match
    fewer than page_size rows (via a narrow filter, .eq() on a unique column,
    etc.) MUST go through this instead of query.execute() directly — this is
    especially critical for src/ingest.py's own-existing-jobs read, since an
    incomplete read there means real duplicate-insert/duplicate-alias bugs on
    the next run, not just an incomplete report.
    """
    rows: list[dict] = []
    start = 0
    while True:
        page = query.range(start, start + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def chunked_upsert(table, rows: list[dict], *, on_conflict: str, chunk_size: int = _WRITE_CHUNK_SIZE) -> None:
    """Upsert *rows* into *table* (a client.table(name) handle) in chunks of at
    most chunk_size, so one oversized upsert can't hit Postgres's statement
    timeout. Confirmed for real: an unchunked ~1136-row upsert (each row
    carrying a jsonb enrichment blob) failed outright with 'canceling statement
    due to statement timeout' once True North v2's enrichment backlog grew large
    enough — and because the whole statement is one transaction, EVERY row in
    it was lost, not just the rows past some cutoff. Mirrors src/seen_store.py's
    established _CHUNK_SIZE=500 convention (that module's chunking predates this
    shared helper and stays local there, per this codebase's per-module
    duplication precedent for pre-existing v1 code — this helper is for v2's
    newer call sites).
    """
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        table.upsert(chunk, on_conflict=on_conflict).execute()


def chunked_insert(table, rows: list[dict], *, chunk_size: int = _WRITE_CHUNK_SIZE) -> None:
    """Same rationale as chunked_upsert, for insert()."""
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        table.insert(chunk).execute()


# ---------------------------------------------------------------------------
# Batch lifecycle — crash-safety for the Anthropic Batches API modules
# (enrich_jobs.py/tier1_rerank.py/tier2_enrich.py). Confirmed for real: a
# batch's LLM output can be fully paid for and successfully retrieved, then
# lost entirely (along with its cost record) if the process crashes before
# the final DB write. These four functions track a batch through
# submitted -> retrieved, and stage every raw result immediately on
# retrieval, so a crash anywhere after submission is recoverable without
# re-calling the (already-paid-for) API.
# ---------------------------------------------------------------------------

def record_batch_submitted(client, batch_id: str, module: str) -> None:
    """Call immediately after anthropic_client.messages.batches.create()
    returns — before any polling begins. Without this, a runner that dies
    mid-poll (batches can legitimately take up to 24h) leaves no durable
    trace that an already-paid-for batch exists, and the next run would
    blindly submit duplicate work."""
    client.table("batch_submissions").insert({
        "batch_id": batch_id, "module": module, "status": "submitted",
    }).execute()


def record_batch_retrieved(client, batch_id: str) -> None:
    """Call immediately after .results(batch_id) succeeds — marks the batch
    no longer outstanding, so find_outstanding_batch() stops surfacing it."""
    client.table("batch_submissions").update({
        "status": "retrieved",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("batch_id", batch_id).execute()


def find_outstanding_batch(client, module: str) -> str | None:
    """Return the batch_id of an outstanding (submitted but not yet
    retrieved) batch for *module*, if any. Callers check this before
    submitting new work, so a run resuming after a timed-out/killed prior run
    resumes polling the existing in-flight batch instead of duplicating an
    already-paid-for submission. Returns None if there's no outstanding
    batch, or — a genuine anomaly, not a normal case — if more than one
    exists; that's logged and left for manual investigation rather than
    guessed at, matching this codebase's fail-open-but-loud philosophy for
    unexpected diagnostic states."""
    rows = (
        client.table("batch_submissions").select("batch_id")
        .eq("module", module).eq("status", "submitted").execute().data or []
    )
    if not rows:
        return None
    if len(rows) > 1:
        log.warning(
            "find_outstanding_batch: %d outstanding submitted batches for module %r "
            "(expected at most 1) — not auto-resuming any of them: %s",
            len(rows), module, [r["batch_id"] for r in rows],
        )
        return None
    return rows[0]["batch_id"]


def persist_batch_results(client, batch_id: str, results: dict) -> None:
    """Persist every raw batch result for *batch_id* to batch_results,
    immediately on retrieval — before any parsing/validation/DB-write risk.
    results: {custom_id: raw_dict} (already-normalized plain dicts, e.g. via
    result.model_dump(mode="json") on the Anthropic SDK object — see each
    module's _run_batch()). Upserts (not insert-only): idempotent, so a run
    that resumes an outstanding batch via find_outstanding_batch() and
    re-persists the same results doesn't crash on the unique(batch_id,
    custom_id) constraint.
    """
    rows = [
        {"batch_id": batch_id, "custom_id": custom_id, "raw": raw, "processed": False}
        for custom_id, raw in results.items()
    ]
    if rows:
        chunked_upsert(client.table("batch_results"), rows, on_conflict="batch_id,custom_id")
