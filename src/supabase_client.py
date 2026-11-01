"""Supabase client construction. Service-role key bypasses RLS — server-side use only."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_PAGE_SIZE = 1000
_WRITE_CHUNK_SIZE = 500

# PostgREST encodes EVERY filter — including .in_() — as part of the request
# URL's query string, for GET *and* for PATCH/UPDATE (the row data goes in
# the body, but the WHERE-equivalent filters never do). An unbounded id list
# therefore makes an unbounded URL, which can fail as a proxy-level 5xx long
# before any Postgres-side limit is reached: confirmed for real when
# src.closure.close_jobs() re-fetched 1,665 UUIDs via a single
# .in_("id", job_ids) — a ~65KB GET URL — and Cloudflare returned a 520, not
# a PostgREST error. _WRITE_CHUNK_SIZE above is a DIFFERENT, unrelated limit
# (POST-body row count, for chunked_upsert/chunked_insert's statement-timeout
# concern) — it does not protect against this failure mode at all, since the
# body isn't the problem here. See fetch_all_in().
_IN_FILTER_CHUNK_SIZE = 100


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


def _chunk_values(values, size: int = _IN_FILTER_CHUNK_SIZE):
    """Yield consecutive chunks of at most *size* from *values* — the shared
    building block for any PostgREST .in_() filter. See _IN_FILTER_CHUNK_SIZE's
    docstring for why this must be small (URL length), not the chunked_upsert/
    chunked_insert row-count limit (POST body size / statement timeout)."""
    values = list(values)
    for i in range(0, len(values), size):
        yield values[i:i + size]


def fetch_all_in(
    table, column: str, values, *, select_cols: str = "*", filter_fn=None,
    chunk_size: int = _IN_FILTER_CHUNK_SIZE,
) -> list[dict]:
    """SELECT *select_cols* FROM *table* WHERE *column* IN *values*, chunking
    *values* into groups of at most *chunk_size* (default _IN_FILTER_CHUNK_SIZE)
    so no single request's .in_() filter can grow an unbounded URL — see
    _IN_FILTER_CHUNK_SIZE's docstring for the real Cloudflare 520 this
    prevents. Each chunk's own result set is paginated via fetch_all (a
    single id-filtered chunk is normally well under fetch_all's 1000-row
    page cap, but this stays correct even if a column isn't unique per id).
    Merges every chunk's rows into one list, in no particular cross-chunk
    order.

    table: a client.table(name) handle (not a pre-built query) — this
    function owns calling .select()/.in_() itself, once per chunk.
    filter_fn: optional callable(query) -> query applied to EACH chunk's
    query builder before .execute(), for an additional filter the original
    (unchunked) call site had — e.g. `lambda q: q.eq("status", "enriched")`.
    Returns [] immediately for an empty/falsy *values*, no request made.
    """
    if not values:
        return []
    rows: list[dict] = []
    for chunk in _chunk_values(values, chunk_size):
        query = table.select(select_cols).in_(column, chunk)
        if filter_fn is not None:
            query = filter_fn(query)
        rows.extend(fetch_all(query))
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

def record_batch_submitted(client, batch_id: str, module: str, *, manifest: dict | None = None) -> None:
    """Call immediately after anthropic_client.messages.batches.create()
    returns — before any polling begins. Without this, a runner that dies
    mid-poll (batches can legitimately take up to 24h) leaves no durable
    trace that an already-paid-for batch exists, and the next run would
    blindly submit duplicate work.

    manifest: optional request-time snapshot (currently only
    src.tier1_rerank.rerank_tier1_batch uses this) of exactly what was sent
    in THIS batch — e.g. {custom_id: {"user_id", "profile_content_hash",
    "jobs": {job_id: job_enriched_at}}}. Persisted here, alongside the
    submission, specifically so a verdict's validity can be judged against
    what the LLM actually saw at request time, not whatever `jobs`/`profiles`
    happen to say by the time results are applied — which can be hours
    later (the external hourly drainer) and after a job's since been
    re-enriched or a profile's since changed. See fetch_batch_manifest.
    """
    client.table("batch_submissions").insert({
        "batch_id": batch_id, "module": module, "status": "submitted", "manifest": manifest,
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


def insert_run_metrics(client, payload: dict) -> None:
    """Write one run_metrics row. NEVER raises — run_metrics is pure
    observability (cost/volume reporting), and a failure writing it must
    never abort a pipeline run that otherwise did real, successful work.

    Confirmed for real: close_jobs() closed 1,665 jobs (4 chunked upserts,
    all 200 OK) and only THEN tried to write its run_metrics row, which
    400'd with PGRST204 ("Could not find the 'closure_summary' column") —
    a migration that existed in the repo but hadn't actually been applied
    to the live schema. That exception propagated straight up and aborted
    the run: enrich/matching never ran that day, even though the closure
    work itself had already fully and correctly committed. Observability
    must never be able to do that.

    On failure, logs the FULL PAYLOAD at ERROR (not just the exception) so
    the metrics this row would have recorded are still recoverable from
    logs even though the row itself is lost — there is no retry, by design:
    a run_metrics row describes a stage that's already finished, so retrying
    the insert later has nothing fresher to report and isn't worth the extra
    code path for a telemetry-only write.
    """
    try:
        client.table("run_metrics").insert(payload).execute()
    except Exception as exc:
        log.error("run_metrics insert failed (row lost, pipeline continues): %r — payload=%s", exc, payload)


def fetch_batch_manifest(client, batch_id: str) -> dict | None:
    """The request-time manifest persisted by record_batch_submitted() for
    *batch_id*, if any — see that function's docstring. None for a batch
    submitted without one (every module except tier1_rerank, or a
    tier1_rerank batch submitted before this existed)."""
    rows = client.table("batch_submissions").select("manifest").eq("batch_id", batch_id).execute().data or []
    return rows[0]["manifest"] if rows else None


def mark_batch_results_processed(client, batch_id: str, custom_ids: list[str], *, chunk_size: int = _IN_FILTER_CHUNK_SIZE) -> None:
    """Mark batch_results rows (batch_id, custom_id) as processed=True. Called by
    each Batches-API module's LIVE path once the corresponding downstream write
    (or, for a module with no further write of its own, parsing) has safely
    committed — mirroring the timing each module's reprocess_staged() recovery
    path already uses when marking its own staged rows. Plain UPDATE, not
    upsert: these rows already exist (persist_batch_results wrote them at
    retrieval time), so there's no NOT-NULL-on-insert-branch concern here —
    only `processed` needs to change.

    Without this, every live-path success leaves its batch_results row
    indistinguishable from one an incomplete run never got to apply — confirmed
    for real against production: every batch_results row inspected had
    processed=False regardless of outcome, including rows from months-old,
    fully successful enrichments, because only reprocess_staged() ever set this
    column.

    chunk_size defaults to _IN_FILTER_CHUNK_SIZE (100), NOT _WRITE_CHUNK_SIZE
    (500) — this UPDATE's .in_("custom_id", chunk) filter is still URL query
    string, same as any SELECT's; the 500-row convention is for POST-body
    row counts (chunked_upsert/chunked_insert), a different limit entirely.
    """
    for chunk in _chunk_values(custom_ids, chunk_size):
        client.table("batch_results").update({"processed": True}).eq("batch_id", batch_id).in_("custom_id", chunk).execute()


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
