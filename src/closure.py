"""True North v2 job closure: marks jobs.status='closed' for postings no
longer seen from a source whose fetch THIS RUN was confirmed complete, or —
a safety ceiling for a broken/misconfigured adapter that never reports
complete=True for real — any open job not seen in _SAFETY_CEILING_DAYS
regardless of source. 'closed' is a value in jobs.status's CHECK constraint
(see supabase/migrations/20261101161715_phase1_job_lifecycle.sql); no v1
table touched.

Wired into src/pipeline_v2.py's run(), right after ingest — see that
module's call site for how *complete_sources* is derived (from the exact
fetch_log_rows this run just wrote, not a historical time-window query:
rule (a) means "this run's fetch for that source was complete", not
"sometime recently").
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.supabase_client import chunked_upsert, fetch_all

log = logging.getLogger(__name__)

# A job closes under rule (a) once it hasn't been seen in this long AND its
# source's fetch THIS RUN was complete=true — 3 days, not 1, so a single
# missed/incomplete run elsewhere doesn't close jobs that are still
# genuinely live (cron is daily; this is slack for a skipped day).
_RECENTLY_COMPLETE_STALE_DAYS = 3

# Rule (b): regardless of source completeness, a job not seen in this long is
# closed anyway — the backstop for a source that silently never reports
# complete=True (a broken adapter, or one disabled outright — see
# src/pipeline_v2.py's IAP note. A disabled adapter's jobs WILL eventually all
# close under this rule since nothing ever refreshes their last_seen again;
# that's a known, accepted consequence of "regardless of source", not a bug).
_SAFETY_CEILING_DAYS = 30

_OPEN_STATUSES = ("pending_enrich", "enriched")
_JOB_SELECT_COLUMNS = "id,canonical_url,source_board,content_hash,last_seen"


def plan_closures(client, *, complete_sources: frozenset[str] = frozenset()) -> dict:
    """Compute, WITHOUT writing anything, which open jobs (status in
    pending_enrich/enriched) would be closed right now and under which rule.

    complete_sources: source_board names whose fetch THIS RUN was
    complete=true (see src/pipeline_v2.py's call site) — rule (a) only
    applies to a source in this set. Defaults to empty (no source gets rule
    (a) treatment), used by ad-hoc/review callers that aren't inside an
    actual pipeline run and have no "this run" to speak of; everything still
    falls through to rule (b)'s safety ceiling in that case.

    Returns {"by_rule": {"complete_source_stale": [job_id, ...],
    "safety_ceiling": [job_id, ...]}, "by_source": {source_board: count},
    "total": int, "rows_by_id": {job_id: row}}. job_ids are included (not
    just counts) so a caller — close_jobs() below, or a one-off review
    script — can act on or inspect the exact set, not just a number.
    rows_by_id carries every qualifying row's full _JOB_SELECT_COLUMNS data
    (id/canonical_url/source_board/content_hash/last_seen) specifically so
    close_jobs() can build its update payload from THESE rows instead of
    re-fetching the same job_ids a second time — confirmed for real: a
    second fetch_all(.in_("id", job_ids)) with 1,665 ids made a ~65KB GET
    URL that Cloudflare answered with a 520, not a PostgREST error.

    A job is counted under at most one rule: complete_source_stale takes
    priority over safety_ceiling for a job that qualifies under both (rule
    (a) already fully explains it — rule (b) is the backstop for jobs rule
    (a) DOESN'T catch, not a second vote on ones it does).
    """
    now = datetime.now(timezone.utc)
    complete_cutoff = now - timedelta(days=_RECENTLY_COMPLETE_STALE_DAYS)
    ceiling_cutoff = now - timedelta(days=_SAFETY_CEILING_DAYS)

    rows = fetch_all(client.table("jobs").select(_JOB_SELECT_COLUMNS).in_("status", list(_OPEN_STATUSES)))

    by_rule: dict[str, list[str]] = {"complete_source_stale": [], "safety_ceiling": []}
    by_source: dict[str, int] = {}
    rows_by_id: dict[str, dict] = {}

    for row in rows:
        last_seen = datetime.fromisoformat(row["last_seen"])
        if row["source_board"] in complete_sources and last_seen < complete_cutoff:
            by_rule["complete_source_stale"].append(row["id"])
            by_source[row["source_board"]] = by_source.get(row["source_board"], 0) + 1
            rows_by_id[row["id"]] = row
        elif last_seen < ceiling_cutoff:
            by_rule["safety_ceiling"].append(row["id"])
            by_source[row["source_board"]] = by_source.get(row["source_board"], 0) + 1
            rows_by_id[row["id"]] = row

    total = len(by_rule["complete_source_stale"]) + len(by_rule["safety_ceiling"])
    log.info(
        "plan_closures: %d job(s) would close (%d complete_source_stale, %d safety_ceiling) — by_source=%s",
        total, len(by_rule["complete_source_stale"]), len(by_rule["safety_ceiling"]), by_source,
    )
    return {"by_rule": by_rule, "by_source": by_source, "total": total, "rows_by_id": rows_by_id}


def close_jobs(
    client, *, complete_sources: frozenset[str] = frozenset(),
    run_id: str | None = None, trigger: str | None = None,
) -> dict:
    """Apply plan_closures()'s plan for real: status -> 'closed' for every
    job_id it identifies. Writes one run_metrics row (stage='closure') with
    the per-source/per-rule breakdown, same observability convention as
    every other stage in this pipeline.

    Returns {"closed": int, "by_source": {...}, "by_rule": {rule: count}}.
    """
    plan = plan_closures(client, complete_sources=complete_sources)
    job_ids = plan["by_rule"]["complete_source_stale"] + plan["by_rule"]["safety_ceiling"]
    by_rule_counts = {rule: len(ids) for rule, ids in plan["by_rule"].items()}

    if job_ids:
        # Built from plan_closures()'s OWN rows (rows_by_id) — deliberately
        # NOT a second fetch_all(.in_("id", job_ids)). That re-fetch used to
        # exist here and, against a real 1,665-job backlog, built a ~65KB
        # GET URL that Cloudflare answered with a 520 before PostgREST ever
        # saw it: ingest completed, but closure wrote nothing and
        # enrich/matching never ran that day. plan_closures already selected
        # canonical_url/source_board/content_hash for every qualifying row —
        # NOT NULL with no default, same upsert-INSERT-branch quirk as every
        # other chunked_upsert call site in this codebase (src/ingest.py,
        # src/enrich_jobs.py) — carried through here even though this write
        # only means to touch `status`. chunked_upsert's own 500-row-per-call
        # POST body chunking (not a URL filter, so not subject to the same
        # failure mode) is what keeps this safe at any backlog size.
        updates = []
        for jid in job_ids:
            row = plan["rows_by_id"][jid]
            updates.append({
                "id": jid, "canonical_url": row["canonical_url"], "source_board": row["source_board"],
                "content_hash": row["content_hash"], "status": "closed",
            })
        chunked_upsert(client.table("jobs"), updates, on_conflict="id")
        closed = len(updates)
    else:
        closed = 0

    client.table("run_metrics").insert({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "stage": "closure",
        "trigger": trigger,
        "closure_summary": {"closed": closed, "by_source": plan["by_source"], "by_rule": by_rule_counts},
    }).execute()

    log.info("close_jobs: closed %d job(s) — by_source=%s, by_rule=%s", closed, plan["by_source"], by_rule_counts)
    return {"closed": closed, "by_source": plan["by_source"], "by_rule": by_rule_counts}
