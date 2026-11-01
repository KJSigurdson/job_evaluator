"""True North v2 job closure: marks jobs.status='closed' for postings no
longer seen from a source whose most recent fetch was confirmed complete, or
— a safety ceiling for a broken/misconfigured adapter that never reports
complete=True for real — any open job not seen in _SAFETY_CEILING_DAYS
regardless of source. 'closed' is a new value in jobs.status's CHECK
constraint (see this phase's migration); no v1 table touched.

Phase 1 ships this as report-only: plan_closures() computes and returns
exactly what close_jobs() would do, without writing anything, so its counts
can be reviewed before close_jobs() is ever wired into src/pipeline_v2.py's
run(). Do not call close_jobs() in production before that review — see the
Phase 1 report for the live dry-run counts and the caveats around running it
too soon after the alias-sighting last_seen fix (src/ingest.py) deploys.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.supabase_client import chunked_upsert, fetch_all

log = logging.getLogger(__name__)

# A job closes under rule (a) once its source's last confirmed-complete fetch
# is at least this old AND the job itself hasn't been seen in that long either
# — 3 days, not 1, so a single missed/incomplete run doesn't close jobs that
# are still genuinely live (cron is daily; this is slack for a skipped day).
_RECENTLY_COMPLETE_STALE_DAYS = 3

# Rule (b): regardless of source completeness, a job not seen in this long is
# closed anyway — the backstop for a source that silently never reports
# complete=True (a broken adapter, or one disabled outright — see
# src/pipeline_v2.py's IAP note. A disabled adapter's jobs WILL eventually all
# close under this rule since nothing ever refreshes their last_seen again;
# that's a known, accepted consequence of "regardless of source", not a bug).
_SAFETY_CEILING_DAYS = 30

# A source_fetch_log row counts as "this source's fetch was recently
# confirmed complete" only within this window — stops a single complete=True
# row from several weeks ago (a source that's since gone dark) from
# permanently green-lighting rule (a) for that source forever.
_COMPLETE_SIGNAL_WINDOW_HOURS = 48

_OPEN_STATUSES = ("pending_enrich", "enriched")
_JOB_SELECT_COLUMNS = "id,canonical_url,source_board,content_hash,last_seen"


def _sources_with_recent_complete_fetch(client, *, within_hours: int = _COMPLETE_SIGNAL_WINDOW_HOURS) -> set[str]:
    """source_board values with at least one source_fetch_log row,
    complete=True, within the last *within_hours*. Rule (a) only applies to a
    source in this set — a source whose most recent fetch was incomplete (or
    whose last fetch is itself stale, e.g. a disabled adapter) must not have
    its missing postings closed on the assumption the fetch saw everything;
    rule (b)'s safety ceiling exists precisely for that case instead."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
    rows = fetch_all(
        client.table("source_fetch_log").select("source_board,complete,fetched_at").eq("complete", True)
    )
    return {r["source_board"] for r in rows if datetime.fromisoformat(r["fetched_at"]) > cutoff}


def plan_closures(client) -> dict:
    """Compute, WITHOUT writing anything, which open jobs (status in
    pending_enrich/enriched) would be closed right now and under which rule.

    Returns {"by_rule": {"complete_source_stale": [job_id, ...],
    "safety_ceiling": [job_id, ...]}, "by_source": {source_board: count},
    "total": int}. job_ids are included (not just counts) so a caller —
    close_jobs() below, or a one-off review script — can act on or inspect
    the exact set, not just a number.

    A job is counted under at most one rule: complete_source_stale takes
    priority over safety_ceiling for a job that qualifies under both (rule
    (a) already fully explains it — rule (b) is the backstop for jobs rule
    (a) DOESN'T catch, not a second vote on ones it does).
    """
    now = datetime.now(timezone.utc)
    complete_cutoff = now - timedelta(days=_RECENTLY_COMPLETE_STALE_DAYS)
    ceiling_cutoff = now - timedelta(days=_SAFETY_CEILING_DAYS)

    rows = fetch_all(client.table("jobs").select(_JOB_SELECT_COLUMNS).in_("status", list(_OPEN_STATUSES)))
    complete_sources = _sources_with_recent_complete_fetch(client)

    by_rule: dict[str, list[str]] = {"complete_source_stale": [], "safety_ceiling": []}
    by_source: dict[str, int] = {}

    for row in rows:
        last_seen = datetime.fromisoformat(row["last_seen"])
        if row["source_board"] in complete_sources and last_seen < complete_cutoff:
            by_rule["complete_source_stale"].append(row["id"])
            by_source[row["source_board"]] = by_source.get(row["source_board"], 0) + 1
        elif last_seen < ceiling_cutoff:
            by_rule["safety_ceiling"].append(row["id"])
            by_source[row["source_board"]] = by_source.get(row["source_board"], 0) + 1

    total = len(by_rule["complete_source_stale"]) + len(by_rule["safety_ceiling"])
    log.info(
        "plan_closures: %d job(s) would close (%d complete_source_stale, %d safety_ceiling) — by_source=%s",
        total, len(by_rule["complete_source_stale"]), len(by_rule["safety_ceiling"]), by_source,
    )
    return {"by_rule": by_rule, "by_source": by_source, "total": total}


def close_jobs(client) -> dict:
    """Apply plan_closures()'s plan for real: status -> 'closed' for every
    job_id it identifies. NOT called anywhere in src/pipeline_v2.py as of
    Phase 1 — see this module's docstring; wire it in only after the
    migration lands and the dry-run counts have been reviewed.

    Returns {"closed": int, "by_source": {...}, "by_rule": {rule: count}}.
    """
    plan = plan_closures(client)
    job_ids = plan["by_rule"]["complete_source_stale"] + plan["by_rule"]["safety_ceiling"]
    by_rule_counts = {rule: len(ids) for rule, ids in plan["by_rule"].items()}
    if not job_ids:
        log.info("close_jobs: nothing to close")
        return {"closed": 0, "by_source": {}, "by_rule": by_rule_counts}

    # canonical_url/source_board/content_hash are NOT NULL with no default —
    # same upsert-INSERT-branch quirk as every other chunked_upsert call site
    # in this codebase (src/ingest.py, src/enrich_jobs.py), confirmed the hard
    # way against production there — must be carried through here too even
    # though this write only means to touch `status`.
    rows = fetch_all(client.table("jobs").select("id,canonical_url,source_board,content_hash").in_("id", job_ids))
    updates = [
        {"id": r["id"], "canonical_url": r["canonical_url"], "source_board": r["source_board"],
         "content_hash": r["content_hash"], "status": "closed"}
        for r in rows
    ]
    chunked_upsert(client.table("jobs"), updates, on_conflict="id")

    log.info("close_jobs: closed %d job(s) — by_source=%s, by_rule=%s", len(updates), plan["by_source"], by_rule_counts)
    return {"closed": len(updates), "by_source": plan["by_source"], "by_rule": by_rule_counts}
