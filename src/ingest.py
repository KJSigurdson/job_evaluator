"""True North v2 ingest: adapter output -> the `jobs`/`job_aliases` tables, with
dedup and content-change detection. Additive only — writes exclusively to the v2
tables (jobs, job_aliases, run_metrics); nothing here reads or writes any v1 table
(profiles, scoring_weights, matches, seen, search_quota, pipeline_runs,
pipeline_run_users), and v1's own pipeline.py is not called from here.

Dedup has two layers:
  1. Exact canonical_url match (after job_normalize.normalize_job_url) -> the same
     posting seen again. Same content_hash -> just touch last_seen. Different
     content_hash -> the posting changed since we last saw it; update the stored
     text/hash and reset status to 'pending_enrich' so it gets re-enriched.
  2. No URL match, but (normalized org, normalized title) matches an existing job
     (dedup.org_title_key — the same key the v1 pipeline already uses for its own
     re-post fallback) -> this is very likely the same role re-posted under a new
     URL (a fresh listing, a redirected/renamed board URL, cross-posting). Recorded
     as a job_aliases row pointing at the existing job, NOT a new jobs row — the
     org/title index used for this check is updated as new jobs are decided within
     the SAME run, so two re-posts of a role that both arrive as "new" in one run
     still correctly resolve to one job + one alias, not two job rows.

Writes are batched into a small, FIXED number of round trips regardless of pool
size (the v1 seen-cache lesson: uncapped per-row writes blew the GitHub Actions
time budget once already). The batching itself follows a stricter rule learned
from that same incident: a single bulk upsert whose rows have HETEROGENEOUS key
sets gets NULL-filled for any column a given row didn't include — so "new job"
inserts, "content changed" updates, and "last_seen only" touches are each their
own call with a UNIFORM payload shape, never mixed into one heterogeneous upsert.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.adapters.base import JobPosting
from src.dedup import org_title_key
from src.job_normalize import content_hash, normalize_job_url
from src.supabase_client import chunked_insert, chunked_upsert, fetch_all

log = logging.getLogger(__name__)


@dataclass
class IngestCounts:
    ingested: int = 0
    new: int = 0
    updated: int = 0       # content changed -> re-enrich
    unchanged: int = 0     # same content -> last_seen touch only
    aliased: int = 0

    def as_dict(self) -> dict:
        return {
            "ingested": self.ingested, "new": self.new, "updated": self.updated,
            "unchanged": self.unchanged, "aliased": self.aliased,
        }


def _fetch_existing_jobs(client) -> tuple[dict[str, dict], dict[str, str]]:
    """Return (by_canonical_url, by_org_title) — one (possibly paginated) query
    over `jobs`, id/canonical_url/content_hash/org/title only (not the
    potentially-large raw_description/enrichment/embedding columns, which this
    step never reads). Uses fetch_all(), not a plain .execute(): PostgREST
    silently truncates a response at db.max_rows (1000) with no error, and an
    incomplete read here means every row past that cutoff gets wrongly decided
    "new" on this run — a real duplicate-insert/duplicate-alias bug, not just
    an incomplete report (confirmed for real once `jobs` crossed 1000 rows)."""
    rows = fetch_all(client.table("jobs").select("id,canonical_url,content_hash,org,title"))
    by_url = {row["canonical_url"]: row for row in rows}
    by_org_title = {org_title_key(row["org"] or "", row["title"] or ""): row["id"] for row in rows}
    return by_url, by_org_title


def ingest(postings: list[JobPosting], *, client, dry_run: bool = False) -> IngestCounts:
    """Ingest *postings* into `jobs`/`job_aliases`, batched, with dedup as described
    in this module's docstring. dry_run=True computes and returns the same
    IngestCounts without writing anything — no jobs/job_aliases/run_metrics call is
    made at all (mirrors this codebase's existing dry_run convention: log/return
    what WOULD happen, skip every network write).
    """
    by_url, by_org_title = _fetch_existing_jobs(client)

    now = datetime.now(timezone.utc).isoformat()
    counts = IngestCounts()
    per_board_counts: dict[str, IngestCounts] = {}

    new_jobs: list[dict] = []
    # Keyed by their upsert's conflict-target column (canonical_url / alias_url), not
    # plain lists: a source board can list the exact same canonical_url 2+ times in
    # one fetch (observed for real against 80,000 Hours), so a naive list-append lets
    # the SAME key appear twice in one bulk upsert call, which Postgres rejects
    # ("ON CONFLICT DO UPDATE command cannot affect row a second time" — confirmed
    # the hard way against production). Keying by conflict-target gives free
    # last-write-wins de-duplication: a later occurrence just overwrites the dict
    # entry for that key instead of appending a second one.
    unchanged_updates: dict[str, dict] = {}  # canonical_url -> {canonical_url, last_seen} only
    changed_updates: dict[str, dict] = {}    # canonical_url -> {canonical_url, raw_description, content_hash, last_seen, status}
    new_aliases: dict[str, dict] = {}        # alias_url -> {alias_url, job_id, source_board}

    for posting in postings:
        counts.ingested += 1
        board_counts = per_board_counts.setdefault(posting.source_board, IngestCounts())
        board_counts.ingested += 1

        canonical = normalize_job_url(posting.url)
        this_hash = content_hash(posting.description)

        existing = by_url.get(canonical)
        if existing is not None:
            if existing["content_hash"] == this_hash:
                # canonical_url/source_board/content_hash are NOT NULL with no
                # default — Postgres validates NOT NULL on an upsert's INSERT
                # branch even when the row already exists and only the ON
                # CONFLICT DO UPDATE branch will actually fire, so every payload
                # must include them regardless of which branch is "really" meant
                # to run (confirmed the hard way against production: omitting
                # them here raised a not-null violation on source_board).
                unchanged_updates[canonical] = {
                    "canonical_url": canonical,
                    "source_board": posting.source_board,
                    "content_hash": this_hash,
                    "last_seen": now,
                }
                changed_updates.pop(canonical, None)  # a later "unchanged" occurrence supersedes an earlier "changed" one
                counts.unchanged += 1
                board_counts.unchanged += 1
            else:
                changed_updates[canonical] = {
                    "canonical_url": canonical,
                    "source_board": posting.source_board,
                    "raw_description": posting.description,
                    "location_field": posting.location_field,
                    "content_hash": this_hash,
                    "last_seen": now,
                    "status": "pending_enrich",
                }
                unchanged_updates.pop(canonical, None)  # a later "changed" occurrence supersedes an earlier "unchanged" one
                counts.updated += 1
                board_counts.updated += 1
            # Keep in-memory state current so a THIRD+ occurrence of this same
            # canonical_url later in the batch compares against what this run just
            # decided, not the original pre-batch Supabase state.
            by_url[canonical] = {"content_hash": this_hash}
            continue

        key = org_title_key(posting.org, posting.title)
        matched_job_id = by_org_title.get(key)
        if matched_job_id is not None:
            new_aliases[canonical] = {
                "alias_url": canonical, "job_id": matched_job_id, "source_board": posting.source_board,
            }
            log.info(
                "Aliasing %r @ %r (%s) to existing job %s via org/title match",
                posting.title, posting.org, canonical, matched_job_id,
            )
            counts.aliased += 1
            board_counts.aliased += 1
            # Deliberately NOT added to by_url: an alias isn't a `jobs` row, so a
            # repeat of this canonical_url later in the batch should fall through
            # this same org/title-match branch again, not be compared against a
            # nonexistent jobs row. new_aliases is keyed by canonical_url, so a
            # repeat just overwrites its own dict entry — no duplicate-key risk.
            continue

        # Genuinely new job. Client-generated id so this run's own later postings
        # can alias against it (see module docstring) without a read-back round trip.
        new_id = str(uuid.uuid4())
        new_jobs.append({
            "id": new_id,
            "canonical_url": canonical,
            "source_board": posting.source_board,
            "org": posting.org,
            "title": posting.title,
            "content_hash": this_hash,
            "raw_description": posting.description,
            "location_field": posting.location_field,
            "first_seen": now,
            "last_seen": now,
            "status": "pending_enrich",
        })
        by_org_title[key] = new_id  # visible to later postings in this same run
        # Also visible by URL: a source board can list the exact same canonical_url
        # twice in one fetch (observed for real against 80,000 Hours — not a
        # re-post under a different URL, the literal same URL twice). Without this,
        # the second occurrence would ALSO be decided "new", producing two inserts
        # with the same canonical_url and crashing on the unique constraint. The
        # write order below (new_jobs inserted before changed/unchanged updates)
        # guarantees this row exists by the time any such follow-up update runs.
        by_url[canonical] = {"content_hash": this_hash}
        counts.new += 1
        board_counts.new += 1

    if dry_run:
        log.info(
            "DRY RUN — ingest would write: %d new job(s), %d updated (content changed), "
            "%d unchanged (last_seen touch), %d aliased",
            counts.new, counts.updated, counts.unchanged, counts.aliased,
        )
        return counts

    # chunked_insert/chunked_upsert, not one call for the whole list: an
    # oversized single statement can hit Postgres's statement timeout (confirmed
    # for real in src/enrich_jobs.py's equivalent write) and lose the entire
    # batch, not just rows past some cutoff.
    if new_jobs:
        chunked_insert(client.table("jobs"), new_jobs)
    if changed_updates:
        chunked_upsert(client.table("jobs"), list(changed_updates.values()), on_conflict="canonical_url")
    if unchanged_updates:
        chunked_upsert(client.table("jobs"), list(unchanged_updates.values()), on_conflict="canonical_url")
    if new_aliases:
        chunked_upsert(client.table("job_aliases"), list(new_aliases.values()), on_conflict="alias_url")

    per_board_payload = {board: c.as_dict() for board, c in per_board_counts.items()}
    client.table("run_metrics").insert({
        "run_at": now,
        "jobs_ingested": counts.ingested,
        "per_board": per_board_payload,
    }).execute()

    log.info(
        "Ingest complete: %d posting(s) -> %d new, %d updated, %d unchanged, %d aliased",
        counts.ingested, counts.new, counts.updated, counts.unchanged, counts.aliased,
    )
    return counts


# ---------------------------------------------------------------------------
# CLI entry point — all registered adapters, fetched and combined into ONE
# postings list before calling ingest() once, so cross-board org/title aliasing
# resolves within the same batch (the same job cross-posted on 80k AND
# ProbablyGood, say, must alias to one jobs row, not two independent "new"
# decisions from two separate ingest() calls). Graceful per-source failure:
# one adapter erroring is logged and skipped, not fatal to the others — same
# principle as v1's per-source scrape failure handling in pipeline.py.
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "80000_hours": "src.adapters.eighty_thousand_hours.EightyThousandHoursAdapter",
    "probably_good": "src.adapters.probably_good.ProbablyGoodAdapter",
    "iap": "src.adapters.iap.IAPAdapter",
}


def _load_adapter(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="True North v2 ingest")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    parser.add_argument(
        "--sources", nargs="+", choices=list(_ADAPTERS), default=list(_ADAPTERS),
        help="Subset of source boards to fetch (default: all)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from src.supabase_client import get_client

    postings: list[JobPosting] = []
    for name in args.sources:
        try:
            adapter = _load_adapter(_ADAPTERS[name])
            fetched = adapter.fetch()
        except Exception:
            log.exception("Adapter %r failed to fetch — skipping this source", name)
            continue
        print(f"Fetched {len(fetched)} posting(s) from {name}")
        postings.extend(fetched)

    print(f"Total: {len(postings)} posting(s) across {len(args.sources)} source(s)")

    result = ingest(postings, client=get_client(), dry_run=args.dry_run)
    print(result.as_dict())
