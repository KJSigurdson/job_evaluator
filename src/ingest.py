"""True North v2 ingest: adapter output -> the `jobs`/`job_aliases` tables, with
dedup and content-change detection. Additive only — writes exclusively to the v2
tables (jobs, job_aliases, run_metrics); nothing here reads or writes any v1 table
(profiles, scoring_weights, matches, seen, search_quota, pipeline_runs,
pipeline_run_users), and v1's own pipeline.py is not called from here.

Dedup has two layers:
  1. Exact canonical_url match (after job_normalize.normalize_job_url) -> the same
     posting seen again.
       - Different source_board than what's stored, and NEITHER board is
         restricted (see _RESTRICTED_SOURCE_BOARDS), or BOTH are: this is a
         cross-board rediscovery of an already-known job (the same posting
         legitimately cross-listed on two boards under one URL), NOT a content
         update — each board formats `description` completely differently
         (80k/ProbablyGood: LLM-prose bullets; IAP: a terse "org | title |
         location | comp | cause_area" string), so comparing content_hash
         across boards here would flag nearly every shared posting as
         "changed" every single run forever, regardless of whether the
         posting itself ever changed (confirmed for real: 314/827 of one
         board's postings on a single day resolved to a `jobs` row already
         stamped with a DIFFERENT source_board, entirely explaining that
         day's ~320 "updated" count). Only last_seen is touched;
         raw_description/content_hash/source_board are left as whichever
         board's version is already stored.
       - Different source_board, and the STORED board is restricted while the
         INCOMING one isn't: a permitted board's sighting of a URL currently
         attributed to a restricted board always wins — the stored
         raw_description/content_hash/source_board are overwritten with the
         incoming (permitted) posting's, same enrichment-preserving status
         logic as a same-board real content change below (confirmed live:
         72 of 85 still-active IAP-sourced jobs also appear, same URL, in
         80k's current feed — IAP simply got ingested first, historically,
         and nothing until now ever let a later permitted-board sighting
         correct that).
       - Same source_board, same content_hash -> just touch last_seen.
       - Same source_board, different content_hash -> a real same-source change.
         If the normalized-text similarity to the stored text is still at/above
         RE_ENRICH_SIMILARITY_THRESHOLD (job_normalize.text_similarity), treat it
         as cosmetic: refresh the stored text/hash but do NOT reset status (no
         re-enrich spend for a one-word tweak). Otherwise, update the stored
         text/hash and reset status to 'pending_enrich' so it gets re-enriched.
  2. No URL match, but (normalized org, normalized title) matches an existing job
     (dedup.org_title_key — the same key the v1 pipeline already uses for its own
     re-post fallback) -> this is very likely the same role re-posted under a new
     URL (a fresh listing, a redirected/renamed board URL, cross-posting).
       - If the matched job's stored board is restricted and the incoming
         posting's board isn't: same "permitted wins" precedence as above, but
         the URL itself also changed, so this ALSO re-points the job's
         canonical_url to the incoming (permitted) URL and demotes the OLD
         (restricted-board) URL to a job_aliases row instead of the reverse —
         the matched job keeps its id (so matches_v2/batch_results references
         stay valid), only canonical_url/source_board/raw_description/
         content_hash/location_field change.
       - Otherwise: recorded as a job_aliases row pointing at the existing job,
         NOT a new jobs row — the org/title index used for this check is
         updated as new jobs are decided within the SAME run, so two re-posts
         of a role that both arrive as "new" in one run still correctly
         resolve to one job + one alias, not two job rows. The MATCHED job's
         own last_seen is also bumped (see the post-loop alias_matched_job_ids
         step) — without this, a job only ever rediscovered via alias would
         never have its own row's last_seen touched at all, and
         src/closure.py's last_seen-based rules would wrongly treat it as gone.

A canonical_url match whose stored row is status='closed' (see src/closure.py)
reopens instead of going through the comparisons above: 'enriched' if an
enrichment already exists (plus reenrich_requested_at if the content ALSO
changed since closing — same as the real-change branch below), else
'pending_enrich'. Checked first, before the cross-board/hash comparisons,
since "it's back" matters more here than which board revived it.

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
from src.job_normalize import RE_ENRICH_SIMILARITY_THRESHOLD, content_hash, normalize_job_url, text_similarity
from src.supabase_client import chunked_insert, chunked_upsert, fetch_all, insert_run_metrics

log = logging.getLogger(__name__)

# Boards True North isn't permitted to pull postings' CONTENT from going
# forward (the board itself may still legitimately be the one that first
# surfaced a given URL/org-title, historically) — a permitted board's
# sighting of the same job always overrides a restricted board's stored
# version, never the other way round. IAP specifically: the referral sheet
# is fine to use for discovery/dedup, just not as the board of record for a
# job's own description once a permitted board also carries it.
_RESTRICTED_SOURCE_BOARDS = frozenset({"iap"})


@dataclass
class IngestCounts:
    ingested: int = 0
    new: int = 0
    updated: int = 0       # content changed (same source board) -> re-enrich
    unchanged: int = 0     # same content -> last_seen touch only
    aliased: int = 0
    cross_board: int = 0   # same canonical_url, different source board, neither/both restricted -> last_seen
                           # touch only, no re-enrich
    minor_update: int = 0  # content_hash changed but normalized-text similarity is above threshold -> text
                           # refreshed, no re-enrich (see RE_ENRICH_SIMILARITY_THRESHOLD)
    reopened: int = 0      # status='closed' seen again -> 'enriched' or 'pending_enrich' (see src/closure.py)
    board_upgraded: int = 0  # a permitted board's sighting overrode a restricted board's stored version
                             # (same URL or, via org/title match, a different one) -> re-enrich queued

    def as_dict(self) -> dict:
        return {
            "ingested": self.ingested, "new": self.new, "updated": self.updated,
            "unchanged": self.unchanged, "aliased": self.aliased,
            "cross_board": self.cross_board, "minor_update": self.minor_update,
            "reopened": self.reopened, "board_upgraded": self.board_upgraded,
        }


def _fetch_existing_jobs(client) -> tuple[dict[str, dict], dict[str, str], dict[str, dict]]:
    """Return (by_canonical_url, by_org_title, by_id) — one (possibly paginated)
    query over `jobs`. source_board is needed to detect a cross-board same-URL
    collision (see the module docstring's "layer 1" note); raw_description is
    needed to compute normalized-text similarity when content_hash differs (see
    RE_ENRICH_SIMILARITY_THRESHOLD). enrichment (presence only, not its
    contents) decides whether a real content change keeps a job matchable
    (see the real-change branch below); reenrich_requested_at is read so that
    branch only ever sets it when it's currently null, never bumping an
    already-pending request. embedding is still excluded, not needed by this
    step. by_id (job_id -> row) backs the alias-sighting last_seen bump below
    — a job matched via org/title (no URL in by_url for IT specifically) still
    needs its OWN stored row's canonical_url/source_board/content_hash to
    build that touch-only update. Uses fetch_all(), not a plain .execute():
    PostgREST silently truncates a response at db.max_rows (1000) with no
    error, and an incomplete read here means every row past that cutoff gets
    wrongly decided "new" on this run — a real duplicate-insert/duplicate-alias
    bug, not just an incomplete report (confirmed for real once `jobs` crossed
    1000 rows)."""
    rows = fetch_all(client.table("jobs").select(
        "id,canonical_url,content_hash,org,title,source_board,raw_description,enrichment,reenrich_requested_at,status"
    ))
    by_url = {row["canonical_url"]: row for row in rows}
    by_org_title = {org_title_key(row["org"] or "", row["title"] or ""): row["id"] for row in rows}
    by_id = {row["id"]: row for row in rows}
    return by_url, by_org_title, by_id


def ingest(
    postings: list[JobPosting], *, client, dry_run: bool = False,
    run_id: str | None = None, trigger: str | None = None,
) -> IngestCounts:
    """Ingest *postings* into `jobs`/`job_aliases`, batched, with dedup as described
    in this module's docstring. dry_run=True computes and returns the same
    IngestCounts without writing anything — no jobs/job_aliases/run_metrics call is
    made at all (mirrors this codebase's existing dry_run convention: log/return
    what WOULD happen, skip every network write).

    run_id/trigger are stamped onto the run_metrics row this call writes
    (stage='ingest') — run_id ties every stage of one pipeline_v2 run together;
    trigger is 'scheduled' or 'manual' (see pipeline_v2.py). Both are optional
    (written as NULL) for standalone/CLI use outside pipeline_v2.
    """
    by_url, by_org_title, by_id = _fetch_existing_jobs(client)

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
    unchanged_updates: dict[str, dict] = {}  # canonical_url -> {canonical_url, source_board, content_hash, last_seen} only
    changed_updates: dict[str, dict] = {}    # canonical_url -> {canonical_url, raw_description, content_hash, last_seen, status}
    minor_updates: dict[str, dict] = {}      # canonical_url -> same shape as changed_updates minus status (no re-enrich)
    reopened_updates: dict[str, dict] = {}   # canonical_url -> same shape as changed_updates, status='enriched'/'pending_enrich'
                                              # (never 'closed') — kept separate from changed_updates so this
                                              # module's counts distinguish a genuine content edit from a revival.
    new_aliases: dict[str, dict] = {}        # alias_url -> {alias_url, job_id, source_board}
    board_upgrade_by_id: dict[str, dict] = {}  # job_id -> {id, canonical_url, source_board, raw_description,
                                                # location_field, content_hash, last_seen, status,
                                                # reenrich_requested_at?} — a permitted board's org/title match
                                                # against a restricted-board-sourced job re-points that job's
                                                # OWN canonical_url, unlike every other bucket here which is
                                                # keyed by canonical_url and upserted on_conflict="canonical_url"
                                                # (this one is upserted on_conflict="id" instead, since the url
                                                # itself is what's changing).
    alias_matched_job_ids: set[str] = set()  # job_ids the alias branch below resolved to this run — see
                                              # the post-loop last_seen bump: without it, a job only ever
                                              # seen via alias match (re-posted under a new URL every time)
                                              # never gets its OWN last_seen touched, and a closure step
                                              # keyed on last_seen would wrongly treat it as gone stale.

    for posting in postings:
        counts.ingested += 1
        board_counts = per_board_counts.setdefault(posting.source_board, IngestCounts())
        board_counts.ingested += 1

        canonical = normalize_job_url(posting.url)
        this_hash = content_hash(posting.description)

        existing = by_url.get(canonical)
        if existing is not None:
            if existing.get("status") == "closed":
                # Seen again — reopen (see src/closure.py and this module's
                # docstring). Checked before cross-board/hash comparisons:
                # "it's back" matters more than which board revived it or
                # whether the text happens to match what it was closed with.
                #
                # source_board on reopen: PRESERVE the original owner, same
                # "permitted beats restricted, otherwise leave it alone"
                # precedence as the cross-board touch branch below — not
                # posting.source_board unconditionally. Confirmed for real:
                # AAC rediscovering a closed probably_good job relabeled it
                # 'aac' outright, even though probably_good isn't restricted
                # and had every right to keep owning it. Only when the
                # ORIGINAL board is restricted and the incoming one isn't
                # does the incoming board actually win — e.g. AAC reviving a
                # closed IAP-owned job, which SHOULD end up attributed to
                # AAC, the same as a live cross-board touch would.
                original_board = existing["source_board"]
                original_restricted = original_board in _RESTRICTED_SOURCE_BOARDS
                incoming_restricted = posting.source_board in _RESTRICTED_SOURCE_BOARDS
                reopened_source_board = (
                    posting.source_board if (original_restricted and not incoming_restricted) else original_board
                )
                has_enrichment = existing.get("enrichment") is not None
                content_changed = existing["content_hash"] != this_hash
                reopen_payload = {
                    "canonical_url": canonical,
                    "source_board": reopened_source_board,
                    "raw_description": posting.description,
                    "location_field": posting.location_field,
                    "content_hash": this_hash,
                    "last_seen": now,
                }
                if has_enrichment:
                    reopen_payload["status"] = "enriched"
                    if content_changed and existing.get("reenrich_requested_at") is None:
                        reopen_payload["reenrich_requested_at"] = now
                else:
                    reopen_payload["status"] = "pending_enrich"
                reopened_updates[canonical] = reopen_payload
                counts.reopened += 1
                board_counts.reopened += 1
                by_url[canonical] = {
                    "content_hash": this_hash, "source_board": reopened_source_board,
                    "raw_description": posting.description,
                    "enrichment": existing.get("enrichment"),
                    "reenrich_requested_at": reopen_payload.get("reenrich_requested_at", existing.get("reenrich_requested_at")),
                    "status": reopen_payload["status"],
                }
                continue
            if existing["source_board"] != posting.source_board:
                existing_restricted = existing["source_board"] in _RESTRICTED_SOURCE_BOARDS
                incoming_restricted = posting.source_board in _RESTRICTED_SOURCE_BOARDS
                if existing_restricted and not incoming_restricted:
                    # A permitted board's sighting of a URL currently
                    # attributed to a restricted board — permitted always
                    # wins (see module docstring and src/ingest.py's
                    # _RESTRICTED_SOURCE_BOARDS). Same URL, so this is really
                    # just "whose description do we trust", not a URL
                    # change — treated exactly like a same-board real
                    # content change below, using the incoming posting's
                    # data, just across a board boundary.
                    has_enrichment = existing.get("enrichment") is not None
                    board_upgrade_payload = {
                        "canonical_url": canonical,
                        "source_board": posting.source_board,
                        "raw_description": posting.description,
                        "location_field": posting.location_field,
                        "content_hash": this_hash,
                        "last_seen": now,
                    }
                    if has_enrichment:
                        board_upgrade_payload["status"] = "enriched"
                        board_upgrade_payload["reenrich_requested_at"] = existing.get("reenrich_requested_at") or now
                    else:
                        board_upgrade_payload["status"] = "pending_enrich"
                    changed_updates[canonical] = board_upgrade_payload
                    unchanged_updates.pop(canonical, None)
                    minor_updates.pop(canonical, None)
                    counts.board_upgraded += 1
                    board_counts.board_upgraded += 1
                    by_url[canonical] = {
                        "content_hash": this_hash, "source_board": posting.source_board,
                        "raw_description": posting.description,
                        "enrichment": existing.get("enrichment"),
                        "reenrich_requested_at": board_upgrade_payload.get("reenrich_requested_at"),
                        "status": board_upgrade_payload["status"],
                    }
                    continue
                # Cross-board rediscovery of an already-known job (see module
                # docstring) — NOT a content update: neither board is
                # restricted, or both are (vacuously, with today's single
                # restricted board, this is just "neither"). Leave
                # raw_description/content_hash/source_board as whichever
                # board's version is already stored; only touch last_seen.
                # Deliberately does NOT update the in-memory by_url cache
                # below (this row's stored state didn't change), so a later
                # occurrence of this same URL from yet another board in this
                # same batch still compares against the original stored
                # owner, not this posting.
                unchanged_updates[canonical] = {
                    "canonical_url": canonical,
                    "source_board": existing["source_board"],
                    "content_hash": existing["content_hash"],
                    "last_seen": now,
                }
                changed_updates.pop(canonical, None)
                minor_updates.pop(canonical, None)
                counts.cross_board += 1
                board_counts.cross_board += 1
                continue
            # Default for the in-memory by_url carry-forward below: unchanged/
            # minor branches don't touch reenrich_requested_at at all (their
            # upsert payloads omit the key, so the stored value is untouched),
            # so the cache should likewise just carry forward whatever's
            # already there. Only the real-change branch below overrides this.
            next_reenrich_requested_at = existing.get("reenrich_requested_at")
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
                changed_updates.pop(canonical, None)  # a later "unchanged" occurrence supersedes an earlier "changed"/"minor" one
                minor_updates.pop(canonical, None)
                counts.unchanged += 1
                board_counts.unchanged += 1
            elif text_similarity(existing.get("raw_description") or "", posting.description) >= RE_ENRICH_SIMILARITY_THRESHOLD:
                # Same board, hash differs, but the text is still near-identical
                # (e.g. a single word/punctuation tweak) — refresh the stored
                # text/hash so future comparisons are against the latest fetch,
                # but don't spend a re-enrich on a cosmetic change.
                minor_updates[canonical] = {
                    "canonical_url": canonical,
                    "source_board": posting.source_board,
                    "raw_description": posting.description,
                    "location_field": posting.location_field,
                    "content_hash": this_hash,
                    "last_seen": now,
                }
                changed_updates.pop(canonical, None)
                unchanged_updates.pop(canonical, None)
                counts.minor_update += 1
                board_counts.minor_update += 1
            else:
                # A job that's already been successfully enriched once keeps
                # status='enriched' through a real content change instead of
                # flipping to pending_enrich — reenrich_requested_at records
                # that a re-enrich is owed, WITHOUT pulling the job out of
                # _fetch_enriched_jobs_pool's matching pool for the up-to-
                # _RE_ENRICH_COOLDOWN_DAYS it takes enrich_jobs.py to actually
                # get back to it (see that module's selection query). A job
                # with no enrichment yet (still mid-ingest, never successfully
                # enriched) has nothing to preserve — pending_enrich as before.
                # Only set reenrich_requested_at when it's currently null: a
                # second real change before the first request is even
                # serviced must not push the cooldown clock out further by
                # overwriting an earlier request time.
                has_enrichment = existing.get("enrichment") is not None
                payload = {
                    "canonical_url": canonical,
                    "source_board": posting.source_board,
                    "raw_description": posting.description,
                    "location_field": posting.location_field,
                    "content_hash": this_hash,
                    "last_seen": now,
                }
                if has_enrichment:
                    payload["status"] = "enriched"
                    next_reenrich_requested_at = existing.get("reenrich_requested_at") or now
                    payload["reenrich_requested_at"] = next_reenrich_requested_at
                else:
                    payload["status"] = "pending_enrich"
                    next_reenrich_requested_at = None
                changed_updates[canonical] = payload
                unchanged_updates.pop(canonical, None)  # a later "changed" occurrence supersedes an earlier "unchanged"/"minor" one
                minor_updates.pop(canonical, None)
                counts.updated += 1
                board_counts.updated += 1
            # Keep in-memory state current so a THIRD+ occurrence of this same
            # canonical_url later in the batch compares against what this run just
            # decided, not the original pre-batch Supabase state. enrichment
            # itself never changes here (ingest never writes it — only
            # enrich_jobs.py does) so it's always safe to carry forward
            # unchanged; reenrich_requested_at carries forward whatever this
            # occurrence's payload just decided, so a third occurrence doesn't
            # wrongly re-derive has_enrichment=False from a cache entry that
            # dropped the field.
            by_url[canonical] = {
                "content_hash": this_hash, "source_board": posting.source_board,
                "raw_description": posting.description,
                "enrichment": existing.get("enrichment"),
                "reenrich_requested_at": next_reenrich_requested_at,
                # Never 'closed' by this point — the reopen branch above
                # handles that case with its own `continue` before reaching
                # here, and none of unchanged/minor/changed ever sets it.
                "status": existing.get("status"),
            }
            continue

        key = org_title_key(posting.org, posting.title)
        matched_job_id = by_org_title.get(key)
        if matched_job_id is not None:
            matched_row = by_id.get(matched_job_id)
            matched_restricted = matched_row is not None and matched_row.get("source_board") in _RESTRICTED_SOURCE_BOARDS
            incoming_restricted = posting.source_board in _RESTRICTED_SOURCE_BOARDS
            if matched_restricted and not incoming_restricted:
                # Permitted board's org/title match against a job whose
                # CANONICAL row is currently attributed to a restricted
                # board — same "permitted always wins" precedence as the
                # direct-URL case above, but the URL itself ALSO changes
                # here: re-point canonical_url to the incoming (permitted)
                # URL and demote the OLD (restricted-board) URL to a
                # job_aliases row instead of the reverse. The job keeps its
                # id, so matches_v2/batch_results references stay valid.
                has_enrichment = matched_row.get("enrichment") is not None
                old_canonical_url = matched_row["canonical_url"]
                board_upgrade_payload = {
                    "id": matched_job_id,
                    "canonical_url": canonical,
                    "source_board": posting.source_board,
                    "raw_description": posting.description,
                    "location_field": posting.location_field,
                    "content_hash": this_hash,
                    "last_seen": now,
                }
                if has_enrichment:
                    board_upgrade_payload["status"] = "enriched"
                    board_upgrade_payload["reenrich_requested_at"] = matched_row.get("reenrich_requested_at") or now
                else:
                    board_upgrade_payload["status"] = "pending_enrich"
                board_upgrade_by_id[matched_job_id] = board_upgrade_payload
                new_aliases[old_canonical_url] = {
                    "alias_url": old_canonical_url, "job_id": matched_job_id,
                    "source_board": matched_row["source_board"],
                }
                log.info(
                    "Board-upgrading job %s: %r @ %r now sourced from %s (was %s)",
                    matched_job_id, posting.title, posting.org, posting.source_board, matched_row["source_board"],
                )
                counts.board_upgraded += 1
                board_counts.board_upgraded += 1
                # Keep by_id/by_url consistent for a possible THIRD+
                # occurrence of this org/title key later in this same run —
                # the job is no longer on a restricted board.
                by_id[matched_job_id] = {
                    **matched_row, "canonical_url": canonical, "source_board": posting.source_board,
                    "content_hash": this_hash,
                }
                by_url[canonical] = {
                    "content_hash": this_hash, "source_board": posting.source_board,
                    "raw_description": posting.description, "enrichment": matched_row.get("enrichment"),
                    "reenrich_requested_at": board_upgrade_payload.get("reenrich_requested_at"),
                    "status": board_upgrade_payload["status"],
                }
                continue
            new_aliases[canonical] = {
                "alias_url": canonical, "job_id": matched_job_id, "source_board": posting.source_board,
            }
            log.info(
                "Aliasing %r @ %r (%s) to existing job %s via org/title match",
                posting.title, posting.org, canonical, matched_job_id,
            )
            counts.aliased += 1
            board_counts.aliased += 1
            alias_matched_job_ids.add(matched_job_id)
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
        by_url[canonical] = {
            "content_hash": this_hash, "source_board": posting.source_board,
            "raw_description": posting.description, "enrichment": None,
            "reenrich_requested_at": None, "status": "pending_enrich",
        }
        counts.new += 1
        board_counts.new += 1

    # Alias-matched jobs get their OWN row's last_seen bumped too — a job only
    # ever re-discovered via org/title alias (e.g. re-posted under a new URL
    # every time) would otherwise never have its last_seen touched at all,
    # since new_aliases only ever writes to job_aliases, never to `jobs`
    # itself. Deferred to here (after the main loop, not inline in the alias
    # branch above) so it only ever applies when that job's canonical_url
    # wasn't ALSO directly touched this run by the URL-match branch above —
    # that branch's own payload (whichever of changed/minor/unchanged it
    # landed in) already carries a fresher last_seen and, for the "changed"
    # case, a content_hash/status transition an alias-bump's stale snapshot
    # must not clobber. A job_id with no by_id entry is one this run itself
    # just inserted (new_jobs already stamped it with last_seen=now) — no
    # action needed.
    for job_id in alias_matched_job_ids:
        row = by_id.get(job_id)
        if row is None:
            continue
        own_canonical = row["canonical_url"]
        if own_canonical in changed_updates or own_canonical in minor_updates or own_canonical in unchanged_updates:
            continue
        unchanged_updates[own_canonical] = {
            "canonical_url": own_canonical,
            "source_board": row["source_board"],
            "content_hash": row["content_hash"],
            "last_seen": now,
        }

    if dry_run:
        log.info(
            "DRY RUN — ingest would write: %d new job(s), %d updated (content changed), %d minor "
            "(cosmetic, no re-enrich), %d cross-board (no re-enrich), %d unchanged (last_seen touch), "
            "%d aliased, %d reopened, %d board-upgraded",
            counts.new, counts.updated, counts.minor_update, counts.cross_board, counts.unchanged,
            counts.aliased, counts.reopened, counts.board_upgraded,
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
    if board_upgrade_by_id:
        # on_conflict="id", not "canonical_url" — this bucket changes a
        # job's canonical_url itself, so the conflict target has to be the
        # column that ISN'T changing (see board_upgrade_by_id's own comment
        # at declaration).
        chunked_upsert(client.table("jobs"), list(board_upgrade_by_id.values()), on_conflict="id")
    if minor_updates:
        chunked_upsert(client.table("jobs"), list(minor_updates.values()), on_conflict="canonical_url")
    if reopened_updates:
        chunked_upsert(client.table("jobs"), list(reopened_updates.values()), on_conflict="canonical_url")
    if unchanged_updates:
        chunked_upsert(client.table("jobs"), list(unchanged_updates.values()), on_conflict="canonical_url")
    if new_aliases:
        chunked_upsert(client.table("job_aliases"), list(new_aliases.values()), on_conflict="alias_url")

    per_board_payload = {board: c.as_dict() for board, c in per_board_counts.items()}
    insert_run_metrics(client, {
        "run_at": now,
        "run_id": run_id,
        "stage": "ingest",
        "trigger": trigger,
        "jobs_ingested": counts.ingested,
        "per_board": per_board_payload,
    })

    log.info(
        "Ingest complete: %d posting(s) -> %d new, %d updated, %d minor, %d cross-board, %d unchanged, "
        "%d aliased, %d reopened, %d board-upgraded",
        counts.ingested, counts.new, counts.updated, counts.minor_update, counts.cross_board,
        counts.unchanged, counts.aliased, counts.reopened, counts.board_upgraded,
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

    result = ingest(postings, client=get_client(), dry_run=args.dry_run, run_id=str(uuid.uuid4()), trigger="manual")
    print(result.as_dict())
