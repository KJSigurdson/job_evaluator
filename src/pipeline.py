"""Top-level orchestration. Scrapes every source ONCE into a shared pool, then loops
over every Supabase-registered user and gates/scores/enriches that shared pool against
their own profile, weights, and thresholds. Writes RunLog.

Usage:
    python -m src.pipeline [--dry-run] [--limit N]

--dry-run   still reads profiles/scoring_weights/seen/matches from Supabase (needed to
            build accurate per-user skip-sets), but skips seen/matches WRITES and prints
            would-be matches instead. SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY are still
            required even in dry-run, since user context comes from Supabase.
--limit N   caps the shared postings pool after the recency/staleness filter (a single
            global cap applied once, before the per-user loop — not per user). The pool
            is shuffled (seeded on run_id) immediately before the cap is applied, so a
            capped run draws roughly proportionally across sources instead of always
            favoring whichever source was scraped first.
--only-user-id ID   scopes the run to a single user (a one-off request) instead of
            every registered user. When set and not --dry-run, writes
            running → complete/failed status to `search_quota` for that user — the
            normal multi-user daily run never touches that table.

Seen-table writes are batched per user (src/seen_store.py): record_verdict() only
queues a payload in memory during the gate/Tier1/Tier2 loops; upsert_verdicts()
flushes everything queued for a user in one chunked call after that user's Tier 2
loop finishes, instead of one Supabase round-trip per posting. With an uncapped
shared pool (thousands of postings) across several users, per-posting writes were
the dominant cost and blew the workflow's time budget.

Match-digest email (src/notify.py):
    At the end of each user's iteration, if profile["email_on_match"] is true and at
    least one match was inserted THIS run (not pre-existing ones filtered out by the
    skip-set) and the run isn't --dry-run, one digest email is sent for that user's
    new matches. This is strictly best-effort: notify.send_match_digest() never
    raises, and the call site here is wrapped in try/except as a second layer of
    defense — an email failure can never fail the run, flip search_quota to 'failed',
    or stop processing of subsequent users.

Staleness note (14-day cutoff for deadline-less postings):
    Postings with neither `deadline` nor `posted_at` (e.g. IAP rows) have no natural
    recency signal, so we anchor staleness to `first_seen` in the `seen` table instead.
    Because the recency filter runs ONCE, before the per-user loop, it needs a
    user-independent signal — so it uses the EARLIEST first_seen across ALL users for
    that canonical_url (seen_store.fetch_global_first_seen). If nobody has ever seen a
    posting, it's fresh and kept. Once the earliest sighting is >14 days old, the
    posting is dropped from the shared pool for every user, on the theory that if it
    hasn't resolved (gated out / scored / inserted) for anyone in 14 days, it's not
    worth continuing to spend LLM calls re-evaluating it. Note this is orthogonal to
    the ordinary per-user skip-set: once ANY terminal verdict is recorded for a user,
    that user's skip-set filters the posting out immediately and permanently regardless
    of this 14-day window — the window only matters for postings that keep failing to
    resolve (e.g. repeated parse failures, which are deliberately never cached).
"""
from __future__ import annotations

import logging
import os
import random
import uuid
from datetime import date, datetime, timezone

from src import gate as gate_mod
from src import scoring as scoring_mod
from src.dedup import canonical_url
from src.enrich import EnrichmentError, enrich
from src.matches_store import fetch_existing_match_urls, upsert_match
from src.notify import send_match_digest
from src.quota_store import set_status
from src.schemas import (
    MatchRow,
    ParseFailure,
    RunCounts,
    RunLog,
    SourceResult,
    UserRunResult,
)
from src.scoring import ScoringError
from src.seen_store import fetch_global_first_seen, load_seen_urls, record_verdict, upsert_verdicts
from src.sources import eightyk, iap, probablygood
from src.supabase_client import get_client
from src.user_store import fetch_users

log = logging.getLogger(__name__)

_TIER1_MODEL_DEFAULT  = "claude-haiku-4-5-20251001"
_TIER2_MODEL_DEFAULT  = "claude-sonnet-4-6"
_RECENCY_DAYS_DEFAULT = 14
_STALE_CUTOFF_DAYS    = 14  # first-seen cutoff for deadline-less, posted_at-less postings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    only_user_id: str | None = None,
    _client=None,
) -> RunLog:
    """Run the full pipeline. Returns a RunLog regardless of per-source/per-user failures.

    dry_run=True       — skips seen/matches writes; still reads Supabase for user context.
    limit=N            — caps the shared postings pool after recency/staleness filtering.
    only_user_id=X     — scope the run to a single user X instead of every registered
                          user. When set and not dry_run, writes running/complete/failed
                          status to `search_quota` for X — the normal multi-user run
                          (only_user_id=None) never touches that table.
    _client            — injectable Supabase client (tests pass a fake; omit in production).
    """
    run_id     = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc)
    today      = date.today()

    recency_days = int(os.environ.get("RECENCY_DAYS") or _RECENCY_DAYS_DEFAULT)
    tier1_model  = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT
    tier2_model  = os.environ.get("TIER2_MODEL") or _TIER2_MODEL_DEFAULT

    client = _client if _client is not None else get_client()
    track_quota = only_user_id is not None and not dry_run
    succeeded = False

    try:
        if track_quota:
            set_status(client, only_user_id, "running")

        # -------------------------------------------------------------------------
        # 0. Load users — a user is only processed if they have a profile + weights row
        # -------------------------------------------------------------------------
        users = fetch_users(client, only_user_id=only_user_id)
        if not users:
            if track_quota:
                # A one-off request for a user with no profile/weights row can't be
                # fulfilled — surface it as a failure rather than silently no-op'ing.
                raise RuntimeError(
                    f"only_user_id={only_user_id}: no profile+scoring_weights row found"
                )
            log.warning("No users with both a profile and scoring_weights row — nothing to do")
            return RunLog(
                run_id=run_id, timestamp=started_at,
                model=f"tier1={tier1_model},tier2={tier2_model}", temperature=0.0,
                counts=RunCounts(),
            )

        # -------------------------------------------------------------------------
        # 1. Scrape sources — ONCE, shared across every user. Each in its own try/except.
        # -------------------------------------------------------------------------
        counts: RunCounts = RunCounts()
        source_results: list[SourceResult] = []
        all_postings = []

        for src_name, src_fn in [
            ("iap",           iap.fetch),
            ("80k",           eightyk.fetch),
            ("probably_good", probablygood.fetch),
        ]:
            try:
                postings = src_fn()
                all_postings.extend(postings)
                counts.scraped += len(postings)
                source_results.append(SourceResult(
                    source=src_name, success=True, postings_scraped=len(postings)
                ))
                log.info("Source %-15s scraped %d posting(s)", src_name, len(postings))
            except Exception as exc:
                log.error("Source %s FAILED: %s", src_name, exc, exc_info=True)
                source_results.append(SourceResult(
                    source=src_name, success=False, error=str(exc)
                ))

        # -------------------------------------------------------------------------
        # 2. Shared recency + staleness filter (once, before the per-user loop)
        # -------------------------------------------------------------------------
        global_first_seen = fetch_global_first_seen(client)

        fresh: list = []
        recency_dropped = 0
        stale_dropped = 0
        for p in all_postings:
            if p.posted_at is not None:
                if (today - p.posted_at).days <= recency_days:
                    fresh.append(p)
                else:
                    recency_dropped += 1
            elif p.deadline is not None:
                fresh.append(p)  # exempt: has a deadline, just no posted_at
            else:
                gfs = global_first_seen.get(canonical_url(p.url))
                if gfs is None or (today - gfs.date()).days <= _STALE_CUTOFF_DAYS:
                    fresh.append(p)
                else:
                    stale_dropped += 1

        counts.recency_dropped = recency_dropped
        counts.stale_dropped = stale_dropped
        log.info(
            "Recency filter: dropped %d (>%dd old) + %d (stale, first seen >%dd ago) → %d fresh",
            recency_dropped, recency_days, stale_dropped, _STALE_CUTOFF_DAYS, len(fresh),
        )

        # -------------------------------------------------------------------------
        # 3. Apply --limit — a single global cap on the shared pool, before per-user loop.
        # Shuffle first (seeded on run_id, so a given run is reproducible for debugging)
        # so the cap draws roughly proportionally across sources instead of always
        # favoring whichever source was scraped first (fixed scrape order above).
        # -------------------------------------------------------------------------
        if limit is not None:
            random.Random(run_id).shuffle(fresh)
            fresh = fresh[:limit]
            log.info("--limit %d: capped shared pool to %d posting(s)", limit, len(fresh))

        counts.shared_pool_size = len(fresh)

        # -------------------------------------------------------------------------
        # 4. Per-user loop: skip-set → gate → Tier 1 → Tier 2 → matches/seen writes
        # -------------------------------------------------------------------------
        user_results: list[UserRunResult] = []
        parse_failures: list[ParseFailure] = []

        for user in users:
            seen_map = load_seen_urls(client, user.user_id)
            existing_match_urls = fetch_existing_match_urls(client, user.user_id)
            skip_set = set(seen_map.keys()) | existing_match_urls

            user_postings = [p for p in fresh if canonical_url(p.url) not in skip_set]
            result = UserRunResult(user_id=user.user_id, new_after_dedup=len(user_postings))
            new_matches: list[dict] = []  # matches actually inserted THIS run, for the opt-in digest email
            pending_verdicts: list[dict] = []  # queued seen-verdict payloads; flushed once below, not per posting

            threshold     = user.rubric["threshold"]
            near_miss_min = user.rubric["near_miss_min"]

            # -- Hard gate --
            gate_survivors: list[tuple] = []
            for p in user_postings:
                g = gate_mod.check(p, user.profile)
                if g.passed:
                    gate_survivors.append((p, g))
                else:
                    result.gated_out += 1
                    record_verdict(pending_verdicts, seen_map, user.user_id, p.url, "gated_out", None)

            # -- Tier 1 scoring --
            insert_candidates = []
            for posting, g in gate_survivors:
                try:
                    scored = scoring_mod.score(posting, g, user.profile, user.rubric)
                    result.scored += 1
                except ScoringError as exc:
                    log.error("Tier 1 failed for user %s, %s: %s", user.user_id, posting.url, exc)
                    result.parse_failures += 1
                    parse_failures.append(ParseFailure(user_id=user.user_id, url=posting.url))
                    continue  # don't cache parse failures — retry next run

                if scored.fit_score >= threshold:
                    insert_candidates.append(scored)
                elif scored.fit_score >= near_miss_min:
                    result.near_misses += 1
                    record_verdict(
                        pending_verdicts, seen_map, user.user_id, posting.url,
                        "below_threshold", scored.fit_score,
                    )
                else:
                    record_verdict(
                        pending_verdicts, seen_map, user.user_id, posting.url,
                        "below_threshold", scored.fit_score,
                    )

            # -- Tier 2 enrichment + matches upsert --
            for scored in insert_candidates:
                try:
                    enrichment = enrich(scored, user.profile)
                except EnrichmentError as exc:
                    log.error("Tier 2 failed for user %s, %s: %s", user.user_id, scored.posting.url, exc)
                    result.parse_failures += 1
                    parse_failures.append(ParseFailure(user_id=user.user_id, url=scored.posting.url))
                    continue  # don't cache Tier 2 failures — retry next run

                row = MatchRow(
                    title=scored.posting.title,
                    org=scored.posting.org,
                    org_summary=enrichment.org_summary,
                    source=scored.posting.source,
                    url=scored.posting.url,
                    canonical_url=canonical_url(scored.posting.url),
                    date_found=today,
                    deadline=scored.posting.deadline,
                    comp=scored.posting.comp,
                    location=scored.posting.location,
                    seniority=scored.posting.seniority,
                    cause_area=scored.posting.cause_area,
                    fit_score=scored.fit_score,
                    dimension_scores={
                        dim: getattr(scored.scores, dim).score
                        for dim in type(scored.scores).model_fields
                    },
                    why_fits=enrichment.why_fits,
                    why_not_fits=enrichment.why_not_fits,
                    emphasize_in_cv=enrichment.emphasize_in_cv,
                    deemphasize=enrichment.deemphasize,
                )

                upsert_match(client, user.user_id, row, dry_run=dry_run)
                record_verdict(
                    pending_verdicts, seen_map, user.user_id, scored.posting.url,
                    "inserted", scored.fit_score,
                )
                result.inserted += 1
                new_matches.append({
                    "title": row.title,
                    "org": row.org,
                    "location": row.location,
                    "fit_score": row.fit_score,
                    "why_fits": row.why_fits,
                    "url": row.url,
                })

            # -- Flush this user's queued seen-verdicts in one (or a few chunked) batch --
            upsert_verdicts(client, pending_verdicts, dry_run=dry_run)

            log.info(
                "User %s: new=%d gated_out=%d scored=%d inserted=%d near_miss=%d parse_fail=%d",
                user.user_id, result.new_after_dedup, result.gated_out,
                result.scored, result.inserted, result.near_misses, result.parse_failures,
            )
            user_results.append(result)

            # -- Opt-in match-digest email. Best-effort: must never fail the run,
            # flip search_quota, or interrupt other users — notify.send_match_digest
            # itself never raises, and this try/except is a second layer of defense.
            if user.profile.get("email_on_match") and new_matches and not dry_run:
                try:
                    send_match_digest(user.user_id, new_matches)
                except Exception as exc:
                    log.warning("Match-digest email failed for user %s: %s", user.user_id, exc)

        # -------------------------------------------------------------------------
        # 5. Emit RunLog
        # -------------------------------------------------------------------------
        run_log = RunLog(
            run_id=run_id,
            timestamp=started_at,
            model=f"tier1={tier1_model},tier2={tier2_model}",
            temperature=0.0,
            counts=counts,
            source_results=source_results,
            user_results=user_results,
            parse_failures=parse_failures,
        )

        log.info(
            "Run %s complete | users=%d scraped=%d shared_pool=%d total_inserted=%d",
            run_id, len(users), counts.scraped, counts.shared_pool_size,
            sum(r.inserted for r in user_results),
        )

        succeeded = True
        return run_log

    except Exception:
        if track_quota:
            set_status(client, only_user_id, "failed")
        raise
    finally:
        if track_quota and succeeded:
            set_status(client, only_user_id, "complete", completed=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="EA job-fit pipeline (multi-user)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip seen/matches writes; print would-be matches instead",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Cap the shared postings pool after recency/staleness filtering (global cap)",
    )
    parser.add_argument(
        "--only-user-id", type=str, default=None, metavar="ID",
        help="Scope the run to a single user (one-off request); writes status to search_quota",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    result = run(dry_run=args.dry_run, limit=args.limit, only_user_id=args.only_user_id)
    print(result.model_dump_json(indent=2))
