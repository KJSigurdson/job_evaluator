"""Top-level orchestration. Ties all stages together; writes RunLog.

Usage:
    python -m src.pipeline [--dry-run] [--limit N]

--dry-run   runs everything except Notion inserts and seen-cache writes.
--limit N   caps the number of postings processed after dedup (cheap testing).
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from src import gate as gate_mod
from src import scoring as scoring_mod
from src.dedup import canonical_url, is_duplicate
from src.enrich import EnrichmentError, enrich
from src.notion_client import fetch_existing_urls
from src.notion_client import insert_row as notion_insert_row
from src.schemas import (
    NearMiss,
    NotionInsertRow,
    RunCounts,
    RunLog,
    SourceResult,
)
from src.scoring import ScoringError
from src.seen_cache import is_seen, load_seen, prune, record, save_seen
from src.sources import eightyk, iap, probablygood

log = logging.getLogger(__name__)

_ROOT                = Path(__file__).parent.parent
_SEEN_CACHE_PATH     = _ROOT / "state" / "seen.json"
_TIER1_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
_TIER2_MODEL_DEFAULT = "claude-sonnet-4-6"
_RECENCY_DAYS_DEFAULT = 14


def _load_yaml(name: str) -> dict:
    with open(_ROOT / name) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(*, dry_run: bool = False, limit: int | None = None) -> RunLog:
    """Run the full pipeline. Returns a RunLog regardless of per-source failures.

    dry_run=True  — skips Notion writes and seen-cache writes; no NOTION_TOKEN required.
    limit=N       — caps postings processed after dedup (for cheap testing).
    """
    run_id     = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc)
    today      = date.today()

    # --- Config ---
    profile       = _load_yaml("profile.yaml")
    rubric        = _load_yaml("rubric.yaml")
    threshold     = rubric["threshold"]
    near_miss_min = rubric["near_miss_min"]
    recency_days  = int(os.environ.get("RECENCY_DAYS") or _RECENCY_DAYS_DEFAULT)
    tier1_model   = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT
    tier2_model   = os.environ.get("TIER2_MODEL") or _TIER2_MODEL_DEFAULT

    # --- Seen-URL cache (read always; written only when not dry-run) ---
    seen_data = load_seen(_SEEN_CACHE_PATH)
    seen_url_keys: set[str] = set(seen_data["entries"].keys())
    log.info("Seen cache: %d entries loaded from %s", len(seen_url_keys), _SEEN_CACHE_PATH)

    # --- Notion client (skipped entirely in dry-run) ---
    if dry_run:
        notion        = None
        db_id         = ""
        existing_urls: set[str] = set()
        log.info("DRY RUN — Notion reads/writes disabled; dedup set is empty")
    else:
        from notion_client import Client
        notion        = Client(auth=os.environ["NOTION_TOKEN"])
        db_id         = os.environ["NOTION_DB_ID"]
        existing_urls = fetch_existing_urls(notion, db_id)
        log.info("Loaded %d existing URLs from Notion", len(existing_urls))

    # Skip set = Notion URLs ∪ seen-cache URLs
    skip_set = existing_urls | seen_url_keys

    counts: RunCounts = RunCounts()
    source_results: list[SourceResult] = []
    all_postings = []

    # -------------------------------------------------------------------------
    # 1. Scrape sources — each in its own try/except
    # -------------------------------------------------------------------------
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
    # 2. Recency filter — postings with posted_at=None are exempt (e.g. IAP)
    # -------------------------------------------------------------------------
    fresh: list = []
    recency_dropped = 0
    for p in all_postings:
        if p.posted_at is None or (today - p.posted_at).days <= recency_days:
            fresh.append(p)
        else:
            recency_dropped += 1
    if recency_dropped:
        log.info(
            "Recency filter: dropped %d posting(s) older than %d days",
            recency_dropped, recency_days,
        )

    # -------------------------------------------------------------------------
    # 3. Dedup against Notion state + seen cache (combined skip set)
    # -------------------------------------------------------------------------
    new_postings = [p for p in fresh if not is_duplicate(p.url, skip_set)]
    counts.new_after_dedup = len(new_postings)
    log.info(
        "Dedup: %d fresh → %d new (skipped %d via Notion+cache)",
        len(fresh), counts.new_after_dedup, len(fresh) - counts.new_after_dedup,
    )

    # -------------------------------------------------------------------------
    # 4. Apply --limit (after dedup so counts are meaningful)
    # -------------------------------------------------------------------------
    if limit is not None:
        new_postings = new_postings[:limit]
        log.info("--limit %d: capped to %d posting(s)", limit, len(new_postings))

    # -------------------------------------------------------------------------
    # 5. Hard gate — binary location + seniority; no LLM cost
    # -------------------------------------------------------------------------
    gate_survivors: list[tuple] = []
    for p in new_postings:
        g = gate_mod.check(p, profile)
        if g.passed:
            gate_survivors.append((p, g))
        else:
            counts.gated_out += 1
            record(seen_data, p.url, "gated_out", None)
            log.debug("GATE FAIL [%s] %s: %s", p.source, p.url, g.reason)

    log.info(
        "Gate: %d passed, %d rejected",
        len(gate_survivors), counts.gated_out,
    )

    # -------------------------------------------------------------------------
    # 6. Tier 1 scoring — cheap LLM, all gate survivors
    # -------------------------------------------------------------------------
    near_misses:        list[NearMiss] = []
    parse_failure_urls: list[str]      = []
    insert_candidates                  = []

    for posting, gate in gate_survivors:
        try:
            result = scoring_mod.score(posting, gate, profile, rubric)
            counts.scored += 1
        except ScoringError as exc:
            log.error("Tier 1 failed for %s: %s", posting.url, exc)
            counts.parse_failures += 1
            parse_failure_urls.append(posting.url)
            continue  # don't cache parse failures — retry next run

        if result.fit_score >= threshold:
            insert_candidates.append(result)
            log.info(
                "FIT  (%.3f) [%s] %s @ %s",
                result.fit_score, posting.source, posting.title, posting.org,
            )
        elif result.fit_score >= near_miss_min:
            record(seen_data, posting.url, "below_threshold", result.fit_score)
            near_misses.append(NearMiss(
                posting=result.posting,
                scores=result.scores,
                fit_score=result.fit_score,
            ))
            counts.near_misses += 1
            log.info(
                "NEAR MISS (%.3f) [%s] %s @ %s",
                result.fit_score, posting.source, posting.title, posting.org,
            )
        else:
            record(seen_data, posting.url, "below_threshold", result.fit_score)

    log.info(
        "Tier 1: %d scored → %d above threshold, %d near-miss, %d parse failures",
        counts.scored, len(insert_candidates), counts.near_misses, counts.parse_failures,
    )

    # -------------------------------------------------------------------------
    # 7. Tier 2 enrichment + Notion insert (or dry-run print)
    # -------------------------------------------------------------------------
    for result in insert_candidates:
        try:
            enrichment = enrich(result, profile)
        except EnrichmentError as exc:
            log.error("Tier 2 failed for %s: %s", result.posting.url, exc)
            counts.parse_failures += 1
            parse_failure_urls.append(result.posting.url)
            continue  # don't cache Tier 2 failures — retry next run

        row = NotionInsertRow(
            role=result.posting.title,
            org=result.posting.org,
            org_summary=enrichment.org_summary,
            source=result.posting.source,
            url=result.posting.url,
            date_found=today,
            deadline=result.posting.deadline,
            comp=result.posting.comp,
            fit_score=result.fit_score,
            cause_mission_fit=result.scores.cause_mission_fit.score,
            role_function_fit=result.scores.role_function_fit.score,
            location_compatibility=result.scores.location_compatibility.score,
            seniority_match=result.scores.seniority_match.score,
            comp_adequacy=result.scores.comp_adequacy.score,
            values_alignment=result.scores.values_alignment.score,
            skill_growth=result.scores.skill_growth.score,
            why_fits=enrichment.why_fits,
            why_not_fits=enrichment.why_not_fits,
            emphasize_in_cv="\n".join(enrichment.emphasize_in_cv),
            deemphasize="\n".join(enrichment.deemphasize),
        )

        if dry_run:
            print(
                f"DRY RUN — would insert:\n"
                f"  {row.role} @ {row.org}  fit={row.fit_score:.3f}\n"
                f"  url: {row.url}\n"
                f"  org_summary: {row.org_summary[:120]}\n"
            )
            counts.inserted += 1
            # Don't cache in dry-run — nothing was actually committed
        else:
            try:
                notion_insert_row(notion, db_id, row)
                existing_urls.add(canonical_url(row.url))  # prevent same-run re-insert
                record(seen_data, row.url, "inserted", result.fit_score)
                counts.inserted += 1
                log.info(
                    "INSERTED: %s @ %s  fit=%.3f",
                    row.role, row.org, row.fit_score,
                )
            except RuntimeError as exc:
                log.warning("Insert skipped (second-line dedup): %s", exc)

    # -------------------------------------------------------------------------
    # 8. Persist seen cache (skip in dry-run)
    # -------------------------------------------------------------------------
    if not dry_run:
        seen_data = prune(seen_data)
        save_seen(_SEEN_CACHE_PATH, seen_data)
        log.info(
            "Seen cache: saved %d entries to %s",
            len(seen_data["entries"]), _SEEN_CACHE_PATH,
        )

    # -------------------------------------------------------------------------
    # 9. Emit RunLog
    # -------------------------------------------------------------------------
    run_log = RunLog(
        run_id=run_id,
        timestamp=started_at,
        model=f"tier1={tier1_model},tier2={tier2_model}",
        temperature=0.0,
        counts=counts,
        source_results=source_results,
        near_misses=near_misses,
        parse_failure_urls=parse_failure_urls,
    )

    log.info(
        "Run %s complete | scraped=%d new=%d gated_out=%d scored=%d "
        "inserted=%d near_miss=%d parse_fail=%d",
        run_id,
        counts.scraped, counts.new_after_dedup, counts.gated_out,
        counts.scored, counts.inserted, counts.near_misses, counts.parse_failures,
    )

    return run_log


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

    parser = argparse.ArgumentParser(description="EA job-fit pipeline")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip Notion inserts and seen-cache writes; print would-be rows",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Cap postings processed after dedup (cheap testing)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    result = run(dry_run=args.dry_run, limit=args.limit)
    print(result.model_dump_json(indent=2))
