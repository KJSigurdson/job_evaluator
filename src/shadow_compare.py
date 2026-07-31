"""Shadow-mode gate comparison: for every real user x posting pair, run v1's
gate.check() (against v1's own live 80,000 Hours scrape) and v2's gate_v2.check()
(against the persisted, enriched v2 `jobs` row for the SAME posting, matched by
canonical_url) and report where the two gates' pass/fail verdicts diverge.

Read-only: fetches users/jobs, runs both gates in memory, prints a report. Writes
nothing to matches/seen/search_quota/jobs/job_aliases/run_metrics. Standalone —
not called from src/pipeline.py or src/ingest.py/enrich_jobs.py.

Run:
    python -m src.shadow_compare

Only postings v1's CURRENT scrape and v2's persisted `jobs` table both have (by
canonical_url) are compared — a v2 job with no v1 match today has likely expired
off the live board since it was ingested; that's expected, not an error, and is
reported as a count, not treated as a divergence.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from src.gate import check as gate_v1_check
from src.gate_v2 import check as gate_v2_check
from src.job_normalize import normalize_job_url
from src.sources import eightyk
from src.supabase_client import fetch_all, get_client
from src.user_store import fetch_users

log = logging.getLogger(__name__)


@dataclass
class Divergence:
    user_id: str
    title: str
    org: str
    canonical_url: str
    v1_passed: bool
    v2_passed: bool
    v1_reason: str | None
    v2_reason: str | None


def run(client=None) -> dict:
    client = client or get_client()

    users, excluded = fetch_users(client)
    if excluded:
        log.warning(
            "fetch_users excluded %d user(s) with malformed gate config: %s",
            len(excluded), [e.user_id for e in excluded],
        )
    if not users:
        log.info("No users with profile+weights — nothing to compare")
        return {"users": 0, "postings_matched": 0, "pairs_compared": 0, "divergences": [],
                "v1_passed_v2_failed": 0, "v1_failed_v2_passed": 0}

    log.info("Fetching v1's own 80,000 Hours scrape for a like-for-like comparison...")
    v1_postings = eightyk.fetch()
    v1_by_url = {normalize_job_url(p.url): p for p in v1_postings}

    log.info("Fetching v2's enriched jobs pool...")
    v2_rows = fetch_all(
        client.table("jobs")
        .select("id,title,org,canonical_url,location_field,enrichment")
        .eq("status", "enriched")
        .eq("source_board", "80000_hours")
    )

    matched = [(v1_by_url[row["canonical_url"]], row) for row in v2_rows if row["canonical_url"] in v1_by_url]
    log.info(
        "%d v2 job(s), %d matched to a posting in v1's current scrape (%d unmatched, likely expired "
        "off the live board since ingest)",
        len(v2_rows), len(matched), len(v2_rows) - len(matched),
    )

    divergences: list[Divergence] = []
    pairs_compared = 0
    direction_counts: Counter = Counter()

    for user in users:
        for v1_posting, v2_row in matched:
            pairs_compared += 1
            v1_result = gate_v1_check(v1_posting, user.profile)
            v2_result = gate_v2_check(v2_row, user.profile)
            if v1_result.passed != v2_result.passed:
                direction_counts[(v1_result.passed, v2_result.passed)] += 1
                divergences.append(Divergence(
                    user_id=user.user_id, title=v2_row["title"], org=v2_row["org"],
                    canonical_url=v2_row["canonical_url"],
                    v1_passed=v1_result.passed, v2_passed=v2_result.passed,
                    v1_reason=v1_result.reason, v2_reason=v2_result.reason,
                ))

    log.info(
        "Compared %d user x posting pair(s) across %d user(s) and %d posting(s): %d divergence(s) (%.1f%%)",
        pairs_compared, len(users), len(matched), len(divergences),
        (len(divergences) / pairs_compared * 100) if pairs_compared else 0.0,
    )

    return {
        "users": len(users),
        "postings_matched": len(matched),
        "pairs_compared": pairs_compared,
        "divergences": divergences,
        "v1_passed_v2_failed": direction_counts[(True, False)],
        "v1_failed_v2_passed": direction_counts[(False, True)],
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    result = run()

    print(f"\n{'=' * 70}")
    print(f"Users compared:                 {result['users']}")
    print(f"Postings matched (v1 <-> v2):   {result['postings_matched']}")
    print(f"User x posting pairs compared:  {result['pairs_compared']}")
    print(f"Divergences:                    {len(result['divergences'])}")
    print(f"  v1 passed, v2 failed:         {result['v1_passed_v2_failed']}")
    print(f"  v1 failed, v2 passed:         {result['v1_failed_v2_passed']}")

    if result["divergences"]:
        print(f"\n{'=' * 70}")
        print("Sample divergences (up to 15):")
        for d in result["divergences"][:15]:
            print(f"\n  user={d.user_id} | {d.title} @ {d.org}")
            print(f"    url: {d.canonical_url}")
            print(f"    v1: passed={d.v1_passed} reason={d.v1_reason}")
            print(f"    v2: passed={d.v2_passed} reason={d.v2_reason}")
