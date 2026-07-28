"""Supabase `matches` integration. Insert-only from the app's perspective: this module
only ever writes job fields + model output (see MatchRow) — it has no way to touch
status/user_notes/discarded, which are user-owned columns the app manages.

On a re-encounter of an existing match, the pipeline's per-user skip-set (built from
fetch_existing_match_urls, unioned with the seen-table skip-set) means the posting is
filtered out before it ever reaches upsert_match again — so tracking columns are never
at risk of being clobbered.
"""
from __future__ import annotations

import logging

from src.schemas import MatchRow

log = logging.getLogger(__name__)


def fetch_existing_match_urls(client, user_id: str) -> set[str]:
    """Return canonical URLs already in `matches` for this user (dedup source)."""
    resp = client.table("matches").select("canonical_url").eq("user_id", user_id).execute()
    return {row["canonical_url"] for row in (resp.data or [])}


def upsert_match(client, user_id: str, row: MatchRow, *, dry_run: bool = False) -> None:
    """Upsert *row* for *user_id* on (user_id, canonical_url)."""
    payload = {"user_id": user_id, **row.model_dump(mode="json")}

    if dry_run:
        log.info(
            "DRY RUN — would upsert match: %s @ %s  fit=%.3f  url=%s",
            row.title, row.org, row.fit_score, row.url,
        )
        return

    client.table("matches").upsert(payload, on_conflict="user_id,canonical_url").execute()
    log.info("MATCH: %s @ %s  fit=%.3f  url=%s", row.title, row.org, row.fit_score, row.url)


def update_enrichment(
    client,
    user_id: str,
    canonical_url: str,
    *,
    why_fits: str,
    why_not_fits: str,
    emphasize_in_cv: list[str],
    deemphasize: list[str],
    dry_run: bool = False,
) -> None:
    """Targeted update of ONLY why_fits/why_not_fits/emphasize_in_cv/deemphasize on an
    existing match row, scoped to (user_id, canonical_url). Used by the one-off
    src/reformat_match_enrichment.py migration to rewrite enrichment text on rows that
    already exist — never fit_score/dimension_scores/org_summary/status/user_notes/
    discarded/anything else. Uses .update() (not .upsert()): a non-matching row is a
    no-op rather than creating a new, incomplete row the way an upsert would.
    """
    payload = {
        "why_fits": why_fits,
        "why_not_fits": why_not_fits,
        "emphasize_in_cv": emphasize_in_cv,
        "deemphasize": deemphasize,
    }

    if dry_run:
        log.info(
            "DRY RUN — would update enrichment for user=%s url=%s", user_id, canonical_url,
        )
        return

    (
        client.table("matches")
        .update(payload)
        .eq("user_id", user_id)
        .eq("canonical_url", canonical_url)
        .execute()
    )
    log.info("Updated enrichment for user=%s url=%s", user_id, canonical_url)
