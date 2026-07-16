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
