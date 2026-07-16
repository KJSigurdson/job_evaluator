"""Per-user seen-cache backed by the Supabase `seen` table.

Replaces the old JSON file at state/seen.json. Schema: (user_id, canonical_url) unique,
verdict in (gated_out, below_threshold, inserted), fit_score, first_seen.

Prevents re-scoring sub-threshold and gated-out roles *per user*. Parse failures are
never recorded here — they should be retried on the next run, not permanently skipped.

first_seen is also the anchor for the 14-day staleness cutoff applied to postings that
have neither a deadline nor a posted_at date (see fetch_global_first_seen / pipeline.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.dedup import canonical_url

log = logging.getLogger(__name__)


def load_seen_urls(client, user_id: str) -> dict[str, dict]:
    """Return {canonical_url: row} for every seen entry belonging to *user_id*."""
    resp = client.table("seen").select("*").eq("user_id", user_id).execute()
    return {row["canonical_url"]: row for row in (resp.data or [])}


def record_verdict(
    client,
    seen_map: dict[str, dict],
    user_id: str,
    url: str,
    verdict: str,
    fit_score: float | None,
    *,
    dry_run: bool = False,
) -> None:
    """Record (or update) the terminal verdict for *url* for this user.

    first_seen is preserved across updates: only included in the upsert payload the
    first time a URL is recorded for this user (per the in-memory *seen_map*, loaded
    once at the start of that user's loop). Mutates *seen_map* so subsequent calls in
    the same run see the update.
    """
    key = canonical_url(url)

    if dry_run:
        log.info("DRY RUN — would record verdict %s (%s) for %s", verdict, fit_score, key)
        return

    payload = {"user_id": user_id, "canonical_url": key, "verdict": verdict, "fit_score": fit_score}
    if key not in seen_map:
        payload["first_seen"] = datetime.now(timezone.utc).isoformat()

    client.table("seen").upsert(payload, on_conflict="user_id,canonical_url").execute()
    seen_map[key] = {**seen_map.get(key, {}), **payload}


def fetch_global_first_seen(client) -> dict[str, datetime]:
    """Return {canonical_url: earliest first_seen across ALL users}.

    Powers the shared 14-day staleness cutoff for deadline-less, posted_at-less
    postings (e.g. IAP rows) — that filter runs once, before the per-user loop, so it
    needs a user-independent signal for "how long has this posting existed in the
    system," not any single user's first_seen.
    """
    resp = client.table("seen").select("canonical_url,first_seen").execute()
    out: dict[str, datetime] = {}
    for row in (resp.data or []):
        ts = _parse_ts(row["first_seen"])
        if ts is None:
            continue
        cu = row["canonical_url"]
        if cu not in out or ts < out[cu]:
            out[cu] = ts
    return out


def _parse_ts(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None
