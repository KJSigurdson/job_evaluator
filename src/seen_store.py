"""Per-user seen-cache backed by the Supabase `seen` table.

Replaces the old JSON file at state/seen.json. Schema: (user_id, canonical_url) unique,
verdict in (gated_out, below_threshold, inserted), fit_score, first_seen.

Prevents re-scoring sub-threshold and gated-out roles *per user*. Parse failures are
never recorded here — they should be retried on the next run, not permanently skipped.

first_seen is also the anchor for the 14-day staleness cutoff applied to postings that
have neither a deadline nor a posted_at date (see fetch_global_first_seen / pipeline.py).

Writes are batched: record_verdict() only queues a payload dict in memory (no network
call), and upsert_verdicts() flushes everything queued for a user in one call — chunked
at _CHUNK_SIZE rows — instead of one Supabase round-trip per posting. With an uncapped
shared pool (thousands of postings) across several users, per-posting writes were the
pipeline's dominant cost and blew the workflow's time budget; batching turns that into
a handful of round-trips per user regardless of pool size.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.dedup import canonical_url

log = logging.getLogger(__name__)

_CHUNK_SIZE = 500


def load_seen_urls(client, user_id: str) -> dict[str, dict]:
    """Return {canonical_url: row} for every seen entry belonging to *user_id*."""
    resp = client.table("seen").select("*").eq("user_id", user_id).execute()
    return {row["canonical_url"]: row for row in (resp.data or [])}


def record_verdict(
    pending: list[dict],
    seen_map: dict[str, dict],
    user_id: str,
    url: str,
    verdict: str,
    fit_score: float | None,
) -> None:
    """Queue the terminal verdict for *url* for this user — no network call here.

    Appends the upsert payload to *pending* (flush with upsert_verdicts once the
    caller is done accumulating, typically once per user). first_seen is preserved
    across runs exactly as before: only included in the payload the first time a URL
    is recorded for this user (per the in-memory *seen_map*, loaded once at the start
    of that user's loop). Mutates *seen_map* so a URL touched twice in the same run
    still gets first_seen preserved correctly, though in practice each posting is
    visited at most once per user per run.
    """
    key = canonical_url(url)
    payload = {"user_id": user_id, "canonical_url": key, "verdict": verdict, "fit_score": fit_score}
    if key not in seen_map:
        payload["first_seen"] = datetime.now(timezone.utc).isoformat()

    pending.append(payload)
    seen_map[key] = {**seen_map.get(key, {}), **payload}


def upsert_verdicts(client, pending: list[dict], *, dry_run: bool = False) -> None:
    """Upsert every queued verdict payload in *pending*, chunked at _CHUNK_SIZE rows
    per call to stay under payload/row limits. On_conflict is (user_id, canonical_url) —
    identical to the old per-item upsert, just batched.

    dry_run logs the count that would have been written and skips the network call
    entirely (no chunking, no upsert calls of any kind).
    """
    if not pending:
        return

    if dry_run:
        log.info("DRY RUN — would upsert %d seen verdict(s)", len(pending))
        return

    for i in range(0, len(pending), _CHUNK_SIZE):
        chunk = pending[i:i + _CHUNK_SIZE]
        client.table("seen").upsert(chunk, on_conflict="user_id,canonical_url").execute()

    batches = -(-len(pending) // _CHUNK_SIZE)  # ceil division
    log.info("Upserted %d seen verdict(s) in %d batch(es)", len(pending), batches)


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
