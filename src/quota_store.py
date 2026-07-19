"""Status write-back for one-off single-user runs, backed by the Supabase
`search_quota` table: (user_id pk, status, requested_at, completed_at).

Only a one-off run (pipeline.run(only_user_id=...), not dry-run) writes here — the
daily multi-user cron never touches this table.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_VALID_STATUSES = frozenset({"not_started", "queued", "running", "complete", "failed"})


def set_status(client, user_id: str, status: str, *, completed: bool = False) -> None:
    """Upsert *status* for *user_id* in search_quota.

    completed=True also sets completed_at=now() — used for the terminal 'complete'
    status once a one-off run has finished (zero matches written is still 'complete').
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid search_quota status: {status!r}")

    payload = {"user_id": user_id, "status": status}
    if completed:
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()

    client.table("search_quota").upsert(payload, on_conflict="user_id").execute()
    log.info("search_quota: user %s -> %s", user_id, status)
