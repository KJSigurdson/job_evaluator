"""Opt-in match-digest email notification. POSTs to a Supabase Edge Function.

This is a best-effort side channel, not part of the pipeline's success/failure
contract: send_match_digest() NEVER raises. A missing config, network failure, or
non-2xx response is logged and swallowed so an email problem can never fail a run,
flip search_quota to 'failed', or interrupt processing of other users.
"""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10


def send_match_digest(user_id: str, matches: list[dict]) -> None:
    """POST {user_id, matches} to MATCH_DIGEST_URL with the shared-secret header.

    Never raises. If MATCH_DIGEST_URL or PIPELINE_SHARED_SECRET is unset, skips
    sending and logs at info (so local/dev runs without email configured still work).
    Any request exception or non-2xx response is logged at warning and swallowed.
    """
    url = os.environ.get("MATCH_DIGEST_URL")
    secret = os.environ.get("PIPELINE_SHARED_SECRET")

    if not url or not secret:
        log.info(
            "MATCH_DIGEST_URL/PIPELINE_SHARED_SECRET not configured — "
            "skipping match-digest email for user %s",
            user_id,
        )
        return

    try:
        response = httpx.post(
            url,
            json={"user_id": user_id, "matches": matches},
            headers={"x-pipeline-secret": secret},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as exc:
        log.warning("Match-digest email failed for user %s: %s", user_id, exc)
        return

    log.info("Sent match digest for user %s (%d match(es))", user_id, len(matches))
