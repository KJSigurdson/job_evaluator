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


def send_match_digest(user_id: str, matches: list[dict]) -> bool:
    """POST {user_id, matches} to MATCH_DIGEST_URL with the shared-secret header.

    Never raises. Returns True only on a confirmed 2xx response — False for
    missing config, a request/network exception, or a non-2xx response.
    Callers that only logged-and-moved-on previously (pipeline_v2.py set
    notified=True unconditionally right after calling this) must switch to
    checking this return value — confirmed for real (2026-11-03) that a run
    reported notified=True while MATCH_DIGEST_URL was actually 401ing the
    whole time, because nothing upstream of this change ever looked at
    whether the POST actually succeeded.

    If MATCH_DIGEST_URL or PIPELINE_SHARED_SECRET is unset, skips sending and
    logs at info (so local/dev runs without email configured still work). A
    non-2xx response logs the status code and response body at warning (not
    just the exception's str, which for httpx.HTTPStatusError omits the
    body) — needed because a 401 gives no other clue (API secret mismatch?
    Edge Function's own JWT verification rejecting the request first?).
    """
    url = os.environ.get("MATCH_DIGEST_URL")
    secret = os.environ.get("PIPELINE_SHARED_SECRET")

    if not url or not secret:
        log.info(
            "MATCH_DIGEST_URL/PIPELINE_SHARED_SECRET not configured — "
            "skipping match-digest email for user %s",
            user_id,
        )
        return False

    try:
        response = httpx.post(
            url,
            json={"user_id": user_id, "matches": matches},
            headers={"x-pipeline-secret": secret},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        log.warning(
            "Match-digest email failed for user %s: HTTP %s - %s",
            user_id, resp.status_code if resp is not None else "?",
            resp.text if resp is not None else "<no response>",
        )
        return False
    except Exception as exc:
        log.warning("Match-digest email failed for user %s: %s", user_id, exc)
        return False

    log.info("Sent match digest for user %s (%d match(es))", user_id, len(matches))
    return True
