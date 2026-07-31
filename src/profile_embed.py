"""True North v2 profile embeddings: builds a semantic-similarity embedding text
from a user's profile (career_goals, cause_priorities, experience, skills,
values_notes — deliberately NOT location/comp_needs, which are hard constraints
gate_v2.py already enforces, not soft-similarity signal), embeds it via Voyage AI
(reusing src.enrich_jobs.embed_texts/embeddings_available), and writes
profiles.profile_embedding.

Re-embed-on-change: profiles.profile_content_hash (sha256 of the rendered
embedding text, via job_normalize.content_hash) is compared before embedding —
a user whose rendered text hasn't changed since their last embed is skipped, so
re-running this is cheap and idempotent.

Additive: only ever writes profile_embedding/profile_content_hash on `profiles`.
Nothing here touches hard_constraints/scoring_weights or any v1 table.
"""
from __future__ import annotations

import logging

from src.enrich_jobs import embed_texts, embeddings_available
from src.job_normalize import content_hash
from src.supabase_client import chunked_upsert, fetch_all
from src.user_store import fetch_users

log = logging.getLogger(__name__)

_SECTIONS = [
    ("Career goals", "career_goals"),
    ("Cause priorities", "cause_priorities"),
    ("Experience", "experience_inventory"),
    ("Skills", "skills"),
    ("Values", "values_notes"),
]


def build_embedding_text(profile: dict) -> str:
    """profile: the dict shape user_store._build_profile() produces (i.e. a
    UserContext.profile) — reuses its already-resolved experience_inventory
    (structured experiences rows preferred, free-text profiles.experience
    fallback) rather than re-deriving it from raw columns."""
    parts = []
    for label, key in _SECTIONS:
        value = profile.get(key)
        if isinstance(value, str):
            value = value.strip()
        if not value:
            continue
        parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


def backfill_profile_embeddings(client, *, only_user_id: str | None = None) -> dict:
    """Embed every user (with a profile + scoring_weights row, per
    user_store.fetch_users — a user without weights can't be matched anyway)
    whose rendered embedding text is non-empty and has changed since their
    stored profile_content_hash. only_user_id scopes this to a single user —
    used by pipeline_v2.py's manual DB-only path, where refreshing just that
    one user's embedding (a no-op if their hash hasn't changed) is cheap
    enough to always do, unlike a full backfill pass. Returns
    {selected, embedded, skipped_unchanged, skipped_empty}.
    """
    if not embeddings_available():
        log.warning("VOYAGE_API_KEY not set — backfill_profile_embeddings is a no-op")
        return {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0}

    users, excluded = fetch_users(client, only_user_id=only_user_id)
    if excluded:
        log.warning(
            "fetch_users excluded %d user(s) with malformed gate config: %s",
            len(excluded), [e.user_id for e in excluded],
        )
    if not users:
        return {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0}

    existing_hashes = {
        row["user_id"]: row.get("profile_content_hash")
        for row in fetch_all(client.table("profiles").select("user_id,profile_content_hash"))
    }

    to_embed: list[tuple[str, str, str]] = []  # (user_id, text, hash)
    skipped_unchanged = 0
    skipped_empty = 0

    for user in users:
        text = build_embedding_text(user.profile)
        if not text:
            skipped_empty += 1
            continue
        text_hash = content_hash(text)
        if existing_hashes.get(user.user_id) == text_hash:
            skipped_unchanged += 1
            continue
        to_embed.append((user.user_id, text, text_hash))

    if not to_embed:
        log.info(
            "backfill_profile_embeddings: nothing to embed (%d unchanged, %d empty)",
            skipped_unchanged, skipped_empty,
        )
        return {"selected": len(users), "embedded": 0,
                "skipped_unchanged": skipped_unchanged, "skipped_empty": skipped_empty}

    vectors = embed_texts([text for _, text, _ in to_embed])
    updates = [
        {"user_id": user_id, "profile_embedding": vector, "profile_content_hash": text_hash}
        for (user_id, _, text_hash), vector in zip(to_embed, vectors)
    ]
    chunked_upsert(client.table("profiles"), updates, on_conflict="user_id")

    log.info(
        "backfill_profile_embeddings: %d selected, %d embedded, %d unchanged, %d empty",
        len(users), len(updates), skipped_unchanged, skipped_empty,
    )
    return {"selected": len(users), "embedded": len(updates),
            "skipped_unchanged": skipped_unchanged, "skipped_empty": skipped_empty}


if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    from src.supabase_client import get_client

    result = backfill_profile_embeddings(get_client())
    print(json.dumps(result, indent=2))
