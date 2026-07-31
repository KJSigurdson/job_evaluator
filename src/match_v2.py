"""True North v2 per-user matching: gate_v2 hard-filter survivors, then a
pgvector cosine-similarity rerank to the top N candidates. Shadow mode: reads
jobs/profiles only, writes nothing to matches/seen/search_quota/pipeline_runs
or any other v1 table. Not called from src/pipeline.py.

The similarity rerank runs entirely in SQL via the match_v2_top_jobs() Postgres
function (see supabase/migrations/<timestamp>_match_v2_top_jobs.sql) — pgvector's
<=> cosine-distance operator over the *already gate-filtered* candidate id set,
not a Python-side computation. `1 - (embedding <=> query_embedding)` is the
standard cosine-similarity idiom for pgvector's <=> operator regardless of
whether the stored vectors are pre-normalized.
"""
from __future__ import annotations

import logging

from src.gate_v2 import check as gate_v2_check
from src.supabase_client import fetch_all

log = logging.getLogger(__name__)

TOP_N = 40


def fetch_matched_job_ids(client, user_id: str) -> set[str]:
    """job_ids already in matches_v2 for *user_id* — direct analog of v1's
    src.matches_store.fetch_existing_match_urls(). Callers exclude these from
    the jobs list passed into match_user() so gate_v2/vector/Tier1/Tier2 only
    ever evaluate jobs this user hasn't already been matched to. Without this,
    (a) a user gets re-emailed about the same jobs every run forever, (b)
    Tier 2 (Sonnet) re-spends on the same pair every run, and — the most
    severe consequence — (c) since vector_rerank is capped at TOP_N, a user's
    existing strong matches would keep winning top-N slots indefinitely and
    could permanently starve new jobs from ever reaching Tier 1 for an active
    user with several matches already."""
    rows = fetch_all(client.table("matches_v2").select("job_id").eq("user_id", user_id))
    return {row["job_id"] for row in rows}


def gate_survivors(jobs: list[dict], profile: dict) -> list[dict]:
    """jobs: rows with at least id/location_field/enrichment. Returns the subset
    that passes gate_v2.check for *profile*."""
    return [job for job in jobs if gate_v2_check(job, profile).passed]


def vector_rerank(client, *, profile_embedding, candidate_job_ids: list[str], top_n: int = TOP_N) -> list[dict]:
    """Cosine-similarity rerank of *candidate_job_ids* against *profile_embedding*
    via match_v2_top_jobs(). Returns [{id, similarity}, ...] descending by
    similarity, capped at top_n. Empty candidate_job_ids returns [] without an
    RPC round trip — there's nothing for Postgres to rank.
    """
    if not candidate_job_ids:
        return []
    response = client.rpc("match_v2_top_jobs", {
        "query_embedding": profile_embedding,
        "candidate_job_ids": candidate_job_ids,
        "match_limit": top_n,
    }).execute()
    return response.data or []


def match_user(client, *, user_id: str, profile: dict, profile_embedding, jobs: list[dict], top_n: int = TOP_N) -> dict:
    """Full v2 matching path for one user: gate_v2 survivors -> pgvector rerank to
    top_n. Returns {gate_survivor_count, top: [{...job fields, similarity}]}.
    """
    survivors = gate_survivors(jobs, profile)
    survivor_ids = [job["id"] for job in survivors]
    ranked = vector_rerank(client, profile_embedding=profile_embedding, candidate_job_ids=survivor_ids, top_n=top_n)

    by_id = {job["id"]: job for job in survivors}
    top = []
    for row in ranked:
        job = by_id.get(row["id"])
        if job is None:
            continue  # defensive: shouldn't happen, candidate_job_ids came from survivors
        top.append({**job, "similarity": row["similarity"]})

    log.info(
        "match_user %s: %d gate survivor(s), %d in top-%d by similarity",
        user_id, len(survivors), len(top), top_n,
    )
    return {"gate_survivor_count": len(survivors), "top": top}
