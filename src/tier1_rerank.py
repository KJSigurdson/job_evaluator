"""True North v2 Tier 1 rerank — Phase 2: ONE Anthropic Batches submission per
run covering every active user's full candidate set, not one batch per user
(the Phase 0 finding: a per-user call site meant a separate submission+poll
per user, and the gate+vector top-40 cutoff was small enough that the same
~100 distinct jobs per user got rescored on almost every run — ~35x each on
average, confirmed against production).

Every score (above AND below each user's own threshold) is upserted into
`tier1_verdicts` (user_id, job_id) — a durable cache, not an in-memory
dict handed straight to Tier 2 the way Phase 0/1 worked. A verdict is valid
only while verdict.job_enriched_at == jobs.enriched_at (the job hasn't been
re-enriched since) AND verdict.profile_content_hash == profiles.
profile_content_hash (the profile hasn't changed since) — see
fetch_valid_verdict_job_ids() and tier2_enrich.fetch_candidates(). This is
what lets TIER1_MAX_PER_USER widen from the old top-40 to 200 without
rescoring the same already-verdicted jobs every single day: a job only
needs scoring once per (job content, profile content) pair, ever.

custom_id = f"{user_id}:{chunk_index}" (NOT user_id alone, unlike the old
one-batch-per-user design) — each user's candidate set is split into chunks
of at most _CHUNK_SIZE job cards (the Tier 1 prompt's own per-call size
budget), and every chunk across every user goes into the SAME Batches
submission. "{uuid}:{int}" is well under the Batches API's 64-character
custom_id cap.

Non-blocking via src.batch_poll.resume_or_submit, same as src/enrich_jobs.py
— exhausting the poll budget returns status='pending', NEVER raises;
src.pipeline_v2.run() stops before Tier 2/digest on that outcome, the same
gate enrich_jobs's own pending status already uses. src/drain_batches.py
(tier1_rerank + tier1_rerank_retry added to _DRAIN_MODULES) can also resume
and apply an outstanding batch between runs — drain-applied results write
ONLY to tier1_verdicts, same as the live path; the drainer must never
trigger Tier 2 or a digest send (see that module's own docstring).

Every verdict's job_enriched_at/profile_content_hash come from a
REQUEST-TIME manifest built at submission (_build_manifest) and persisted
alongside the batch_submissions row (src.batch_poll.resume_or_submit's
manifest param), NOT from a fresh `jobs`/`profiles` read at write time —
see _write_verdicts. This matters most for the drainer/reprocess_staged
paths: a batch's results can be applied hours after submission, by a
process with no in-memory trace of the original request, and if the job
got re-enriched (or the profile changed) in the meantime, write-time state
would silently record a verdict as scored against content the LLM never
actually saw. The manifest is what keeps that verdict correctly flagged
INVALID on its next read instead.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import yaml
from pydantic import ValidationError

from src.batch_poll import DEFAULT_POLL_BUDGET_SECONDS, resume_or_submit
from src.models.tier1_rerank import JobScore, Tier1RerankOutput
from src.supabase_client import (
    chunked_upsert,
    fetch_all,
    fetch_all_in,
    fetch_batch_manifest,
    insert_run_metrics,
    mark_batch_results_processed,
)

log = logging.getLogger(__name__)

_MODULE = "tier1_rerank"
_MODEL = "claude-haiku-4-5"
_TOOL_NAME = "score_jobs"
_MAX_TOKENS = 8192  # scaled for _CHUNK_SIZE job cards x (int score + one-line reason)

# Each user's gate+vector survivors are capped here before Tier 1 ever sees
# them — widened from the old top-40 (src.match_v2.TOP_N) specifically
# because verdict caching (see module docstring) means a wider pool no
# longer means re-paying for the same jobs every run; it means more DISTINCT
# jobs get a chance to reach Tier 1 at all, which top-40 was starving.
TIER1_MAX_PER_USER = 200

# Per-request job-card cap — unrelated to TIER1_MAX_PER_USER; this is purely
# about keeping one Batches request's prompt a manageable size. A user's
# TIER1_MAX_PER_USER candidates are split into ceil(200/40)=5 chunks at most.
_CHUNK_SIZE = 40

# Same pricing caveat as src/enrich_jobs.py: VERIFY AGAINST
# https://www.anthropic.com/pricing BEFORE TRUSTING estimated_cost_usd.
_HAIKU_SYNC_INPUT_USD_PER_MTOK = 1.00
_HAIKU_SYNC_OUTPUT_USD_PER_MTOK = 5.00
_BATCH_DISCOUNT = 0.5


# ---------------------------------------------------------------------------
# Prompt — identical to Phase 0/1's, operating on a CHUNK of job_cards
# (at most _CHUNK_SIZE), not always-exactly-40.
# ---------------------------------------------------------------------------

_TIER1_RERANK_STRIP_KEYS = frozenset({"location", "seniority_level", "comp_needs"})


def _tier1_rerank_profile_view(profile: dict) -> dict:
    return {k: v for k, v in profile.items() if k not in _TIER1_RERANK_STRIP_KEYS}


def _job_card_text(job: dict) -> str:
    enrichment = job.get("enrichment") or {}
    return (
        f"job_id: {job['id']}\n"
        f"title: {job.get('title')}\n"
        f"org: {job.get('org')}\n"
        f"summary: {enrichment.get('summary')}\n"
        f"seniority: {enrichment.get('seniority')}\n"
        f"cause_area: {enrichment.get('cause_area')}\n"
        f"skill_tags: {enrichment.get('skill_tags')}"
    )


def build_prompt(*, profile: dict, gate_constraints: dict, job_cards: list[dict]) -> str:
    """Pure/testable — no API call. profile: the UserContext.profile dict (same
    shape v1's Tier 1 prompt YAML-renders). gate_constraints:
    profile["hard_constraints"] — included as context only, per the explicit
    instruction below not to re-apply it. job_cards: dicts with at least
    id/title/org/enrichment — one CHUNK (at most _CHUNK_SIZE) of a user's
    gate+vector survivors, not necessarily all of them."""
    profile_text = yaml.dump(_tier1_rerank_profile_view(profile), allow_unicode=True, default_flow_style=False)
    constraints_text = yaml.dump(gate_constraints, allow_unicode=True, default_flow_style=False)
    cards_text = "\n\n".join(f"=== JOB {job['id']} ===\n{_job_card_text(job)}" for job in job_cards)

    return (
        "Score each of the following job postings for fit against the candidate "
        "profile below, on a 0-100 scale.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        "CANDIDATE'S HARD GATE CONSTRAINTS (context only — read the instruction "
        "below before using this):\n"
        f"{constraints_text}\n\n"
        "IMPORTANT: every job listed below has ALREADY passed this candidate's "
        "hard location + seniority gate. Do NOT re-check, re-apply, or penalize "
        "a job for the constraints shown above — that decision is final and out "
        "of scope here. The constraints are included only so you can weigh SOFT "
        "mismatches the hard gate can't see: e.g. a hybrid role that technically "
        "matches an accepted city but implies more office days than the "
        "candidate likely wants, a timezone-overlap preference not captured as "
        "a hard rule, or a stated region that's adjacent to (but not exactly) "
        "the candidate's accepted list. Use these only to nudge a score up or "
        "down within an otherwise-qualified job — never to fail or zero-score "
        "a job solely for something the hard gate already let through.\n\n"
        "For each job, return:\n"
        "  - job_id: copy exactly as given in that job's \"=== JOB ... ===\" header\n"
        "  - score: integer 0-100 — holistic fit considering cause alignment, "
        "role/skill match, seniority fit, and soft friction per the note above\n"
        "  - reason: one sentence explaining the score\n\n"
        "Score each job INDEPENDENTLY on its own merits — do not compare or "
        "rank jobs against each other, and do not let one job's presence "
        "affect another's score. Return exactly one entry per job listed "
        "below, no more, no fewer.\n\n"
        f"JOBS ({len(job_cards)}):\n{cards_text}"
    )


def build_retry_prompt(*, profile: dict, gate_constraints: dict, job_cards: list[dict], validation_error: str) -> str:
    base = build_prompt(profile=profile, gate_constraints=gate_constraints, job_cards=job_cards)
    return (
        f"{base}\n\n"
        "NOTE: A previous attempt at this task failed validation with the "
        f"following error — correct it in this attempt:\n{validation_error}"
    )


# ---------------------------------------------------------------------------
# Response parsing — pure, unit-testable without hitting the Anthropic API.
# ---------------------------------------------------------------------------

def _merge_tool_use_blocks(message: dict, tool_name: str) -> dict:
    """Duplicated from enrich_jobs.py's identical helper, per this codebase's
    established no-shared-utils convention for LLM-call plumbing."""
    merged: dict = {}
    found = False
    for block in message["content"]:
        if block["type"] == "tool_use" and block["name"] == tool_name:
            merged.update(block["input"])
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in message content")
    return merged


def _parse_scores(message: dict) -> Tier1RerankOutput:
    tool_input = _merge_tool_use_blocks(message, _TOOL_NAME)
    return Tier1RerankOutput.model_validate(tool_input)


def _parse_one_result(result: dict) -> tuple[Tier1RerankOutput | None, dict | None, str | None]:
    """Given one raw batch result dict, return (output, usage, error) — shared
    by the live round-1/round-2 loops, reprocess_staged(), and
    apply_drained_batch()."""
    if result["type"] != "succeeded":
        return None, None, f"batch result type={result['type']!r}"
    usage = result["message"]["usage"]
    try:
        return _parse_scores(result["message"]), usage, None
    except (IndexError, AttributeError, ValidationError) as exc:
        return None, usage, str(exc)


_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Output a 0-100 fit score and one-line reason for every job listed.",
    "input_schema": Tier1RerankOutput.model_json_schema(),
}


def _batch_request(custom_id: str, prompt: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _chunk_user_requests(user_requests: dict[str, dict]) -> dict[str, tuple[str, list[dict]]]:
    """user_requests: {user_id: {"profile", "gate_constraints", "job_cards"}}
    (job_cards: up to TIER1_MAX_PER_USER cards, already gate+vector-filtered
    and already excluding jobs with a valid verdict or an existing match —
    see src.pipeline_v2.py's Phase A). Returns {custom_id:
    (user_id, chunk_of_job_cards)} — flattening every user's chunks into one
    namespace, ready to build one Batches request per entry."""
    chunks_by_custom_id: dict[str, tuple[str, list[dict]]] = {}
    for user_id, u in user_requests.items():
        job_cards = u["job_cards"]
        for idx in range(0, len(job_cards), _CHUNK_SIZE):
            chunk_index = idx // _CHUNK_SIZE
            custom_id = f"{user_id}:{chunk_index}"
            chunks_by_custom_id[custom_id] = (user_id, job_cards[idx:idx + _CHUNK_SIZE])
    return chunks_by_custom_id


def _build_manifest(chunks_by_custom_id: dict[str, tuple[str, list[dict]]], user_requests: dict[str, dict]) -> dict:
    """Snapshot, per custom_id, exactly what this chunk's request will put in
    front of the LLM: the user it's for, that user's profile_content_hash
    AT REQUEST-BUILD TIME, and each job's enriched_at AT REQUEST-BUILD TIME
    (job_cards already carry "enriched_at" — see src.pipeline_v2.
    _prepare_tier1_request). Persisted alongside the batch submission (see
    src.batch_poll.resume_or_submit's manifest param) so a verdict's
    validity, later, is judged against what was actually SHOWN to the LLM —
    not against whatever `jobs`/`profiles` say by the time results are
    applied, which can be hours later via the external hourly drainer, and
    after the job's been re-enriched or the profile's changed in between.
    """
    return {
        custom_id: {
            "user_id": user_id,
            "profile_content_hash": user_requests[user_id].get("profile_content_hash"),
            "jobs": {job["id"]: job.get("enriched_at") for job in job_cards},
        }
        for custom_id, (user_id, job_cards) in chunks_by_custom_id.items()
    }


def _split_custom_id(custom_id: str) -> str:
    """custom_id = f'{user_id}:{chunk_index}' — user_id is a UUID (no colons
    in it), so partitioning on the LAST colon is unambiguous and correct
    even though this reads the same as a plain split here."""
    user_id, _, _chunk_index = custom_id.rpartition(":")
    return user_id


def _process_chunk_results(results: dict) -> dict:
    """results: {custom_id: raw_dict}. Returns {"scores_by_user":
    {user_id: [JobScore, ...]} (merged across every chunk that parsed
    successfully for that user), "parse_failures": {custom_id: error},
    "input_tokens": int, "output_tokens": int}. Pure — no I/O — shared by the
    live path, reprocess_staged(), and apply_drained_batch()."""
    scores_by_user: dict[str, list[JobScore]] = {}
    parse_failures: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for custom_id, result in results.items():
        output, usage, error = _parse_one_result(result)
        if usage is not None:
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
        if output is not None:
            user_id = _split_custom_id(custom_id)
            scores_by_user.setdefault(user_id, []).extend(output.scores)
        else:
            parse_failures[custom_id] = error

    return {
        "scores_by_user": scores_by_user, "parse_failures": parse_failures,
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
    }


def _write_verdicts(client, scores_by_user: dict[str, list[JobScore]], batch_id: str) -> int:
    """Upsert tier1_verdicts for every (user_id, score.job_id) in
    *scores_by_user* — every score, above AND below threshold (Tier 2's own
    threshold filtering happens later, reading this table, not here).

    job_enriched_at/profile_content_hash come from *batch_id*'s own
    REQUEST-TIME manifest (src.supabase_client.fetch_batch_manifest) —
    deliberately NOT a fresh read of `jobs`/`profiles` at write time. A
    verdict must record what the LLM actually saw when it produced this
    score; reading current state instead would let a job re-enriched (or a
    profile edited) between submission and this write masquerade as a
    verdict scored against the NEW content, when it was actually scored
    against the OLD content — exactly backwards from what the validity
    check (fetch_valid_verdict_job_ids) exists to catch. This is also what
    makes the function correct for the external hourly drainer and
    reprocess_staged() alike: both apply a batch's results long after
    submission, in a different process with no in-memory trace of the
    original request — the manifest is the only durable record of what was
    actually sent.

    A batch submitted without a manifest (shouldn't happen for this module
    going forward, but possible for one in flight across the migration
    that adds the manifest column) yields no job_enriched_at/
    profile_content_hash for anything — every score is skipped rather than
    guessed at. A score whose job_id isn't in its own chunk's manifest
    entry (shouldn't happen — the manifest is built from the exact same
    job_cards the request was built from) is likewise skipped. Returns the
    count actually written.
    """
    manifest = fetch_batch_manifest(client, batch_id) or {}
    enriched_at_by_job: dict[str, str] = {}
    hash_by_user: dict[str, str | None] = {}
    for entry in manifest.values():
        hash_by_user[entry["user_id"]] = entry.get("profile_content_hash")
        enriched_at_by_job.update(entry.get("jobs") or {})

    now = datetime.now(timezone.utc).isoformat()
    payloads = []
    skipped = 0
    for user_id, scores in scores_by_user.items():
        for score in scores:
            if score.job_id not in enriched_at_by_job:
                skipped += 1
                continue
            payloads.append({
                "user_id": user_id, "job_id": score.job_id, "score": score.score,
                "rationale": {"reason": score.reason} if score.reason else {},
                "job_enriched_at": enriched_at_by_job[score.job_id],
                "profile_content_hash": hash_by_user.get(user_id),
                "batch_id": batch_id, "scored_at": now,
            })

    if payloads:
        chunked_upsert(client.table("tier1_verdicts"), payloads, on_conflict="user_id,job_id")
    if skipped:
        log.warning(
            "_write_verdicts: skipped %d score(s) missing from batch %s's request-time manifest",
            skipped, batch_id,
        )
    return len(payloads)


def fetch_valid_verdict_job_ids(client, user_id: str, profile_content_hash: str | None) -> set[str]:
    """job_ids this user already has a VALID tier1_verdicts row for — valid
    meaning verdict.job_enriched_at still equals the job's CURRENT
    jobs.enriched_at (it hasn't been re-enriched since scoring) AND
    verdict.profile_content_hash still equals *profile_content_hash* (the
    profile hasn't changed since scoring). Callers (src.pipeline_v2.py's
    Phase A) exclude these from Tier 1 selection — re-scoring an
    already-validly-scored job is exactly the waste Phase 2 exists to stop.

    A verdict whose job has since been re-enriched, or whose profile has
    since changed, is INVALID — excluded from the returned set, so that job
    becomes eligible for Tier 1 again (same spirit as before: content or
    profile changed, so the old score no longer means anything).

    profile_content_hash is nullable (mirrors profiles.profile_content_hash
    itself — see 20260731130000_profiles_content_hash.sql), and a verdict's
    stored profile_content_hash can likewise be null (tier1_verdicts.
    profile_content_hash is nullable too — see 20261101190000_tier1_verdicts.
    sql). A null on EITHER side must never compare equal to a null on the
    other: `None != None` is False in Python, which would otherwise treat
    "we don't know this user's profile hash" as "matches the verdict's
    unknown hash" and incorrectly validate it. Null on either side is
    unconditionally INVALID, logged at WARNING (this should be rare —
    profile_content_hash is set as soon as profile_embed.py runs, and
    _prepare_tier1_request already requires a profile_embedding to exist
    before calling this — so a null here usually means the embed/hash step
    hasn't caught up yet for this user).
    """
    verdicts = fetch_all(
        client.table("tier1_verdicts").select("job_id,job_enriched_at,profile_content_hash").eq("user_id", user_id)
    )
    if not verdicts:
        return set()

    # This user's EVERY verdict, not just ones above threshold — grows
    # unbounded over time, so this MUST chunk its .in_() rather than fetch
    # all job_ids in one request (see fetch_all_in/_IN_FILTER_CHUNK_SIZE).
    job_ids = [v["job_id"] for v in verdicts]
    jobs_rows = fetch_all_in(client.table("jobs"), "id", job_ids, select_cols="id,enriched_at")
    current_enriched_at = {r["id"]: r["enriched_at"] for r in jobs_rows}

    valid: set[str] = set()
    for v in verdicts:
        verdict_hash = v.get("profile_content_hash")
        if profile_content_hash is None or verdict_hash is None:
            log.warning(
                "fetch_valid_verdict_job_ids: null profile_content_hash for user %s / job %s "
                "(current=%r, verdict=%r) — treating as INVALID, not a match",
                user_id, v["job_id"], profile_content_hash, verdict_hash,
            )
            continue
        if verdict_hash != profile_content_hash:
            continue
        if current_enriched_at.get(v["job_id"]) != v.get("job_enriched_at"):
            continue
        valid.add(v["job_id"])
    return valid


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def rerank_tier1_batch(
    client, anthropic_client, *, user_requests: dict[str, dict],
    run_id: str | None = None, trigger: str | None = None,
) -> dict:
    """user_requests: {user_id: {"profile", "gate_constraints", "job_cards"}}
    for every user with at least one unverdicted candidate this run (a user
    with zero is simply absent here — see module F in the Phase 2 spec: they
    skip Tier 1 cleanly and Tier 2 still runs from their existing verdicts).

    ONE Batches submission covers every user's every chunk. Any chunk whose
    output fails validation gets ONE retry — a second, smaller batch with
    the error appended. A chunk that fails both rounds is logged and
    dropped — never partial verdicts for it.

    Writes one run_metrics row (stage='tier1', tier1_calls,
    estimated_cost_usd from actual token usage) and every resolved score
    (above AND below threshold) to tier1_verdicts.

    Returns {"status": "complete", "scored": int, "failed": int} or
    {"status": "pending"|"idle", ...} — callers (src.pipeline_v2.run()) must
    check "status" and stop before Tier 2/digest on "pending", same gate
    src.enrich_jobs.enrich_jobs()'s pending status already uses.
    """
    chunks_by_custom_id = _chunk_user_requests(user_requests)
    manifest = _build_manifest(chunks_by_custom_id, user_requests)
    requests = [
        _batch_request(custom_id, build_prompt(
            profile=user_requests[user_id]["profile"],
            gate_constraints=user_requests[user_id]["gate_constraints"],
            job_cards=job_cards,
        ))
        for custom_id, (user_id, job_cards) in chunks_by_custom_id.items()
    ]

    outcome = resume_or_submit(
        anthropic_client, client, module=_MODULE, requests=requests,
        budget_seconds=DEFAULT_POLL_BUDGET_SECONDS, manifest=manifest,
    )
    if outcome.status == "idle":
        log.info("rerank_tier1_batch: no requests and nothing outstanding — nothing to do")
        return {"status": "complete", "scored": 0, "failed": 0}
    if outcome.status == "pending":
        # Nothing was retrieved — see enrich_jobs.enrich_jobs()'s identical
        # reasoning: still write a run_metrics row for the tier1 stage so
        # this run_id has SOME record, rather than a silent gap.
        insert_run_metrics(client, {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "stage": "tier1", "trigger": trigger, "tier1_calls": 0,
        })
        log.info(
            "rerank_tier1_batch: batch %s not finished within budget (elapsed %.0fs, "
            "counts=%s) — resuming next run.",
            outcome.batch_id, outcome.elapsed_seconds, outcome.request_counts,
        )
        return {"status": "pending", "batch_id": outcome.batch_id, "request_counts": outcome.request_counts}

    round1 = _process_chunk_results(outcome.results)
    round1_scores_by_user = round1["scores_by_user"]
    round2_scores_by_user: dict[str, list[JobScore]] = {}
    total_input_tokens = round1["input_tokens"]
    total_output_tokens = round1["output_tokens"]
    total_api_calls = len(requests)
    retry_pending = None
    retry_batch_id = None
    final_failures = dict(round1["parse_failures"])

    if round1["parse_failures"]:
        log.warning(
            "%d chunk(s) failed round 1, retrying once: %s",
            len(round1["parse_failures"]), list(round1["parse_failures"].keys()),
        )
        retry_requests = [
            _batch_request(custom_id, build_retry_prompt(
                profile=user_requests[chunks_by_custom_id[custom_id][0]]["profile"],
                gate_constraints=user_requests[chunks_by_custom_id[custom_id][0]]["gate_constraints"],
                job_cards=chunks_by_custom_id[custom_id][1],
                validation_error=error,
            ))
            for custom_id, error in round1["parse_failures"].items()
        ]
        total_api_calls += len(retry_requests)
        # Same job_cards as round 1 for these custom_ids — same request-time
        # job_enriched_at/profile_content_hash — so the retry's manifest is
        # just round 1's manifest restricted to the chunks being retried.
        retry_manifest = {cid: manifest[cid] for cid in round1["parse_failures"]}
        retry_outcome = resume_or_submit(
            anthropic_client, client, module=f"{_MODULE}_retry", requests=retry_requests,
            budget_seconds=DEFAULT_POLL_BUDGET_SECONDS, manifest=retry_manifest,
        )
        if retry_outcome.status == "pending":
            retry_pending = retry_outcome
            final_failures = {}  # not yet known — round 2 hasn't resolved
        else:
            retry_batch_id = retry_outcome.batch_id
            round2 = _process_chunk_results(retry_outcome.results)
            round2_scores_by_user = round2["scores_by_user"]
            total_input_tokens += round2["input_tokens"]
            total_output_tokens += round2["output_tokens"]
            final_failures = {cid: f"{err} (retry)" for cid, err in round2["parse_failures"].items()}
            mark_batch_results_processed(client, retry_outcome.batch_id, list(retry_outcome.results.keys()))

    for custom_id, reason in final_failures.items():
        log.error("Chunk %s permanently failed Tier 1 rerank: %s", custom_id, reason)

    # Verdicts written, and round 1's batch_results marked processed, BEFORE
    # run_metrics below — matching this codebase's crash-safety ordering
    # everywhere else would be backwards; here it's the other way round
    # deliberately: run_metrics' cost record for enrich_jobs.py gates on NOT
    # losing it to a later risky write, but tier1_verdicts IS the final
    # write here (no further risk downstream), so writing it first and
    # stamping run_metrics last just means a crash before run_metrics would
    # under-report cost for a batch whose verdicts already landed safely —
    # the better failure mode for THIS module specifically.
    #
    # Round 1 and round 2 scores are written against THEIR OWN batch_id —
    # each one's manifest lives under its own batch_submissions row, and a
    # chunk retried in round 2 was scored against the retry batch's request,
    # not round 1's (which is exactly the chunk that failed to parse).
    written = _write_verdicts(client, round1_scores_by_user, outcome.batch_id)
    if round2_scores_by_user:
        written += _write_verdicts(client, round2_scores_by_user, retry_batch_id)
    mark_batch_results_processed(client, outcome.batch_id, list(outcome.results.keys()))

    now = datetime.now(timezone.utc).isoformat()
    insert_run_metrics(client, {
        "run_at": now,
        "run_id": run_id,
        "stage": "tier1",
        "trigger": trigger,
        "tier1_calls": total_api_calls,
        "estimated_cost_usd": round(
            total_input_tokens / 1_000_000 * _HAIKU_SYNC_INPUT_USD_PER_MTOK * _BATCH_DISCOUNT
            + total_output_tokens / 1_000_000 * _HAIKU_SYNC_OUTPUT_USD_PER_MTOK * _BATCH_DISCOUNT,
            4,
        ),
    })

    if retry_pending is not None:
        log.info(
            "rerank_tier1_batch: round 1 verdicts committed (%d written) — retry batch %s "
            "not finished within budget (elapsed %.0fs, counts=%s) — resuming next run.",
            written, retry_pending.batch_id, retry_pending.elapsed_seconds, retry_pending.request_counts,
        )
        return {
            "status": "pending", "batch_id": retry_pending.batch_id,
            "request_counts": retry_pending.request_counts, "scored": written, "failed": 0,
        }

    log.info(
        "rerank_tier1_batch complete: %d user(s), %d chunk(s), %d verdict(s) written, %d chunk(s) failed",
        len(user_requests), len(requests), written, len(final_failures),
    )
    return {"status": "complete", "scored": written, "failed": len(final_failures)}


def apply_drained_batch(client, results: dict, *, batch_id: str) -> dict:
    """Drainer-only (src/drain_batches.py): applies an already-ended batch's
    *results* with NO retry attempt — writes verdicts ONLY, never triggers
    Tier 2 or a digest send (drain_batches.py's own hard rule: it never
    decides what new work to do, only finishes applying what someone else
    already submitted). A chunk that fails to parse here goes straight to
    "dropped, logged" — the same terminal outcome a live-path round-2 retry
    would likely reach anyway; draining just doesn't spend a second LLM call
    finding that out.

    Returns {"scored": int, "failed": int}.
    """
    round_result = _process_chunk_results(results)
    for custom_id, error in round_result["parse_failures"].items():
        log.error("apply_drained_batch: chunk %s permanently failed parsing while draining: %s", custom_id, error)

    written = _write_verdicts(client, round_result["scores_by_user"], batch_id)
    mark_batch_results_processed(client, batch_id, list(results.keys()))

    log.info("tier1 apply_drained_batch: batch %s -> %d verdict(s) written, %d failed", batch_id, written, len(round_result["parse_failures"]))
    return {"scored": written, "failed": len(round_result["parse_failures"])}


def reprocess_staged(client, batch_id: str) -> dict:
    """Recovery path (same process, crashed mid-run before finishing this
    module's own work — not the external hourly drainer, see
    apply_drained_batch for that): reprocess unprocessed `batch_results` rows
    for *batch_id* WITHOUT calling the Anthropic API again. Does NOT retry a
    second time on parse failure (would need a new LLM call) — anything
    that still fails to parse here is dropped, logged. Does NOT write
    run_metrics — the normal path already wrote it before whatever crashed
    necessitated recovery; re-inserting here would double-count cost.

    Returns {"selected", "scored", "failed"}.
    """
    staged = fetch_all(
        client.table("batch_results").select("custom_id,raw").eq("batch_id", batch_id).eq("processed", False)
    )
    if not staged:
        log.info("reprocess_staged: no unprocessed batch_results for batch %s", batch_id)
        return {"selected": 0, "scored": 0, "failed": 0}

    results = {row["custom_id"]: row["raw"] for row in staged}
    round_result = _process_chunk_results(results)
    for custom_id, error in round_result["parse_failures"].items():
        log.error("reprocess_staged: chunk %s permanently failed Tier 1 rerank: %s", custom_id, error)

    written = _write_verdicts(client, round_result["scores_by_user"], batch_id)
    mark_batch_results_processed(client, batch_id, list(results.keys()))

    log.info(
        "reprocess_staged: batch %s -> %d staged, %d verdict(s) written, %d failed",
        batch_id, len(staged), written, len(round_result["parse_failures"]),
    )
    return {"selected": len(staged), "scored": written, "failed": len(round_result["parse_failures"])}
