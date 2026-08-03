"""True North v2 Tier 1 rerank: one Haiku call per user (Anthropic Batches API —
cron-appropriate batch pricing), scoring every job card in that user's gate+
vector top-N (see src/match_v2.py) 0-100 in a single prompt, alongside the
candidate's profile and gate constraints as context.

Shadow mode: writes only run_metrics (tier1_calls, estimated_cost_usd) — no
matches/seen/search_quota/digest write, no v1 table touched.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import yaml
from pydantic import ValidationError

from src.models.tier1_rerank import Tier1RerankOutput
from src.supabase_client import (
    chunked_upsert,
    fetch_all,
    find_outstanding_batch,
    persist_batch_results,
    record_batch_retrieved,
    record_batch_submitted,
)

log = logging.getLogger(__name__)

_MODULE = "tier1_rerank"
_MODEL = "claude-haiku-4-5"
_TOOL_NAME = "score_jobs"
_MAX_TOKENS = 8192  # scaled for TOP_N=40 job cards x (int score + one-line reason)
_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 3600

# Same pricing caveat as src/enrich_jobs.py: VERIFY AGAINST
# https://www.anthropic.com/pricing BEFORE TRUSTING estimated_cost_usd.
_HAIKU_SYNC_INPUT_USD_PER_MTOK = 1.00
_HAIKU_SYNC_OUTPUT_USD_PER_MTOK = 5.00
_BATCH_DISCOUNT = 0.5


# ---------------------------------------------------------------------------
# Prompt — DO NOT invoke the Batches API with this until it's been reviewed.
# ---------------------------------------------------------------------------

# location/seniority_level/comp_needs are dropped from the CANDIDATE PROFILE view
# for this prompt specifically: location + seniority are already fully decided by
# gate_v2 (surfaced separately below as GATE CONSTRAINTS, context-only), and comp
# isn't a Tier 1 rerank dimension here at all — including them would just be
# redundant noise, not signal.
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
    id/title/org/enrichment (from the gate_v2 + vector-rerank top-N)."""
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
# Operates on plain dicts (result.model_dump(mode="json") — see _run_batch()),
# not Anthropic SDK objects, so the live path and reprocess_staged()'s replay
# path share identical parsing code with no dual-shape handling.
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
    by the live round-1/round-2 loops and reprocess_staged()."""
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


# ---------------------------------------------------------------------------
# Batches API plumbing — duplicated from enrich_jobs.py's identical helpers
# ---------------------------------------------------------------------------

def _batch_request(user_id: str, prompt: str) -> dict:
    return {
        "custom_id": user_id,
        "params": {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _run_batch(anthropic_client, requests: list[dict], *, client, module: str = _MODULE) -> dict:
    """Submit *requests*, or resume an outstanding batch for *module* if a prior
    run left one — see enrich_jobs.py's identical helper's docstring for the
    full rationale. Persists every raw result to `batch_results` immediately on
    retrieval, before any parsing risk, and returns {custom_id: raw_dict}."""
    batch_id = find_outstanding_batch(client, module)
    if batch_id is not None:
        log.warning(
            "Resuming outstanding batch %s for module %r instead of submitting new work "
            "(a prior run likely didn't finish polling it)", batch_id, module,
        )
    else:
        batch = anthropic_client.messages.batches.create(requests=requests)
        batch_id = batch.id
        record_batch_submitted(client, batch_id, module)
        log.info("Submitted batch %s with %d request(s)", batch_id, len(requests))

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while True:
        batch = anthropic_client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch {batch_id} did not finish within {_POLL_TIMEOUT_SECONDS}s")
        log.info("Batch %s still %s — polling again in %ds", batch_id, batch.processing_status, _POLL_INTERVAL_SECONDS)
        time.sleep(_POLL_INTERVAL_SECONDS)

    results = {}
    for item in anthropic_client.messages.batches.results(batch_id):
        results[item.custom_id] = item.result.model_dump(mode="json")
    persist_batch_results(client, batch_id, results)
    record_batch_retrieved(client, batch_id)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def rerank_tier1(
    client, anthropic_client, *, users: dict[str, dict],
    run_id: str | None = None, trigger: str | None = None,
) -> dict:
    """users: {user_id: {"profile": ..., "gate_constraints": ..., "job_cards": [...]}}.

    One Batches API call scores every user in one round (custom_id=user_id). Any
    user whose output fails validation gets ONE retry — a second, smaller batch
    with the validation error appended to their prompt. A user who fails both
    rounds gets an empty scores list and is logged — never partial output.

    Writes one run_metrics row (tier1_calls, estimated_cost_usd from actual
    token usage). Returns {user_id: {"scores": [JobScore, ...]}}.
    """
    if not users:
        return {}

    requests = [
        _batch_request(user_id, build_prompt(
            profile=u["profile"], gate_constraints=u["gate_constraints"], job_cards=u["job_cards"],
        ))
        for user_id, u in users.items()
    ]
    results = _run_batch(anthropic_client, requests, client=client)
    total_api_calls = len(requests)

    scored: dict[str, Tier1RerankOutput] = {}
    failures: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for user_id, result in results.items():
        output, usage, error = _parse_one_result(result)
        if usage is not None:
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
        if output is not None:
            scored[user_id] = output
        else:
            failures[user_id] = error

    if failures:
        log.warning("%d user(s) failed round 1, retrying once: %s", len(failures), list(failures.keys()))
        retry_requests = [
            _batch_request(user_id, build_retry_prompt(
                profile=users[user_id]["profile"], gate_constraints=users[user_id]["gate_constraints"],
                job_cards=users[user_id]["job_cards"], validation_error=error,
            ))
            for user_id, error in failures.items()
        ]
        retry_results = _run_batch(anthropic_client, retry_requests, client=client, module=f"{_MODULE}_retry")
        total_api_calls += len(retry_requests)

        still_failing: dict[str, str] = {}
        for user_id, result in retry_results.items():
            output, usage, error = _parse_one_result(result)
            if usage is not None:
                total_input_tokens += usage["input_tokens"]
                total_output_tokens += usage["output_tokens"]
            if output is not None:
                scored[user_id] = output
            else:
                still_failing[user_id] = f"{error} (retry)"

        failures = still_failing

    for user_id, reason in failures.items():
        log.error("User %s permanently failed Tier 1 rerank: %s", user_id, reason)

    now = datetime.now(timezone.utc).isoformat()
    client.table("run_metrics").insert({
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
    }).execute()

    log.info(
        "Tier 1 rerank complete: %d user(s) requested, %d scored, %d failed",
        len(users), len(scored), len(failures),
    )
    return {user_id: {"scores": output.scores} for user_id, output in scored.items()}


def reprocess_staged(client, batch_id: str) -> dict:
    """Recovery path: reprocess unprocessed `batch_results` rows for *batch_id*
    WITHOUT calling the Anthropic API again. Unlike enrich_jobs.py/
    tier2_enrich.py, this module has no final table write of its own to redo —
    its only output is the in-memory scores dict the caller (pipeline_v2.py)
    feeds straight into tier2_enrich — so recovery here just re-parses and
    returns that same shape. Does NOT write run_metrics (already written by the
    normal path before whatever crashed necessitated recovery).

    Returns {user_id: {"scores": [JobScore, ...]}} — same shape as
    rerank_tier1()'s return.
    """
    staged = fetch_all(
        client.table("batch_results").select("custom_id,raw").eq("batch_id", batch_id).eq("processed", False)
    )
    if not staged:
        log.info("reprocess_staged: no unprocessed batch_results for batch %s", batch_id)
        return {}

    scored: dict[str, Tier1RerankOutput] = {}
    processed_rows: list[dict] = []
    for row in staged:
        user_id = row["custom_id"]
        output, _usage, error = _parse_one_result(row["raw"])
        if output is not None:
            scored[user_id] = output
        else:
            log.error("reprocess_staged: user %s permanently failed Tier 1 rerank: %s", user_id, error)
        processed_rows.append({"batch_id": batch_id, "custom_id": user_id, "raw": row["raw"], "processed": True})

    chunked_upsert(client.table("batch_results"), processed_rows, on_conflict="batch_id,custom_id")

    log.info("reprocess_staged: batch %s -> %d staged, %d scored", batch_id, len(staged), len(scored))
    return {user_id: {"scores": output.scores} for user_id, output in scored.items()}
