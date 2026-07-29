"""Tier 1 rubric scoring.

weighted_sum() is pure math.
score() calls the LLM via an injectable _llm_fn (stub in tests, real in production).

Async path (score_many/ascore/_acall_llm/_ainvoke_with_retry) is an additive,
parallel implementation for pipeline.py's real run — it exists alongside score() and
friends, which stay exactly as they were (tests inject _llm_fn against the sync path).
score_many() runs many postings concurrently under a bounded semaphore, with a
rate-limit-specific backoff loop around the actual API call (separate from the
existing one-shot parse-retry, which the async path also preserves unchanged).

Batched Tier 1 path (score_many_batched/_acall_llm_batch) is a further, also
additive, alternative to score_many(): it scores several postings per Anthropic
call to amortize the static prompt prefix (tools schema + profile + rubric) that
score_many() otherwise pays once per posting. pipeline.py opts into it via
TIER1_BATCH_SIZE (>1); TIER1_BATCH_SIZE=1 keeps using score_many() unchanged. Falls
back to score_many's exact single-posting path (ascore/_acall_llm) per-posting
whenever a batch call fails or a posting_index is missing from its response.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import random
from typing import Awaitable, Callable

import yaml
from anthropic import Anthropic, APIStatusError, AsyncAnthropic
from pydantic import ValidationError

from src.schemas import (
    HardGateResult,
    RawPosting,
    ScoredPosting,
    Tier1BatchOutput,
    Tier1ScoreOutput,
    TokenUsage,
)

log = logging.getLogger(__name__)

_TIER1_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 1.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 30.0

# Static comp_adequacy scoring rule text — identical in both _call_llm and
# _acall_llm's prompts, and reused verbatim by scripts/measure_tier1_tokens.py so
# that script's cacheable-prefix measurement can't drift from the real prompt.
_COMP_ADEQUACY_RULE = (
    "SCORING RULE — comp_adequacy: If the posting states NO compensation information, "
    "score comp_adequacy at 0.6 (neutral — missing data is not evidence of low pay). "
    "Only score below 0.5 when stated compensation is demonstrably inadequate for the "
    "candidate's needs. Score above 0.7 when stated comp clearly meets or exceeds needs.\n\n"
)

# Operational/notification metadata that cannot inform any Tier 1 scoring dimension
# (cause/role/location/seniority/comp/values/skill-growth fit). Stripped from the
# profile dict ONLY for the Tier 1 prompt — see _tier1_profile_view. Tier 2
# (enrich.py) and every other consumer still receive the full, untrimmed profile.
# Checked against user_store._build_profile()'s actual output shape (location,
# seniority_level, comp_needs, cause_priorities, values_notes, career_goals,
# email_on_match, skills, experience_inventory, hard_constraints): only
# email_on_match is currently present among these — user_id/is_admin/archived_at/
# created_at/updated_at/display_name are already never included in the profile dict
# handed to gate/scoring/enrich at all, so this set is defensive/future-proofing as
# much as it is an active trim today.
_TIER1_STRIP_KEYS = frozenset({
    "user_id", "is_admin", "archived_at", "created_at", "updated_at",
    "email_on_match", "display_name",
})

# Batched Tier 1 scoring (score_many_batched/_acall_llm_batch): amortizes the
# ~2,801-token static prefix (tools schema + profile + rubric — see
# scripts/measure_tier1_tokens.py) across every posting in a batch instead of
# paying it once per posting. At the pilot batch_size=5, measured input
# tokens/posting drop from ~2,952 to ~710 (~76%); output tokens/posting are
# unchanged (each posting still gets its own full score+rationale set).
_TIER1_BATCH_TOOL_NAME = "score_postings"
# Validated ceiling for the batch_size=5 pilot: ~5 x ~561 observed output
# tokens/posting + headroom for JSON/tool-call structural overhead (repeated field
# names, rationale length variance). At the old single-posting max_tokens=2048,
# a 5-posting batch's response truncates mid-tool_use and the whole batch fails —
# this must scale with batch size, not stay fixed.
_TIER1_BATCH_MAX_TOKENS_PILOT = 8192
_TIER1_BATCH_PILOT_SIZE = 5


def _tier1_batch_max_tokens(batch_size: int) -> int:
    """Scale _TIER1_BATCH_MAX_TOKENS_PILOT linearly with batch_size, floored at the
    single-posting call's own max_tokens (2048) so a small batch is never given
    less headroom than a single posting already gets today."""
    return max(2048, round(_TIER1_BATCH_MAX_TOKENS_PILOT * batch_size / _TIER1_BATCH_PILOT_SIZE))

# Set by _acall_llm right after a successful response, read back by score_many's
# per-item task once ascore() returns — see score_many's docstring for why a
# ContextVar (not a changed return type) is how usage is threaded out of the async
# path without touching ascore()'s/_acall_llm's existing return contracts, which
# would otherwise ripple into every test that monkeypatches _acall_llm directly.
# asyncio.gather() wraps each coroutine argument in its own Task, and each Task gets
# an isolated COPY of the context (contextvars.copy_context()) — so concurrent items
# under score_many's semaphore never see each other's usage, and a stubbed/
# monkeypatched _acall_llm (which never touches this var) correctly leaves it unset,
# read back as a zero-usage default — exactly right, since a stub makes no real call.
_last_usage: contextvars.ContextVar[TokenUsage] = contextvars.ContextVar("scoring._last_usage")


class ScoringError(Exception):
    """Raised when Tier 1 LLM scoring fails after one retry."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weighted_sum(scores: Tier1ScoreOutput, rubric: dict) -> float:
    """Compute the weighted aggregate fit score. Weights read from rubric["dimensions"]."""
    total = 0.0
    for dim, cfg in rubric["dimensions"].items():
        total += getattr(scores, dim).score * cfg["weight"]
    return round(total, 6)


def _tier1_profile_view(profile: dict) -> dict:
    """Tier 1-only view of *profile*: drops operational/notification metadata (see
    _TIER1_STRIP_KEYS) that cannot inform any scoring dimension. Everything else
    (location, seniority_level, comp_needs, cause_priorities, values_notes,
    career_goals, skills, experience_inventory, hard_constraints, and any future
    key not in _TIER1_STRIP_KEYS) passes through unchanged. Tier 2 (enrich.py) is
    NOT affected — it's handed the original, full profile dict by pipeline.py."""
    return {k: v for k, v in profile.items() if k not in _TIER1_STRIP_KEYS}


def _build_dimensions_text(rubric: dict) -> str:
    """The SCORING DIMENSIONS block of the Tier 1 prompt. Pure/static per rubric —
    factored out so scripts/measure_tier1_tokens.py can reuse the exact real text
    instead of re-deriving it."""
    return "\n".join(
        f"- {name} (weight {cfg['weight']}): {cfg['description']}"
        for name, cfg in rubric["dimensions"].items()
    )


def _build_profile_text(profile: dict) -> str:
    """YAML-render *profile* for the CANDIDATE PROFILE block of the Tier 1 prompt.
    Factored out (not the trim itself — callers decide whether to pass the full
    profile or _tier1_profile_view(profile)) so scripts/measure_tier1_tokens.py can
    reuse the exact real rendering instead of re-deriving it."""
    return yaml.dump(profile, allow_unicode=True, default_flow_style=False)


def _build_batch_prompt(postings: list[RawPosting], profile: dict, rubric: dict) -> str:
    """Batched Tier 1 prompt: the same static prefix as the single-posting prompt
    (profile_text via _tier1_profile_view + _build_profile_text, dimensions_text,
    _COMP_ADEQUACY_RULE — byte-identical pieces, reused not re-derived) followed by
    every posting in the batch, each under an explicit "=== POSTING N ===" header
    matching its 0-based index. Includes an explicit instruction to score each
    posting independently — without it, a model given several postings in one
    prompt could plausibly start comparing/ranking them against each other, which
    would silently change what fit_score means relative to today's one-posting-
    at-a-time calls.
    """
    dimensions_text = _build_dimensions_text(rubric)
    profile_text = _build_profile_text(_tier1_profile_view(profile))

    postings_text = "\n\n".join(
        f"=== POSTING {i} ===\n{posting.raw_text}" for i, posting in enumerate(postings)
    )

    return (
        "Score each of the following job postings against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        f"{_COMP_ADEQUACY_RULE}"
        "JOB POSTINGS (score each independently): score each posting INDEPENDENTLY "
        "on its own merits — do NOT compare or rank postings against each other. "
        "Return exactly one entry per posting, with its posting_index matching the "
        "number in that posting's \"=== POSTING N ===\" header below.\n\n"
        f"{postings_text}"
    )


def score(
    posting: RawPosting,
    gate: HardGateResult,
    profile: dict,
    rubric: dict,
    _llm_fn: Callable[[RawPosting, dict], Tier1ScoreOutput] | None = None,
) -> ScoredPosting:
    """Score a posting that has already passed the hard gate.

    _llm_fn injectable: pass a stub in tests; omit in production to use the real Anthropic call.
    Retries once on parse/validation failure, then raises ScoringError.
    """
    llm_fn = _llm_fn if _llm_fn is not None else (lambda p, prof: _call_llm(p, prof, rubric))
    tier1 = _invoke_with_retry(llm_fn, posting, profile)
    fit = weighted_sum(tier1, rubric)
    return ScoredPosting(posting=posting, gate=gate, scores=tier1, fit_score=fit)


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _invoke_with_retry(
    fn: Callable[[RawPosting, dict], Tier1ScoreOutput],
    posting: RawPosting,
    profile: dict,
) -> Tier1ScoreOutput:
    for attempt in range(2):
        try:
            return fn(posting, profile)
        except Exception as exc:
            if attempt == 0:
                log.warning("Tier 1 attempt 1 failed for %s: %s", posting.url, exc)
                continue
            raise ScoringError(
                f"Tier 1 failed after retry for {posting.url}: {exc}"
            ) from exc
    raise ScoringError("unreachable")  # satisfies type checker


# ---------------------------------------------------------------------------
# Response parsing — pure, unit-testable without hitting the Anthropic API
# ---------------------------------------------------------------------------

def _merge_tool_use_blocks(response, tool_name: str) -> dict:
    """Merge every tool_use content block named *tool_name* into one dict.

    A response should normally be one tool_use block with every scoring key, but a
    model can occasionally split it across multiple tool_use blocks (one or a few
    keys each) instead of one block with all of them. response.content[0].input
    alone would then only see a fragment and fail validation identically on both
    retry attempts — a silent, permanent loss. Collecting and merging every
    matching block fixes that. Later blocks' keys win on conflict, though the same
    key appearing in two blocks shouldn't happen in practice.

    Raises IndexError if no matching block is found — same failure mode as the old
    response.content[0].input on an empty/unexpected response, so the existing
    (IndexError, AttributeError, ValidationError) handling downstream still catches it.
    """
    merged: dict = {}
    found = False
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            merged.update(block.input)
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in response.content")
    return merged


def _merge_tool_use_blocks_list(response, tool_name: str, list_key: str) -> dict:
    """Variant of _merge_tool_use_blocks for a tool output whose payload is a single
    LIST-valued key (Tier1BatchOutput's "scores"). _merge_tool_use_blocks's
    merged.update() OVERWRITES on key collision — correct for a single-object
    output where every block should carry the SAME keys, but wrong here: if the
    model splits the batch's "scores" array across multiple tool_use blocks
    (one or a few score entries each), .update() would silently keep only the
    LAST block's list and drop every entry from the others — the exact same class
    of silent data loss _merge_tool_use_blocks was written to fix for the
    single-posting path, just at the list-element level instead of the key level.

    Concatenates *list_key*'s value across every matching block instead. Any OTHER
    (non-list_key) key is still merged via .update() as before, so this stays a
    drop-in generalization rather than a special case — Tier1BatchOutput only has
    "scores" today, but this doesn't assume that.

    Raises IndexError if no matching block is found — same failure mode as
    _merge_tool_use_blocks, so the existing (IndexError, AttributeError,
    ValidationError) handling downstream still catches it. _merge_tool_use_blocks
    itself is unchanged; the single-posting path does not use this variant.
    """
    merged: dict = {}
    concatenated: list = []
    found = False
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            found = True
            block_input = dict(block.input)
            value = block_input.pop(list_key, None)
            if isinstance(value, list):
                concatenated.extend(value)
            merged.update(block_input)
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in response.content")
    merged[list_key] = concatenated
    return merged


# ---------------------------------------------------------------------------
# Real LLM call
# ---------------------------------------------------------------------------

def _call_llm(posting: RawPosting, profile: dict, rubric: dict) -> Tier1ScoreOutput:
    """Call Claude with tool-use to get a structured Tier1ScoreOutput."""
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT

    dimensions_text = _build_dimensions_text(rubric)
    profile_text = _build_profile_text(_tier1_profile_view(profile))

    prompt = (
        "Score the following job posting against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        f"{_COMP_ADEQUACY_RULE}"
        f"JOB POSTING:\n{posting.raw_text}"
    )

    response = Anthropic().messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        tools=[{
            "name": "score_posting",
            "description": "Output the seven-dimension rubric score for a job posting.",
            "input_schema": Tier1ScoreOutput.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "score_posting"},
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        tool_input = _merge_tool_use_blocks(response, "score_posting")
        return Tier1ScoreOutput.model_validate(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        log.error(
            "Tier 1 parse failure for %s (stop_reason=%s): %s\nraw response: %s",
            posting.url,
            response.stop_reason,
            exc,
            response.model_dump_json() if hasattr(response, "model_dump_json") else repr(response),
        )
        raise ValueError(f"Tier 1 response parse error: {exc}") from exc


# ---------------------------------------------------------------------------
# Async path — concurrent Tier 1 scoring, used only by the real pipeline run.
# score()/_call_llm()/_invoke_with_retry() above are untouched; this is a separate,
# parallel implementation so the sync path's test-injection surface stays stable.
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with +/-20% jitter, capped at _RATE_LIMIT_MAX_DELAY_SECONDS.
    attempt is 0-indexed (first retry after the initial attempt is attempt=0)."""
    base = min(_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt), _RATE_LIMIT_MAX_DELAY_SECONDS)
    jitter = base * random.uniform(-0.2, 0.2)
    return max(0.0, base + jitter)


def _extract_usage(response) -> TokenUsage:
    """Build a TokenUsage from a real Anthropic response.usage. cache_read_input_tokens
    / cache_creation_input_tokens are None (not 0) on the SDK's Usage object when
    caching isn't used for that call — normalized to 0 here so downstream summing
    never has to null-check."""
    usage = response.usage
    return TokenUsage(
        calls=1,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_tokens=usage.cache_creation_input_tokens or 0,
    )


def _is_retryable_status(status_code: int | None) -> bool:
    """True for transient, retry-worthy HTTP statuses: 429 (rate limit) and any 5xx
    (500/502/503 + 529 overloaded_error). Every other status — notably 4xx client
    errors like 400 invalid_request/401/403/404 — is permanent for a given request
    and must fail fast rather than burn the backoff budget."""
    return status_code == 429 or (status_code is not None and status_code >= 500)


async def _acall_llm(posting: RawPosting, profile: dict, rubric: dict) -> Tier1ScoreOutput:
    """Async twin of _call_llm — identical prompt construction, tool schema, and
    parse-error handling, via AsyncAnthropic so many calls can run concurrently under
    a semaphore (see score_many).

    The actual API call is wrapped in its own retry loop for transient server-side
    conditions — 429 (rate limit), 529 (overloaded), and 5xx generally, via
    _is_retryable_status — up to _RATE_LIMIT_MAX_ATTEMPTS attempts with exponential
    backoff + jitter. This is separate from _ainvoke_with_retry's one-shot
    parse/validation retry: concurrency produces 429s/529s the existing one-shot retry
    isn't designed for, but a transient error that survives all backoff attempts here
    still just raises and lets _ainvoke_with_retry's existing retry-then-ScoringError
    pattern handle it — no third error path. A non-transient status (any other 4xx —
    400/401/403/404) is re-raised immediately, never retried by this loop.
    """
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT

    dimensions_text = _build_dimensions_text(rubric)
    profile_text = _build_profile_text(_tier1_profile_view(profile))

    prompt = (
        "Score the following job posting against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        f"{_COMP_ADEQUACY_RULE}"
        f"JOB POSTING:\n{posting.raw_text}"
    )

    client = AsyncAnthropic()
    response = None
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0,
                tools=[{
                    "name": "score_posting",
                    "description": "Output the seven-dimension rubric score for a job posting.",
                    "input_schema": Tier1ScoreOutput.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "score_posting"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except APIStatusError as exc:
            # Retry only transient server-side conditions (429/529/5xx). Any other
            # status — permanent 4xx client errors — re-raises immediately and fails
            # fast through _ainvoke_with_retry instead of burning 5 backoff attempts.
            if not _is_retryable_status(exc.status_code):
                raise
            if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            delay = _backoff_delay(attempt)
            log.info(
                "Tier 1 transient API error %s for %s (attempt %d/%d) — backing off %.1fs",
                exc.status_code, posting.url, attempt + 1, _RATE_LIMIT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    try:
        tool_input = _merge_tool_use_blocks(response, "score_posting")
        result = Tier1ScoreOutput.model_validate(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        log.error(
            "Tier 1 parse failure for %s (stop_reason=%s): %s\nraw response: %s",
            posting.url,
            response.stop_reason,
            exc,
            response.model_dump_json() if hasattr(response, "model_dump_json") else repr(response),
        )
        raise ValueError(f"Tier 1 response parse error: {exc}") from exc

    # Only set on a response that actually parsed successfully — a discarded
    # parse-failure-then-retry attempt's tokens aren't separately tracked (the item
    # either ends in this same call's usage, or in ScoringError with zero usage; see
    # score_many's docstring).
    _last_usage.set(_extract_usage(response))
    return result


async def _ainvoke_with_retry(
    fn: Callable[[RawPosting, dict], Awaitable[Tier1ScoreOutput]],
    posting: RawPosting,
    profile: dict,
) -> Tier1ScoreOutput:
    """Async twin of _invoke_with_retry — identical semantics: one retry on any
    exception, then raise ScoringError. Separate from the rate-limit backoff in
    _acall_llm; parse/validation failures still get exactly one retry, as today."""
    for attempt in range(2):
        try:
            return await fn(posting, profile)
        except Exception as exc:
            if attempt == 0:
                log.warning("Tier 1 attempt 1 failed for %s: %s", posting.url, exc)
                continue
            raise ScoringError(
                f"Tier 1 failed after retry for {posting.url}: {exc}"
            ) from exc
    raise ScoringError("unreachable")  # satisfies type checker


async def ascore(
    posting: RawPosting,
    gate: HardGateResult,
    profile: dict,
    rubric: dict,
    _allm_fn: Callable[[RawPosting, dict], Awaitable[Tier1ScoreOutput]] | None = None,
) -> ScoredPosting:
    """Async twin of score(). _allm_fn injectable (async callable) for tests,
    mirroring score()'s _llm_fn pattern; omit to use the real Anthropic call.
    Retries once on parse/validation failure, then raises ScoringError."""
    llm_fn = _allm_fn if _allm_fn is not None else (lambda p, prof: _acall_llm(p, prof, rubric))
    tier1 = await _ainvoke_with_retry(llm_fn, posting, profile)
    fit = weighted_sum(tier1, rubric)
    return ScoredPosting(posting=posting, gate=gate, scores=tier1, fit_score=fit)


async def score_many(
    items: list[tuple[RawPosting, HardGateResult]],
    profile: dict,
    rubric: dict,
    max_concurrency: int = 10,
) -> tuple[list[ScoredPosting | ScoringError], TokenUsage]:
    """Score every (posting, gate) pair in *items* concurrently, bounded by
    max_concurrency via an asyncio.Semaphore. Preserves *items*' input order in the
    returned list (one output per input, same index).

    Per-item ScoringError is caught inside that item's task and returned as a value
    at that list position (not raised) — mirrors the sequential loop's per-posting
    log+skip+don't-cache handling. Any OTHER exception is a real bug, not an expected
    parse/rate-limit failure, and is NOT caught here: it propagates out of gather
    (default return_exceptions=False) and aborts the whole batch, same severity as an
    uncaught exception today.

    Also returns a TokenUsage aggregated across every item that scored successfully
    in this batch — a ScoringError contributes zero usage (see _last_usage's
    docstring). This is purely observational: it augments the return value with a
    second element but does not change scoring/gating/caching behavior or how many
    API calls are made.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_one(posting: RawPosting, gate: HardGateResult) -> tuple[ScoredPosting | ScoringError, TokenUsage]:
        async with semaphore:
            try:
                scored = await ascore(posting, gate, profile, rubric)
                return scored, _last_usage.get(TokenUsage())
            except ScoringError as exc:
                return exc, TokenUsage()

    outcomes = await asyncio.gather(*(_score_one(posting, gate) for posting, gate in items))
    results = [outcome for outcome, _usage in outcomes]
    total_usage = TokenUsage()
    for _outcome, usage in outcomes:
        total_usage = total_usage + usage
    return results, total_usage


# ---------------------------------------------------------------------------
# Batched Tier 1 scoring — scores multiple postings per API call to amortize the
# static prefix (tools schema + profile + rubric). Additive: score(), _call_llm(),
# _invoke_with_retry(), ascore(), _acall_llm(), and score_many() above are
# UNCHANGED — this is a separate path pipeline.py opts into via TIER1_BATCH_SIZE.
# ---------------------------------------------------------------------------

async def _acall_llm_batch(
    postings: list[RawPosting], profile: dict, rubric: dict,
) -> list[Tier1ScoreOutput | None]:
    """Score every posting in *postings* in ONE Anthropic call. Returns a list
    positionally aligned to *postings* — None at any index the model omitted, or
    whose posting_index was out of range (logged as a WARNING and skipped, not
    raised — score_many_batched's per-posting fallback handles filling the gap).

    Reuses _acall_llm's retry-loop machinery verbatim (_is_retryable_status,
    _backoff_delay, _RATE_LIMIT_MAX_ATTEMPTS) rather than duplicating it — same
    429/529/5xx transient backoff, same fail-fast on permanent 4xx. There is
    deliberately NO one-shot parse-retry here (unlike _ainvoke_with_retry wrapping
    _acall_llm): a parse failure or exhausted backoff here raises straight out to
    score_many_batched, whose fallback re-scores every posting in the batch
    individually via the existing ascore()/_acall_llm path — that path already has
    its own one-shot retry, so retrying the batch itself here would be redundant
    with, not additive to, the fallback.
    """
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT
    max_tokens = _tier1_batch_max_tokens(len(postings))
    prompt = _build_batch_prompt(postings, profile, rubric)

    client = AsyncAnthropic()
    response = None
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                tools=[{
                    "name": _TIER1_BATCH_TOOL_NAME,
                    "description": "Output the seven-dimension rubric score for each job posting in the batch.",
                    "input_schema": Tier1BatchOutput.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": _TIER1_BATCH_TOOL_NAME},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except APIStatusError as exc:
            if not _is_retryable_status(exc.status_code):
                raise
            if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            delay = _backoff_delay(attempt)
            log.info(
                "Tier 1 batch transient API error %s (%d posting(s), attempt %d/%d) — backing off %.1fs",
                exc.status_code, len(postings), attempt + 1, _RATE_LIMIT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    try:
        tool_input = _merge_tool_use_blocks_list(response, _TIER1_BATCH_TOOL_NAME, "scores")
        batch_output = Tier1BatchOutput.model_validate(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        log.error(
            "Tier 1 batch parse failure (%d posting(s), stop_reason=%s): %s\nraw response: %s",
            len(postings), response.stop_reason, exc,
            response.model_dump_json() if hasattr(response, "model_dump_json") else repr(response),
        )
        raise ValueError(f"Tier 1 batch response parse error: {exc}") from exc

    _last_usage.set(_extract_usage(response))  # per-batch, not per-posting — one call, one usage record

    results: list[Tier1ScoreOutput | None] = [None] * len(postings)
    for item in batch_output.scores:
        if 0 <= item.posting_index < len(postings):
            # Last-write-wins on a duplicate posting_index — same convention as
            # _merge_tool_use_blocks's "later blocks' keys win" — and never
            # corrupts any OTHER index, since each write targets its own slot.
            results[item.posting_index] = Tier1ScoreOutput(**item.model_dump(exclude={"posting_index"}))
        else:
            log.warning(
                "Tier 1 batch: model returned out-of-range posting_index=%d for a %d-posting batch, ignored",
                item.posting_index, len(postings),
            )
    return results


async def score_many_batched(
    items: list[tuple[RawPosting, HardGateResult]],
    profile: dict,
    rubric: dict,
    batch_size: int = 5,
    max_concurrency: int = 10,
) -> tuple[list[ScoredPosting | ScoringError], TokenUsage]:
    """Score every (posting, gate) pair in *items*, batching *batch_size* postings
    into each Anthropic call to amortize the ~2,801-token static prefix (tools
    schema + profile + rubric — see scripts/measure_tier1_tokens.py) that
    score_many() pays once PER POSTING. At the pilot batch_size=5, measured input
    tokens/posting drop from ~2,952 to ~710 (~76%); output tokens/posting are
    unchanged.

    Batches run concurrently under the existing asyncio.Semaphore(max_concurrency)
    — note this means up to max_concurrency * batch_size postings can be in flight
    at once. That's fewer, larger calls rather than more calls, so it reduces
    rate-limit pressure rather than increasing it, but is logged here so the actual
    in-flight ceiling is visible in the run log.

    PER-POSTING FALLBACK (preserves score_many's existing per-posting isolation
    guarantee): if a batch call raises for ANY reason (parse failure, exhausted
    rate-limit backoff, etc.), every posting in that batch falls back to
    individual scoring via the existing ascore()/_acall_llm path. If the batch
    call succeeds but is missing some posting_index values (the model omitted
    one, or returned one out of range), ONLY those specific postings fall back —
    postings the batch DID score keep their batch result. A posting that also
    fails on its individual fallback becomes a ScoringError at its position,
    exactly as score_many() already does. Both fallback cases are logged at
    WARNING with the batch index and how many postings fell back.

    Fallback calls within a chunk run sequentially, not concurrently: this is the
    rare/exceptional path (a healthy batch call needs none), and keeping it
    sequential keeps the _last_usage ContextVar's per-call set/read pairing
    trivially correct without needing a nested semaphore.

    Returns the same (results, TokenUsage) contract as score_many(): *items*'
    input order is preserved (one output per input, same index), and TokenUsage is
    aggregated across every batch call AND every fallback call.
    """
    if not items:
        return [], TokenUsage()

    semaphore = asyncio.Semaphore(max_concurrency)
    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    log.info(
        "Tier 1 batching: %d posting(s) in %d batch(es) of up to %d, max_concurrency=%d "
        "(up to %d posting(s) in flight)",
        len(items), len(chunks), batch_size, max_concurrency, max_concurrency * batch_size,
    )

    async def _score_chunk(
        chunk_index: int, chunk: list[tuple[RawPosting, HardGateResult]],
    ) -> tuple[list[ScoredPosting | ScoringError], TokenUsage]:
        async with semaphore:
            postings = [posting for posting, _gate in chunk]
            chunk_usage = TokenUsage()
            try:
                batch_scores = await _acall_llm_batch(postings, profile, rubric)
                chunk_usage = chunk_usage + _last_usage.get(TokenUsage())
            except Exception as exc:
                log.warning(
                    "Tier 1 batch %d (%d posting(s)) failed entirely (%s) — falling back to "
                    "individual scoring for all of them",
                    chunk_index, len(chunk), exc,
                )
                batch_scores = [None] * len(chunk)

            missing = sum(1 for s in batch_scores if s is None)
            if 0 < missing < len(chunk):
                log.warning(
                    "Tier 1 batch %d: %d/%d posting(s) missing from the batch response — "
                    "falling back to individual scoring for those",
                    chunk_index, missing, len(chunk),
                )

            outcomes: list[ScoredPosting | ScoringError] = []
            for (posting, gate), tier1 in zip(chunk, batch_scores):
                if tier1 is not None:
                    fit = weighted_sum(tier1, rubric)
                    outcomes.append(ScoredPosting(posting=posting, gate=gate, scores=tier1, fit_score=fit))
                    continue
                try:
                    scored = await ascore(posting, gate, profile, rubric)
                    outcomes.append(scored)
                    chunk_usage = chunk_usage + _last_usage.get(TokenUsage())
                except ScoringError as exc:
                    outcomes.append(exc)
            return outcomes, chunk_usage

    chunk_results = await asyncio.gather(
        *(_score_chunk(i, chunk) for i, chunk in enumerate(chunks))
    )

    results: list[ScoredPosting | ScoringError] = []
    total_usage = TokenUsage()
    for outcomes, chunk_usage in chunk_results:
        results.extend(outcomes)
        total_usage = total_usage + chunk_usage
    return results, total_usage
