"""Tier 1 rubric scoring.

weighted_sum() is pure math.
score() calls the LLM via an injectable _llm_fn (stub in tests, real in production).

Async path (score_many/ascore/_acall_llm/_ainvoke_with_retry) is an additive,
parallel implementation for pipeline.py's real run — it exists alongside score() and
friends, which stay exactly as they were (tests inject _llm_fn against the sync path).
score_many() runs many postings concurrently under a bounded semaphore, with a
rate-limit-specific backoff loop around the actual API call (separate from the
existing one-shot parse-retry, which the async path also preserves unchanged).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Awaitable, Callable

import yaml
from anthropic import Anthropic, AsyncAnthropic, RateLimitError
from pydantic import ValidationError

from src.schemas import HardGateResult, RawPosting, ScoredPosting, Tier1ScoreOutput

log = logging.getLogger(__name__)

_TIER1_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 1.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 30.0


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
# Real LLM call
# ---------------------------------------------------------------------------

def _call_llm(posting: RawPosting, profile: dict, rubric: dict) -> Tier1ScoreOutput:
    """Call Claude with tool-use to get a structured Tier1ScoreOutput."""
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT

    dimensions_text = "\n".join(
        f"- {name} (weight {cfg['weight']}): {cfg['description']}"
        for name, cfg in rubric["dimensions"].items()
    )
    profile_text = yaml.dump(profile, allow_unicode=True, default_flow_style=False)

    prompt = (
        "Score the following job posting against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        "SCORING RULE — comp_adequacy: If the posting states NO compensation information, "
        "score comp_adequacy at 0.6 (neutral — missing data is not evidence of low pay). "
        "Only score below 0.5 when stated compensation is demonstrably inadequate for the "
        "candidate's needs. Score above 0.7 when stated comp clearly meets or exceeds needs.\n\n"
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
        tool_input = response.content[0].input
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


async def _acall_llm(posting: RawPosting, profile: dict, rubric: dict) -> Tier1ScoreOutput:
    """Async twin of _call_llm — identical prompt construction, tool schema, and
    parse-error handling, via AsyncAnthropic so many calls can run concurrently under
    a semaphore (see score_many).

    The actual API call is wrapped in its own retry loop for anthropic.RateLimitError
    specifically — up to _RATE_LIMIT_MAX_ATTEMPTS attempts with exponential backoff +
    jitter. This is separate from _ainvoke_with_retry's one-shot parse/validation
    retry: concurrency produces 429s the existing one-shot retry isn't designed for,
    but a RateLimitError that survives all backoff attempts here still just raises
    and lets _ainvoke_with_retry's existing retry-then-ScoringError pattern handle it
    — no third error path.
    """
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT

    dimensions_text = "\n".join(
        f"- {name} (weight {cfg['weight']}): {cfg['description']}"
        for name, cfg in rubric["dimensions"].items()
    )
    profile_text = yaml.dump(profile, allow_unicode=True, default_flow_style=False)

    prompt = (
        "Score the following job posting against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        "SCORING RULE — comp_adequacy: If the posting states NO compensation information, "
        "score comp_adequacy at 0.6 (neutral — missing data is not evidence of low pay). "
        "Only score below 0.5 when stated compensation is demonstrably inadequate for the "
        "candidate's needs. Score above 0.7 when stated comp clearly meets or exceeds needs.\n\n"
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
        except RateLimitError:
            if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            delay = _backoff_delay(attempt)
            log.info(
                "Tier 1 rate-limited for %s (attempt %d/%d) — backing off %.1fs",
                posting.url, attempt + 1, _RATE_LIMIT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    try:
        tool_input = response.content[0].input
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
) -> list[ScoredPosting | ScoringError]:
    """Score every (posting, gate) pair in *items* concurrently, bounded by
    max_concurrency via an asyncio.Semaphore. Preserves *items*' input order in the
    returned list (one output per input, same index).

    Per-item ScoringError is caught inside that item's task and returned as a value
    at that list position (not raised) — mirrors the sequential loop's per-posting
    log+skip+don't-cache handling. Any OTHER exception is a real bug, not an expected
    parse/rate-limit failure, and is NOT caught here: it propagates out of gather
    (default return_exceptions=False) and aborts the whole batch, same severity as an
    uncaught exception today.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_one(posting: RawPosting, gate: HardGateResult) -> ScoredPosting | ScoringError:
        async with semaphore:
            try:
                return await ascore(posting, gate, profile, rubric)
            except ScoringError as exc:
                return exc

    return await asyncio.gather(*(_score_one(posting, gate) for posting, gate in items))
