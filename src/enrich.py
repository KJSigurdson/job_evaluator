"""Tier 2 enrichment + CV guidance. Only called for fit_score >= threshold.

Async path (enrich_many/aenrich/_acall_llm) is an additive, parallel implementation
for pipeline.py's real run — it exists alongside enrich() and friends, which stay
exactly as they were (tests monkeypatch _call_llm directly against the sync path;
enrich() has no _llm_fn injection parameter, so aenrich() doesn't add one either —
tests monkeypatch _acall_llm the same way). Mirrors scoring.py's Tier 1 concurrency
pattern, including the same RateLimitError-specific backoff loop (duplicated, not
imported — see _merge_tool_use_blocks's docstring for why).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random

import yaml
from anthropic import Anthropic, AsyncAnthropic, RateLimitError
from pydantic import ValidationError

from src.schemas import ScoredPosting, Tier2EnrichmentOutput

log = logging.getLogger(__name__)

_TIER2_MODEL_DEFAULT = "claude-sonnet-4-6"
_LIST_FIELDS = ("emphasize_in_cv", "deemphasize")
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 1.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 30.0


class EnrichmentError(Exception):
    """Raised when Tier 2 LLM enrichment fails after one retry."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich(scored: ScoredPosting, profile: dict) -> Tier2EnrichmentOutput:
    """Generate org summary and CV guidance for a high-fit posting.

    Retries once on parse/validation failure, then raises EnrichmentError.
    """
    for attempt in range(2):
        try:
            return _call_llm(scored, profile)
        except Exception as exc:
            if attempt == 0:
                log.warning("Tier 2 attempt 1 failed for %s: %s", scored.posting.url, exc)
                continue
            raise EnrichmentError(
                f"Tier 2 failed after retry for {scored.posting.url}: {exc}"
            ) from exc
    raise EnrichmentError("unreachable")  # satisfies type checker


# ---------------------------------------------------------------------------
# Response parsing — pure, unit-testable without hitting the Anthropic API
# ---------------------------------------------------------------------------

def _coerce_list_field(value):
    """emphasize_in_cv/deemphasize must end up as real list[str], not a stringified
    JSON array. The tool schema requires a native array, but if a model ever returns
    a JSON-array-in-a-string anyway, parse it back into a real list here rather than
    let Tier2EnrichmentOutput validation fail (and burn the retry) over a formatting
    quirk. Anything else (already a list, or a genuinely malformed/non-array string)
    is passed through unchanged so normal list[str] validation raises its own clear
    error and the standard retry-then-skip path runs.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    return parsed if isinstance(parsed, list) else value


def _parse_enrichment_output(tool_input: dict) -> Tier2EnrichmentOutput:
    """Validate raw tool-use input into Tier2EnrichmentOutput, defensively coercing
    emphasize_in_cv/deemphasize via _coerce_list_field first."""
    normalized = dict(tool_input)
    for field in _LIST_FIELDS:
        if field in normalized:
            normalized[field] = _coerce_list_field(normalized[field])
    return Tier2EnrichmentOutput.model_validate(normalized)


def _merge_tool_use_blocks(response, tool_name: str) -> dict:
    """Merge every tool_use content block named *tool_name* into one dict.

    A response should normally be one tool_use block with every enrichment key, but
    a model can occasionally split it across multiple tool_use blocks (one or a few
    keys each) instead of one block with all of them. response.content[0].input
    alone would then only see a fragment and fail validation identically on both
    retry attempts — a silent, permanent loss. Collecting and merging every
    matching block fixes that. Later blocks' keys win on conflict, though the same
    key appearing in two blocks shouldn't happen in practice.

    Raises IndexError if no matching block is found — same failure mode as the old
    response.content[0].input on an empty/unexpected response, so the existing
    (IndexError, AttributeError, ValidationError) handling downstream still catches it.

    Duplicated from scoring.py's identical helper rather than shared: this codebase
    keeps each module's private helpers self-contained (no cross-module utils
    module), and the two copies differ only in which tool_name they're called with.
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


# ---------------------------------------------------------------------------
# Real LLM call
# ---------------------------------------------------------------------------

def _call_llm(scored: ScoredPosting, profile: dict) -> Tier2EnrichmentOutput:
    """Call Claude with tool-use to get a structured Tier2EnrichmentOutput."""
    model = os.environ.get("TIER2_MODEL") or _TIER2_MODEL_DEFAULT
    profile_text = yaml.dump(profile, allow_unicode=True, default_flow_style=False)

    scores_summary = "\n".join(
        f"  {dim}: {getattr(scored.scores, dim).score:.2f} — "
        f"{getattr(scored.scores, dim).rationale}"
        for dim in type(scored.scores).model_fields
    )

    prompt = (
        "Generate enrichment content for a high-fit job posting.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"JOB POSTING:\n{scored.posting.raw_text}\n\n"
        f"TIER 1 DIMENSION SCORES (fit_score {scored.fit_score:.2f}):\n{scores_summary}\n\n"
        "Ground why_fits and emphasize_in_cv in specific items from the candidate's "
        "experience_inventory. Be concise and role-specific."
    )

    response = Anthropic().messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        tools=[{
            "name": "enrich_posting",
            "description": "Generate enrichment content for a high-fit job posting.",
            "input_schema": Tier2EnrichmentOutput.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "enrich_posting"},
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        tool_input = _merge_tool_use_blocks(response, "enrich_posting")
        return _parse_enrichment_output(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        raise ValueError(f"Tier 2 response parse error: {exc}") from exc


# ---------------------------------------------------------------------------
# Async path — concurrent Tier 2 enrichment, used only by the real pipeline run.
# enrich()/_call_llm() above are untouched (aside from Part 1's merge-block fix,
# which changed both files' _call_llm identically); this is a separate, parallel
# implementation so the sync path's test-injection surface stays stable.
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with +/-20% jitter, capped at _RATE_LIMIT_MAX_DELAY_SECONDS.
    attempt is 0-indexed (first retry after the initial attempt is attempt=0).
    Duplicated from scoring.py's identical helper — see _merge_tool_use_blocks's
    docstring for why duplicated rather than shared."""
    base = min(_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt), _RATE_LIMIT_MAX_DELAY_SECONDS)
    jitter = base * random.uniform(-0.2, 0.2)
    return max(0.0, base + jitter)


async def _acall_llm(scored: ScoredPosting, profile: dict) -> Tier2EnrichmentOutput:
    """Async twin of _call_llm — identical prompt construction, tool schema, and
    parse-error handling (including the multi-block merge fix), via AsyncAnthropic
    so many enrichments can run concurrently under a semaphore (see enrich_many).

    The actual API call is wrapped in its own retry loop for anthropic.RateLimitError
    specifically — up to _RATE_LIMIT_MAX_ATTEMPTS attempts with exponential backoff +
    jitter, same pattern as scoring.py's _acall_llm. Separate from aenrich's one-shot
    parse/validation retry: a RateLimitError that survives all backoff attempts here
    still just raises and lets that existing retry-then-EnrichmentError pattern
    handle it — no third error path.
    """
    model = os.environ.get("TIER2_MODEL") or _TIER2_MODEL_DEFAULT
    profile_text = yaml.dump(profile, allow_unicode=True, default_flow_style=False)

    scores_summary = "\n".join(
        f"  {dim}: {getattr(scored.scores, dim).score:.2f} — "
        f"{getattr(scored.scores, dim).rationale}"
        for dim in type(scored.scores).model_fields
    )

    prompt = (
        "Generate enrichment content for a high-fit job posting.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"JOB POSTING:\n{scored.posting.raw_text}\n\n"
        f"TIER 1 DIMENSION SCORES (fit_score {scored.fit_score:.2f}):\n{scores_summary}\n\n"
        "Ground why_fits and emphasize_in_cv in specific items from the candidate's "
        "experience_inventory. Be concise and role-specific."
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
                    "name": "enrich_posting",
                    "description": "Generate enrichment content for a high-fit job posting.",
                    "input_schema": Tier2EnrichmentOutput.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "enrich_posting"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except RateLimitError:
            if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            delay = _backoff_delay(attempt)
            log.info(
                "Tier 2 rate-limited for %s (attempt %d/%d) — backing off %.1fs",
                scored.posting.url, attempt + 1, _RATE_LIMIT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    try:
        tool_input = _merge_tool_use_blocks(response, "enrich_posting")
        return _parse_enrichment_output(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        raise ValueError(f"Tier 2 response parse error: {exc}") from exc


async def aenrich(scored: ScoredPosting, profile: dict) -> Tier2EnrichmentOutput:
    """Async twin of enrich(). No _llm_fn injection parameter — enrich() has none
    either; tests monkeypatch _acall_llm directly, same as _call_llm for the sync
    path. Retries once on parse/validation failure, then raises EnrichmentError."""
    for attempt in range(2):
        try:
            return await _acall_llm(scored, profile)
        except Exception as exc:
            if attempt == 0:
                log.warning("Tier 2 attempt 1 failed for %s: %s", scored.posting.url, exc)
                continue
            raise EnrichmentError(
                f"Tier 2 failed after retry for {scored.posting.url}: {exc}"
            ) from exc
    raise EnrichmentError("unreachable")  # satisfies type checker


async def enrich_many(
    items: list[ScoredPosting],
    profile: dict,
    max_concurrency: int = 10,
) -> list[Tier2EnrichmentOutput | EnrichmentError]:
    """Enrich every ScoredPosting in *items* concurrently, bounded by max_concurrency
    via an asyncio.Semaphore. Preserves *items*' input order in the returned list
    (one output per input, same index).

    Per-item EnrichmentError is caught inside that item's task and returned as a
    value at that list position (not raised) — mirrors scoring.py's score_many
    contract exactly: log+skip+don't-cache handling for expected failures. Any
    OTHER exception is a real bug, not an expected parse/rate-limit failure, and is
    NOT caught here: it propagates out of gather (default return_exceptions=False)
    and aborts the whole batch, same severity as an uncaught exception today.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _enrich_one(scored: ScoredPosting) -> Tier2EnrichmentOutput | EnrichmentError:
        async with semaphore:
            try:
                return await aenrich(scored, profile)
            except EnrichmentError as exc:
                return exc

    return await asyncio.gather(*(_enrich_one(scored) for scored in items))
