"""Tier 1 rubric scoring.

weighted_sum() is pure math.
score() calls the LLM via an injectable _llm_fn (stub in tests, real in production).
"""
from __future__ import annotations

import logging
import os
from typing import Callable

import yaml
from anthropic import Anthropic
from pydantic import ValidationError

from src.schemas import HardGateResult, RawPosting, ScoredPosting, Tier1ScoreOutput

log = logging.getLogger(__name__)

_TIER1_MODEL_DEFAULT = "claude-haiku-4-5-20251001"


class ScoringError(Exception):
    """Raised when Tier 1 LLM scoring fails after one retry."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weighted_sum(scores: Tier1ScoreOutput, rubric: dict) -> float:
    """Compute the weighted aggregate fit score. Weights read from rubric.yaml dimensions."""
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
