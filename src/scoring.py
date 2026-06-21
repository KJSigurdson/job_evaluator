"""Tier 1 rubric scoring. weighted_sum() is pure math; _call_llm() is wired in step 4."""
from __future__ import annotations

from typing import Callable

from src.schemas import HardGateResult, RawPosting, ScoredPosting, Tier1ScoreOutput


def weighted_sum(scores: Tier1ScoreOutput, rubric: dict) -> float:
    """Compute the weighted aggregate fit score from a Tier1ScoreOutput and rubric.yaml weights."""
    total = 0.0
    for dim, cfg in rubric["dimensions"].items():
        dimension_score = getattr(scores, dim)
        total += dimension_score.score * cfg["weight"]
    return round(total, 6)


def score(
    posting: RawPosting,
    gate: HardGateResult,
    profile: dict,
    rubric: dict,
    _llm_fn: Callable[[RawPosting, dict], Tier1ScoreOutput] | None = None,
) -> ScoredPosting:
    """Run Tier 1 scoring for a posting that has already passed the hard gate.

    _llm_fn is injectable: pass a stub in tests, leave None in production to use the real LLM call.
    """
    llm_fn = _llm_fn if _llm_fn is not None else _call_llm
    tier1 = llm_fn(posting, profile)
    fit = weighted_sum(tier1, rubric)
    return ScoredPosting(posting=posting, gate=gate, scores=tier1, fit_score=fit)


def _call_llm(posting: RawPosting, profile: dict) -> Tier1ScoreOutput:  # noqa: ARG001
    raise NotImplementedError("LLM scoring wired in step 4")
