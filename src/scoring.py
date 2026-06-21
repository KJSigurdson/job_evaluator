"""Tier 1 rubric scoring (step 2). Calls LLM for structured dimension scores, computes weighted sum."""
from __future__ import annotations

from src.schemas import RawPosting, ScoredPosting, Tier1ScoreOutput


def score(posting: RawPosting, profile: dict, rubric: dict) -> ScoredPosting:
    raise NotImplementedError


def weighted_sum(scores: Tier1ScoreOutput, rubric: dict) -> float:
    raise NotImplementedError
