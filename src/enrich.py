"""Tier 2 enrichment + CV guidance (step 5). Only called for fit_score >= threshold."""
from __future__ import annotations

from src.schemas import ScoredPosting, Tier2EnrichmentOutput


def enrich(scored: ScoredPosting, profile: dict) -> Tier2EnrichmentOutput:
    raise NotImplementedError
