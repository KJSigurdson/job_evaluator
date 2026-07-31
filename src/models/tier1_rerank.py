"""Pydantic model for the True North v2 Tier 1 rerank LLM output — one Haiku call
per user scoring every job card in that user's gate+vector top-N in a single
prompt. Standalone from src/schemas.py (v1) and src/models/job_enrichment.py
(v2 enrichment) — this is the v2 per-user reranking step's own output shape.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class JobScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    score: int = Field(ge=0, le=100)
    reason: str


class Tier1RerankOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[JobScore]
