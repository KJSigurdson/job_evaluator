"""Pydantic models mirroring the True North v2 `jobs.enrichment` JSONB column shape
(see supabase/migrations/20260731075824_true_north_v2_schema.sql). Strict — every
model rejects unknown fields, so a shape drift between the v2 enrichment LLM prompt
and this schema fails validation loudly instead of silently dropping/ignoring a
field.

Standalone from src/schemas.py, which holds the existing v1 pipeline's models
(Tier1ScoreOutput, Tier2EnrichmentOutput, MatchRow, etc.) — v1 is untouched by this
module and by the v2 schema in general.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SeniorityLevel = Literal["intern", "junior", "mid", "senior", "director"]


class SalaryRange(BaseModel):
    """All three fields are independently optional — a posting may state a min
    with no max, a currency with no figures at all, or nothing."""
    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    currency: str | None = None


class WorkArrangement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["remote", "hybrid", "onsite", "unspecified"] = "unspecified"
    remote_scope: Literal["global", "region_restricted", "unspecified"] = "unspecified"
    eligible_regions: list[str] = Field(default_factory=list)  # ISO country codes, or "EU"/"EMEA"-style groups
    timezone_constraint: str | None = None
    restriction_type: Literal["authorization", "timezone"] | None = None


class JobEnrichment(BaseModel):
    """Structured output shape for the v2 job-enrichment LLM call, stored verbatim
    into jobs.enrichment (jsonb)."""
    model_config = ConfigDict(extra="forbid")

    summary: str
    seniority: list[SeniorityLevel] = Field(default_factory=list)
    cause_area: list[str] = Field(default_factory=list)
    skill_tags: list[str] = Field(default_factory=list)
    salary: SalaryRange | None = None
    work_arrangement: WorkArrangement
