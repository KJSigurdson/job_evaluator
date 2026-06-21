from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Raw scrape output — produced by every source module
# ---------------------------------------------------------------------------

class RawPosting(BaseModel):
    url: str
    title: str
    org: str
    source: Literal["80k", "probably_good", "iap"]
    location: str | None = None
    seniority: str | None = None
    comp: str | None = None
    deadline: date | None = None
    raw_text: str  # full text fed to the LLM parser / Tier 1 scorer


# ---------------------------------------------------------------------------
# Hard gate
# ---------------------------------------------------------------------------

class HardGateResult(BaseModel):
    passed: bool
    location_pass: bool
    seniority_pass: bool
    reason: str | None = None  # explanation when passed=False


# ---------------------------------------------------------------------------
# Tier 1 — soft-dimension scoring (structured LLM output)
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str  # one-line explanation required by spec


class Tier1ScoreOutput(BaseModel):
    """Structured JSON output from the Tier 1 LLM call. Field names match rubric.yaml keys."""
    cause_mission_fit: DimensionScore
    role_function_fit: DimensionScore
    location_compatibility: DimensionScore
    seniority_match: DimensionScore
    comp_adequacy: DimensionScore
    values_alignment: DimensionScore
    skill_growth: DimensionScore


# ---------------------------------------------------------------------------
# Scored posting — internal object after Tier 1
# ---------------------------------------------------------------------------

class ScoredPosting(BaseModel):
    posting: RawPosting
    gate: HardGateResult
    scores: Tier1ScoreOutput
    fit_score: float = Field(ge=0.0, le=1.0)  # weighted sum; weights from rubric.yaml


# ---------------------------------------------------------------------------
# Tier 2 — enrichment + CV guidance (structured LLM output)
# ---------------------------------------------------------------------------

class Tier2EnrichmentOutput(BaseModel):
    """Structured JSON output from the Tier 2 LLM call. Only produced for fit_score >= threshold."""
    org_summary: str          # 2–3 sentences
    why_fits: str             # grounded in profile experience_inventory + values
    why_not_fits: str
    emphasize_in_cv: list[str]  # strengths/experiences to foreground
    deemphasize: list[str]      # experiences to gloss over as irrelevant


# ---------------------------------------------------------------------------
# Notion insertion shape — assembled from ScoredPosting + Tier2EnrichmentOutput
# ---------------------------------------------------------------------------

class NotionInsertRow(BaseModel):
    role: str
    org: str
    org_summary: str
    source: str
    url: str
    date_found: date
    deadline: date | None = None
    comp: str | None = None
    # Aggregate
    fit_score: float
    # Per-dimension (stored as individual Notion number properties for later recalibration)
    cause_mission_fit: float
    role_function_fit: float
    location_compatibility: float
    seniority_match: float
    comp_adequacy: float
    values_alignment: float
    skill_growth: float
    # Enrichment text
    why_fits: str
    why_not_fits: str
    emphasize_in_cv: str  # newline-joined list for Notion text field
    deemphasize: str      # newline-joined list for Notion text field
    status: str = "Proposed"


# ---------------------------------------------------------------------------
# Observability / run log
# ---------------------------------------------------------------------------

class SourceResult(BaseModel):
    source: str
    success: bool
    postings_scraped: int = 0
    error: str | None = None


class RunCounts(BaseModel):
    scraped: int = 0
    new_after_dedup: int = 0
    gated_out: int = 0
    scored: int = 0
    inserted: int = 0
    near_misses: int = 0
    parse_failures: int = 0


class NearMiss(BaseModel):
    posting: RawPosting
    scores: Tier1ScoreOutput
    fit_score: float


class RunLog(BaseModel):
    run_id: str
    timestamp: datetime
    model: str
    temperature: float
    counts: RunCounts
    source_results: list[SourceResult] = Field(default_factory=list)
    near_misses: list[NearMiss] = Field(default_factory=list)
    parse_failure_urls: list[str] = Field(default_factory=list)
