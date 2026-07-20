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
    cause_area: str | None = None
    deadline: date | None = None
    posted_at: date | None = None  # None → exempt from recency filter (e.g. IAP)
    raw_text: str  # full text fed to the LLM parser / Tier 1 scorer


# ---------------------------------------------------------------------------
# Hard gate
# ---------------------------------------------------------------------------

class HardGateResult(BaseModel):
    passed: bool
    location_pass: bool
    seniority_pass: bool
    reason: str | None = None  # human-readable explanation when passed=False
    # Diagnostic-only category for aggregation (see pipeline.py's per-user gate-
    # rejection summary log): "location" | "seniority" | "hard_constraints" (both
    # failed) | None (passed). Purely additive — does not affect passed/location_pass/
    # seniority_pass or any downstream scoring/insert decision.
    rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# Tier 1 — soft-dimension scoring (structured LLM output)
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str  # one-line explanation required by spec


class Tier1ScoreOutput(BaseModel):
    """Structured JSON output from the Tier 1 LLM call. Field names match rubric dimension keys."""
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
    fit_score: float = Field(ge=0.0, le=1.0)  # weighted sum; weights from the user's scoring_weights row


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
# Per-user context — built from Supabase `profiles` + `scoring_weights` rows
# ---------------------------------------------------------------------------

class UserContext(BaseModel):
    user_id: str
    profile: dict  # shape gate.py / scoring.py / enrich.py expect (see user_store.py)
    rubric: dict   # shape scoring.py expects: {threshold, near_miss_min, dimensions}


# ---------------------------------------------------------------------------
# Matches insertion shape — assembled from ScoredPosting + Tier2EnrichmentOutput.
# Only ever contains job fields + model output. Never status/user_notes/discarded —
# those are user-owned columns the app manages; this schema has no fields for them
# so it is structurally impossible to write them via matches_store.upsert_match.
# ---------------------------------------------------------------------------

class MatchRow(BaseModel):
    title: str
    org: str
    org_summary: str
    source: str
    url: str
    canonical_url: str
    date_found: date
    deadline: date | None = None
    comp: str | None = None
    location: str | None = None
    seniority: str | None = None
    cause_area: str | None = None
    # Aggregate
    fit_score: float
    # Per-dimension scores as jsonb: {cause_mission_fit: 0.8, ...}
    dimension_scores: dict[str, float]
    # Enrichment text
    why_fits: str
    why_not_fits: str
    # jsonb arrays. None only if genuinely absent; a returned-but-empty enrichment
    # list is written as [] (see enrich.py — Tier2EnrichmentOutput requires these as
    # native list[str], defensively coerced from a JSON-array-in-a-string if a model
    # ever returns one). Left as real Python lists all the way to the upsert payload —
    # supabase-py serialises them to jsonb automatically; no manual json.dumps here.
    emphasize_in_cv: list[str] | None = None
    deemphasize: list[str] | None = None


# ---------------------------------------------------------------------------
# Observability / run log
# ---------------------------------------------------------------------------

class SourceResult(BaseModel):
    source: str
    success: bool
    postings_scraped: int = 0
    error: str | None = None


class RunCounts(BaseModel):
    """Shared counts from the single scrape + recency-filter pass, before the per-user loop."""
    scraped: int = 0
    recency_dropped: int = 0
    stale_dropped: int = 0       # dropped by the 14-day first-seen cutoff (deadline-less, posted_at-less)
    shared_pool_size: int = 0    # size of the fresh pool handed to every user's loop


class UserRunResult(BaseModel):
    user_id: str
    new_after_dedup: int = 0
    gated_out: int = 0
    scored: int = 0
    inserted: int = 0
    near_misses: int = 0
    parse_failures: int = 0


class ParseFailure(BaseModel):
    user_id: str
    url: str


class RunLog(BaseModel):
    run_id: str
    timestamp: datetime
    model: str
    temperature: float
    counts: RunCounts
    source_results: list[SourceResult] = Field(default_factory=list)
    user_results: list[UserRunResult] = Field(default_factory=list)
    parse_failures: list[ParseFailure] = Field(default_factory=list)
