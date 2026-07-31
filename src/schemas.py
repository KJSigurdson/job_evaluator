from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
# Tier 1 batched scoring — scores N postings in one call to amortize the static
# prompt prefix (tools schema + profile + rubric) across postings instead of
# paying it once per posting. See src/scoring.py's score_many_batched.
# Tier1ScoreOutput above is UNCHANGED and stays the sync score()/_call_llm and
# Tier 2 contract; these are additive, batch-only types.
# ---------------------------------------------------------------------------

class Tier1BatchItem(Tier1ScoreOutput):
    """One posting's score within a Tier 1 batch call — same 7 dimensions as
    Tier1ScoreOutput, plus posting_index to map this entry back to its posting
    (its 0-based position in the batch prompt's "=== POSTING N ===" list)."""
    posting_index: int


class Tier1BatchOutput(BaseModel):
    """Structured JSON output from a batched Tier 1 LLM call: one entry per posting
    in the batch, each tagged with posting_index. The model may return fewer than
    len(batch) entries (an omitted posting) — callers must not assume completeness."""
    scores: list[Tier1BatchItem]


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


class ExcludedUser(BaseModel):
    """A user user_store.fetch_users() dropped from this run because their stored
    gate config (scoring_weights.location_rule / seniority_rule) failed to parse.
    Distinct from a merely-missing scoring_weights row (which is a normal, quieter
    skip) — this is a data-integrity problem worth a run-level summary. See
    user_store._parse_rule / GateConfigError."""
    user_id: str
    field: str    # "location_rule" | "seniority_rule"
    reason: str   # human-readable parse-failure detail


# ---------------------------------------------------------------------------
# Gate config schema — strict validation of the STORED shape of location_rule /
# seniority_rule, used by scripts/validate_gates.py to audit every user's stored
# JSON. Deliberately NOT wired into the live gate/scoring path: user_store._parse_rule
# only requires syntactically-valid JSON that parses to a dict (a genuine parse
# failure there excludes the user from a live run — see GateConfigError). This is a
# STRICTER, audit-only check (unknown keys rejected, seniority levels constrained to
# the known five buckets) that can flag a row as suspect without changing what the
# live pipeline does with it.
# ---------------------------------------------------------------------------

class LocationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept_fully_remote: bool = False
    accept_hybrid_in: list[str] = Field(default_factory=list)
    accept_onsite_in: list[str] = Field(default_factory=list)


class SeniorityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept_levels: list[Literal["intern", "junior", "mid", "senior", "director"]] = Field(default_factory=list)


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
# Token usage — purely observational cost-visibility bookkeeping (see scoring.py's
# and enrich.py's score_many/enrich_many, and pipeline.py's TOKEN USAGE log lines).
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Aggregated Anthropic API token usage for a batch of calls. cache_read/
    cache_creation default to 0 since prompt caching isn't enabled yet — once it is,
    real non-zero values flow through _extract_usage() automatically, no code change
    needed here."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )


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
    # Diagnostic-only extras, surfaced to run-history telemetry (see run_store.py).
    # Additive — the JSON run-log line and every existing consumer ignore them.
    gate_breakdown: dict[str, int] = Field(default_factory=dict)  # gated_out split by rejection_reason
    notified: bool = False  # True iff a match-digest email was actually sent for this user this run


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
