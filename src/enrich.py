"""Tier 2 enrichment + CV guidance. Only called for fit_score >= threshold."""
from __future__ import annotations

import json
import logging
import os

import yaml
from anthropic import Anthropic
from pydantic import ValidationError

from src.schemas import ScoredPosting, Tier2EnrichmentOutput

log = logging.getLogger(__name__)

_TIER2_MODEL_DEFAULT = "claude-sonnet-4-6"
_LIST_FIELDS = ("emphasize_in_cv", "deemphasize")


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
        tool_input = response.content[0].input
        return _parse_enrichment_output(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        raise ValueError(f"Tier 2 response parse error: {exc}") from exc
