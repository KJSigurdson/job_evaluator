"""Tier 2 enrichment + CV guidance. Only called for fit_score >= threshold."""
from __future__ import annotations

import logging
import os

import yaml
from anthropic import Anthropic
from pydantic import ValidationError

from src.schemas import ScoredPosting, Tier2EnrichmentOutput

log = logging.getLogger(__name__)

_TIER2_MODEL_DEFAULT = "claude-sonnet-4-6"


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
        return Tier2EnrichmentOutput.model_validate(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        raise ValueError(f"Tier 2 response parse error: {exc}") from exc
