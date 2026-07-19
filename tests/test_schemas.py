"""Tests for Pydantic schema validation (schemas.py)."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.dedup import canonical_url
from src.schemas import (
    DimensionScore,
    MatchRow,
    RawPosting,
    RunCounts,
    ScoredPosting,
    Tier1ScoreOutput,
)
from tests.conftest import make_posting, make_tier1


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------

def test_dimension_score_rejects_above_one():
    with pytest.raises(ValidationError):
        DimensionScore(score=1.01, rationale="too high")

def test_dimension_score_rejects_negative():
    with pytest.raises(ValidationError):
        DimensionScore(score=-0.01, rationale="negative")

def test_dimension_score_accepts_boundaries():
    DimensionScore(score=0.0, rationale="zero")
    DimensionScore(score=1.0, rationale="one")

def test_dimension_score_rationale_required():
    with pytest.raises(ValidationError):
        DimensionScore(score=0.5)


# ---------------------------------------------------------------------------
# RawPosting
# ---------------------------------------------------------------------------

def test_raw_posting_rejects_invalid_source():
    with pytest.raises(ValidationError):
        make_posting(source="invalid_source")

def test_raw_posting_accepts_all_valid_sources():
    for src in ("80k", "probably_good", "iap"):
        p = make_posting(source=src)
        assert p.source == src

def test_raw_posting_optional_fields_default_none():
    p = make_posting()
    assert p.location is None
    assert p.seniority is None
    assert p.comp is None
    assert p.cause_area is None
    assert p.deadline is None


# ---------------------------------------------------------------------------
# ScoredPosting
# ---------------------------------------------------------------------------

def test_scored_posting_rejects_fit_above_one():
    from src.schemas import HardGateResult
    gate = HardGateResult(passed=True, location_pass=True, seniority_pass=True)
    with pytest.raises(ValidationError):
        ScoredPosting(posting=make_posting(), gate=gate, scores=make_tier1(), fit_score=1.5)

def test_scored_posting_rejects_fit_below_zero():
    from src.schemas import HardGateResult
    gate = HardGateResult(passed=True, location_pass=True, seniority_pass=True)
    with pytest.raises(ValidationError):
        ScoredPosting(posting=make_posting(), gate=gate, scores=make_tier1(), fit_score=-0.1)


# ---------------------------------------------------------------------------
# Tier1ScoreOutput — field coverage
# ---------------------------------------------------------------------------

def test_tier1_has_seven_dimensions():
    assert len(Tier1ScoreOutput.model_fields) == 7

def test_tier1_field_names_match_expected():
    expected = {
        "cause_mission_fit", "role_function_fit", "location_compatibility",
        "seniority_match", "comp_adequacy", "values_alignment", "skill_growth",
    }
    assert set(Tier1ScoreOutput.model_fields.keys()) == expected


# ---------------------------------------------------------------------------
# RunCounts
# ---------------------------------------------------------------------------

def test_run_counts_defaults_to_zero():
    rc = RunCounts()
    assert rc.scraped == 0
    assert rc.recency_dropped == 0
    assert rc.stale_dropped == 0
    assert rc.shared_pool_size == 0


# ---------------------------------------------------------------------------
# MatchRow — has no status/user_notes/discarded fields (user-owned columns)
# ---------------------------------------------------------------------------

def test_match_row_has_no_user_owned_fields():
    forbidden = {"status", "user_notes", "discarded"}
    assert not (forbidden & set(MatchRow.model_fields.keys()))


def test_match_row_round_trip():
    url = "https://example.com/job/1"
    row = MatchRow(
        title="Head of Data",
        org="GiveDirectly",
        org_summary="...",
        source="80k",
        url=url,
        canonical_url=canonical_url(url),
        date_found=date.today(),
        fit_score=0.85,
        dimension_scores={
            "cause_mission_fit": 0.9,
            "role_function_fit": 0.9,
            "location_compatibility": 0.8,
            "seniority_match": 0.8,
            "comp_adequacy": 0.7,
            "values_alignment": 0.9,
            "skill_growth": 0.6,
        },
        why_fits="Strong mission alignment.",
        why_not_fits="No fundraising experience.",
        emphasize_in_cv=["BI leadership"],
        deemphasize=["IT portfolio work"],
    )
    assert row.fit_score == 0.85
    assert row.emphasize_in_cv == ["BI leadership"]
