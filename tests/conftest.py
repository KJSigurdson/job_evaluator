"""Shared fixtures and stub factory helpers."""
from __future__ import annotations

from datetime import date

import pytest
import yaml

from src.schemas import DimensionScore, RawPosting, Tier1ScoreOutput


REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


@pytest.fixture(scope="session")
def profile() -> dict:
    with open(f"{REPO_ROOT}/profile.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def rubric() -> dict:
    with open(f"{REPO_ROOT}/rubric.yaml") as f:
        return yaml.safe_load(f)


def make_posting(**kwargs) -> RawPosting:
    defaults = dict(
        url="https://example.org/job/123",
        title="Head of Data",
        org="GiveDirectly",
        source="80k",
        location=None,
        seniority=None,
        comp=None,
        deadline=None,
        raw_text="Full job description text.",
    )
    defaults.update(kwargs)
    return RawPosting(**defaults)


def make_tier1(
    cause_mission_fit: float = 0.8,
    role_function_fit: float = 0.8,
    location_compatibility: float = 0.8,
    seniority_match: float = 0.8,
    comp_adequacy: float = 0.8,
    values_alignment: float = 0.8,
    skill_growth: float = 0.8,
) -> Tier1ScoreOutput:
    def dim(score: float) -> DimensionScore:
        return DimensionScore(score=score, rationale="stub")

    return Tier1ScoreOutput(
        cause_mission_fit=dim(cause_mission_fit),
        role_function_fit=dim(role_function_fit),
        location_compatibility=dim(location_compatibility),
        seniority_match=dim(seniority_match),
        comp_adequacy=dim(comp_adequacy),
        values_alignment=dim(values_alignment),
        skill_growth=dim(skill_growth),
    )
