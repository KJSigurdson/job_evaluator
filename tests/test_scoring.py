"""Tests for weighted-sum scoring logic (scoring.py) and rubric.yaml integrity."""
from __future__ import annotations

import pytest

from src.schemas import HardGateResult
from src.scoring import ScoringError, score, weighted_sum
from tests.conftest import make_posting, make_tier1


# ---------------------------------------------------------------------------
# rubric.yaml integrity
# ---------------------------------------------------------------------------

def test_weights_sum_to_one(rubric):
    total = sum(d["weight"] for d in rubric["dimensions"].values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

def test_rubric_has_all_tier1_dimensions(rubric):
    from src.schemas import Tier1ScoreOutput
    expected = set(Tier1ScoreOutput.model_fields.keys())
    actual = set(rubric["dimensions"].keys())
    assert actual == expected, f"Rubric/schema mismatch: {actual.symmetric_difference(expected)}"

def test_threshold_present(rubric):
    assert "threshold" in rubric
    assert 0.0 < rubric["threshold"] <= 1.0

def test_near_miss_min_below_threshold(rubric):
    assert rubric["near_miss_min"] < rubric["threshold"]


# ---------------------------------------------------------------------------
# weighted_sum correctness
# ---------------------------------------------------------------------------

def test_all_ones_gives_one(rubric):
    scores = make_tier1(**{dim: 1.0 for dim in rubric["dimensions"]})
    assert weighted_sum(scores, rubric) == pytest.approx(1.0)

def test_all_zeros_gives_zero(rubric):
    scores = make_tier1(**{dim: 0.0 for dim in rubric["dimensions"]})
    assert weighted_sum(scores, rubric) == pytest.approx(0.0)

def test_single_dimension_weight(rubric):
    # Only cause_mission_fit = 1.0, rest = 0 → fit equals that dimension's weight.
    dims = {dim: 0.0 for dim in rubric["dimensions"]}
    dims["cause_mission_fit"] = 1.0
    scores = make_tier1(**dims)
    expected = rubric["dimensions"]["cause_mission_fit"]["weight"]
    assert weighted_sum(scores, rubric) == pytest.approx(expected)

def test_weighted_sum_above_threshold(rubric):
    scores = make_tier1(**{dim: 1.0 for dim in rubric["dimensions"]})
    assert weighted_sum(scores, rubric) >= rubric["threshold"]

def test_weighted_sum_below_threshold(rubric):
    scores = make_tier1(**{dim: 0.0 for dim in rubric["dimensions"]})
    assert weighted_sum(scores, rubric) < rubric["threshold"]


# ---------------------------------------------------------------------------
# score() with injected stub LLM function
# ---------------------------------------------------------------------------

def _stub_llm(posting, profile):  # noqa: ARG001
    return make_tier1()


_PASSED_GATE = HardGateResult(passed=True, location_pass=True, seniority_pass=True)


def test_score_returns_scored_posting(profile, rubric):
    posting = make_posting()
    result = score(posting, _PASSED_GATE, profile, rubric, _llm_fn=_stub_llm)
    assert result.posting == posting
    assert result.gate == _PASSED_GATE
    assert 0.0 <= result.fit_score <= 1.0

def test_score_fit_matches_weighted_sum(profile, rubric):
    posting = make_posting()
    result = score(posting, _PASSED_GATE, profile, rubric, _llm_fn=_stub_llm)
    expected = weighted_sum(result.scores, rubric)
    assert result.fit_score == pytest.approx(expected)

# ---------------------------------------------------------------------------
# Retry behaviour (offline stubs — no real LLM)
# ---------------------------------------------------------------------------

def test_score_raises_scoring_error_after_two_failures(profile, rubric):
    def _always_fails(posting, profile):  # noqa: ARG001
        raise ValueError("simulated LLM failure")

    with pytest.raises(ScoringError):
        score(make_posting(), _PASSED_GATE, profile, rubric, _llm_fn=_always_fails)

def test_score_succeeds_on_second_attempt(profile, rubric):
    calls = []

    def _fails_once(posting, profile):  # noqa: ARG001
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("first attempt failed")
        return make_tier1()

    result = score(make_posting(), _PASSED_GATE, profile, rubric, _llm_fn=_fails_once)
    assert len(calls) == 2
    assert 0.0 <= result.fit_score <= 1.0
