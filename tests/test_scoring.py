"""Tests for weighted-sum scoring logic (scoring.py) and rubric.yaml integrity."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from anthropic import RateLimitError

from src import scoring as scoring_mod
from src.schemas import HardGateResult
from src.scoring import ScoringError, _merge_tool_use_blocks, score, score_many, weighted_sum
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


# ---------------------------------------------------------------------------
# score_many — concurrent Tier 1 scoring (additive async path)
# ---------------------------------------------------------------------------

def test_score_many_preserves_input_order(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(20)]
    items = [(p, _PASSED_GATE) for p in postings]

    async def _fake_acall_llm(posting, profile, rubric):  # noqa: ARG001
        return make_tier1()

    monkeypatch.setattr(scoring_mod, "_acall_llm", _fake_acall_llm)

    results = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=5))
    assert [r.posting.url for r in results] == [p.url for p in postings]


def test_score_many_scoring_error_does_not_abort_others(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(5)]
    items = [(p, _PASSED_GATE) for p in postings]

    async def _fake_acall_llm(posting, profile, rubric):  # noqa: ARG001
        if posting.url.endswith("/job/2"):
            raise ValueError("simulated LLM failure")  # exhausts _ainvoke_with_retry -> ScoringError
        return make_tier1()

    monkeypatch.setattr(scoring_mod, "_acall_llm", _fake_acall_llm)

    results = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=5))

    assert len(results) == 5
    for i, result in enumerate(results):
        if i == 2:
            assert isinstance(result, ScoringError)
        else:
            assert not isinstance(result, ScoringError)


def test_score_many_non_scoring_error_propagates_and_aborts(monkeypatch, profile, rubric):
    """A non-ScoringError exception is a real bug, not an expected parse/rate-limit
    failure — score_many must not swallow it. Patches ascore() itself (rather than
    _acall_llm, which _ainvoke_with_retry would convert to ScoringError either way)
    to exercise score_many's own catch-only-ScoringError contract directly."""
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(3)]
    items = [(p, _PASSED_GATE) for p in postings]

    async def _fake_ascore(posting, gate, profile, rubric, _allm_fn=None):  # noqa: ARG001
        if posting.url.endswith("/job/1"):
            raise RuntimeError("a real bug, not a scoring failure")
        return score(posting, gate, profile, rubric, _llm_fn=lambda p, prof: make_tier1())

    monkeypatch.setattr(scoring_mod, "ascore", _fake_ascore)

    with pytest.raises(RuntimeError, match="a real bug"):
        asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=5))


def test_score_many_respects_max_concurrency(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(20)]
    items = [(p, _PASSED_GATE) for p in postings]

    max_concurrency = 3
    current = 0
    peak = 0

    async def _fake_acall_llm(posting, profile, rubric):  # noqa: ARG001
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)  # hold the slot briefly so overlaps are observable
        current -= 1
        return make_tier1()

    monkeypatch.setattr(scoring_mod, "_acall_llm", _fake_acall_llm)

    asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=max_concurrency))

    assert peak <= max_concurrency
    assert peak > 1  # sanity: concurrency actually happened, this isn't accidentally serial


# ---------------------------------------------------------------------------
# Rate-limit backoff — _acall_llm's inner retry loop, separate from the outer
# one-shot parse/validation retry (_ainvoke_with_retry)
# ---------------------------------------------------------------------------

def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def create(self, **kwargs):  # noqa: ARG002
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeAsyncClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


class _FakeContentBlock:
    def __init__(self, input_data, *, type="tool_use", name="score_posting"):  # noqa: A002
        self.input = input_data
        self.type = type
        self.name = name


class _FakeResponse:
    """*blocks* is one _FakeContentBlock or a list of them — supports both the
    normal single-block case and the multi-block-split regression case."""

    def __init__(self, blocks):
        self.content = blocks if isinstance(blocks, list) else [blocks]
        self.stop_reason = "tool_use"

    def model_dump_json(self):
        return "{}"


def _valid_tier1_response() -> _FakeResponse:
    return _FakeResponse(_FakeContentBlock(make_tier1().model_dump()))


# ---------------------------------------------------------------------------
# _merge_tool_use_blocks — multi-block tool-response parsing (regression: a model
# occasionally splits one response across several tool_use blocks instead of one
# block with every key)
# ---------------------------------------------------------------------------

def test_merge_tool_use_blocks_single_block():
    full = make_tier1().model_dump()
    response = _FakeResponse(_FakeContentBlock(full))
    assert _merge_tool_use_blocks(response, "score_posting") == full


def test_merge_tool_use_blocks_merges_multiple_blocks():
    full = make_tier1().model_dump()
    keys = list(full.keys())
    # split into 7 single-key blocks, one per dimension
    blocks = [_FakeContentBlock({k: full[k]}) for k in keys]
    response = _FakeResponse(blocks)

    merged = _merge_tool_use_blocks(response, "score_posting")

    assert merged == full


def test_merge_tool_use_blocks_ignores_non_matching_blocks():
    full = make_tier1().model_dump()
    other_tool_block = _FakeContentBlock({"unrelated": "data"}, name="some_other_tool")
    response = _FakeResponse([other_tool_block, _FakeContentBlock(full)])

    assert _merge_tool_use_blocks(response, "score_posting") == full


def test_merge_tool_use_blocks_ignores_non_tool_use_blocks():
    full = make_tier1().model_dump()
    text_block = _FakeContentBlock("some preamble text", type="text", name=None)
    response = _FakeResponse([text_block, _FakeContentBlock(full)])

    assert _merge_tool_use_blocks(response, "score_posting") == full


def test_merge_tool_use_blocks_raises_index_error_when_no_match():
    response = _FakeResponse([_FakeContentBlock({"x": 1}, name="unrelated_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(response, "score_posting")


def test_merge_tool_use_blocks_raises_index_error_on_empty_content():
    response = _FakeResponse([])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(response, "score_posting")


def test_acall_llm_succeeds_with_response_split_across_multiple_blocks(monkeypatch, profile, rubric):
    """End-to-end regression for the real bug: a Tier 1 response split across seven
    single-key tool_use blocks must still validate successfully via _acall_llm."""
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    full = make_tier1().model_dump()
    split_blocks = [_FakeContentBlock({k: v}) for k, v in full.items()]
    fake_client = _FakeAsyncClient([_FakeResponse(split_blocks)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert result.model_dump() == full


def test_acall_llm_succeeds_after_rate_limit_retries(monkeypatch, profile, rubric):
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    fake_client = _FakeAsyncClient([
        _rate_limit_error(), _rate_limit_error(), _valid_tier1_response(),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert result is not None
    assert fake_client.messages.call_count == 3  # 2 rate-limited attempts + 1 success


def test_acall_llm_persistent_rate_limit_raises(monkeypatch, profile, rubric):
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_ATTEMPTS", 3)

    fake_client = _FakeAsyncClient([_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(RateLimitError):
        asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert fake_client.messages.call_count == 3


def test_persistent_rate_limit_surfaces_as_scoring_error_via_ainvoke_with_retry(monkeypatch, profile, rubric):
    """No third error path: a RateLimitError that survives _acall_llm's backoff loop
    is just a normal exception to _ainvoke_with_retry, which retries once (calling
    _acall_llm — and its backoff loop — again) then raises ScoringError as usual."""
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_ATTEMPTS", 2)

    # 2 attempts per _acall_llm call, 2 calls total (initial + one retry) = 4 errors.
    fake_client = _FakeAsyncClient([_rate_limit_error() for _ in range(4)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(ScoringError):
        asyncio.run(scoring_mod.ascore(make_posting(), _PASSED_GATE, profile, rubric))

    assert fake_client.messages.call_count == 4
