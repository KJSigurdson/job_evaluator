"""Tests for weighted-sum scoring logic (scoring.py) and rubric.yaml integrity."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from anthropic import BadRequestError, OverloadedError, RateLimitError

from src import scoring as scoring_mod
from src.schemas import HardGateResult, TokenUsage
from src.scoring import (
    ScoringError,
    _build_batch_prompt,
    _build_dimensions_text,
    _build_profile_text,
    _merge_tool_use_blocks,
    _merge_tool_use_blocks_list,
    _tier1_profile_view,
    score,
    score_many,
    score_many_batched,
    weighted_sum,
)
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

    results, _usage = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=5))
    assert [r.posting.url for r in results] == [p.url for p in postings]


def test_score_many_scoring_error_does_not_abort_others(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(5)]
    items = [(p, _PASSED_GATE) for p in postings]

    async def _fake_acall_llm(posting, profile, rubric):  # noqa: ARG001
        if posting.url.endswith("/job/2"):
            raise ValueError("simulated LLM failure")  # exhausts _ainvoke_with_retry -> ScoringError
        return make_tier1()

    monkeypatch.setattr(scoring_mod, "_acall_llm", _fake_acall_llm)

    results, _usage = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=5))

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


def _overloaded_error() -> OverloadedError:
    """529 overloaded_error — a transient Anthropic-side capacity condition. Distinct
    exception class from RateLimitError (429), so the pre-fix backoff loop, which only
    caught RateLimitError, let it fall straight through to the one-shot retry."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=529, request=request)
    return OverloadedError("overloaded", response=response, body=None)


def _bad_request_error() -> BadRequestError:
    """400 invalid_request — a permanent client error (e.g. credit exhaustion). Must
    NOT be retried by the backoff loop; it should fail fast through the outer path."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError("invalid_request", response=response, body=None)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.last_kwargs = None  # captures the most recent .create() call's kwargs for prompt-content assertions

    async def create(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
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


class _FakeUsage:
    """Mirrors anthropic.types.Usage's field names that _extract_usage reads.
    cache_*_input_tokens default to None, matching the real SDK when caching isn't
    used for a given call — _extract_usage normalizes that to 0."""

    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read_input_tokens=None, cache_creation_input_tokens=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _FakeResponse:
    """*blocks* is one _FakeContentBlock or a list of them — supports both the
    normal single-block case and the multi-block-split regression case."""

    def __init__(self, blocks, *, usage=None):
        self.content = blocks if isinstance(blocks, list) else [blocks]
        self.stop_reason = "tool_use"
        self.usage = usage if usage is not None else _FakeUsage()

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


# ---------------------------------------------------------------------------
# 529 overloaded / 5xx backoff — the broadened clause must treat these exactly
# like 429s (same 5-attempt cap, same backoff), while 4xx client errors fail fast.
# ---------------------------------------------------------------------------

def test_acall_llm_succeeds_after_overloaded_retries(monkeypatch, profile, rubric):
    """529s are now retried by the same backoff loop as 429s — N overloaded errors
    then a success resolves after N retries, not a permanent drop."""
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    fake_client = _FakeAsyncClient([
        _overloaded_error(), _overloaded_error(), _valid_tier1_response(),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert result is not None
    assert fake_client.messages.call_count == 3  # 2 overloaded attempts + 1 success


def test_acall_llm_persistent_overloaded_raises(monkeypatch, profile, rubric):
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_ATTEMPTS", 3)

    fake_client = _FakeAsyncClient([_overloaded_error(), _overloaded_error(), _overloaded_error()])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(OverloadedError):
        asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert fake_client.messages.call_count == 3


def test_persistent_overloaded_surfaces_as_scoring_error(monkeypatch, profile, rubric):
    """A 529 that survives _acall_llm's backoff is just a normal exception to
    _ainvoke_with_retry — no new error type; it still becomes ScoringError."""
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_ATTEMPTS", 2)

    # 2 attempts per _acall_llm call, 2 calls total (initial + one retry) = 4 errors.
    fake_client = _FakeAsyncClient([_overloaded_error() for _ in range(4)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(ScoringError):
        asyncio.run(scoring_mod.ascore(make_posting(), _PASSED_GATE, profile, rubric))

    assert fake_client.messages.call_count == 4


def test_acall_llm_bad_request_not_retried_by_backoff(monkeypatch, profile, rubric):
    """A 400 is permanent for this request — the backoff loop must re-raise it
    immediately (call_count == 1), never consuming further attempts. It falls through
    to the outer one-shot-retry path exactly as any non-transient error always did."""
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    # If the 400 were (wrongly) retried, the loop would go on to consume the valid
    # response; asserting call_count == 1 proves it fails fast on the first attempt.
    fake_client = _FakeAsyncClient([_bad_request_error(), _valid_tier1_response()])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(BadRequestError):
        asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    assert fake_client.messages.call_count == 1


# ---------------------------------------------------------------------------
# Token usage aggregation — audit-only, additive to score_many's return value.
# Uses the real _acall_llm (via _FakeAsyncClient/_FakeResponse) rather than
# monkeypatching it away, since the usage-capture ContextVar is set inside the real
# function — a stubbed _acall_llm correctly contributes zero usage (see below).
# ---------------------------------------------------------------------------

def test_acall_llm_sets_usage_from_real_response(monkeypatch, profile, rubric):
    """_last_usage is a ContextVar scoped to the asyncio Task that sets it — reading
    it back must happen within that same task (a read from outside the task, e.g.
    after asyncio.run() has already returned, would see nothing: Task context
    mutations don't propagate to the caller). Hence the set-and-read both happen
    inside one wrapper coroutine driven by a single asyncio.run() call."""
    fake_client = _FakeAsyncClient([_FakeResponse(
        _FakeContentBlock(make_tier1().model_dump()),
        usage=_FakeUsage(input_tokens=123, output_tokens=45, cache_read_input_tokens=7, cache_creation_input_tokens=9),
    )])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    async def _call_and_read_usage():
        await scoring_mod._acall_llm(make_posting(), profile, rubric)
        return scoring_mod._last_usage.get()

    usage = asyncio.run(_call_and_read_usage())
    assert usage.calls == 1
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45
    assert usage.cache_read_tokens == 7
    assert usage.cache_creation_tokens == 9


def test_acall_llm_usage_defaults_cache_fields_to_zero_when_none(monkeypatch, profile, rubric):
    """Real SDK Usage objects have cache_*_input_tokens=None when caching isn't used
    for that call (it isn't yet, anywhere in this codebase) — must normalize to 0,
    not leak None into the aggregate."""
    fake_client = _FakeAsyncClient([_FakeResponse(
        _FakeContentBlock(make_tier1().model_dump()),
        usage=_FakeUsage(cache_read_input_tokens=None, cache_creation_input_tokens=None),
    )])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    async def _call_and_read_usage():
        await scoring_mod._acall_llm(make_posting(), profile, rubric)
        return scoring_mod._last_usage.get()

    usage = asyncio.run(_call_and_read_usage())
    assert usage.cache_read_tokens == 0
    assert usage.cache_creation_tokens == 0


def test_score_many_aggregates_usage_across_batch(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(3)]
    items = [(p, _PASSED_GATE) for p in postings]

    fake_client = _FakeAsyncClient([
        _FakeResponse(_FakeContentBlock(make_tier1().model_dump()), usage=_FakeUsage(input_tokens=100, output_tokens=10)),
        _FakeResponse(_FakeContentBlock(make_tier1().model_dump()), usage=_FakeUsage(input_tokens=200, output_tokens=20)),
        _FakeResponse(_FakeContentBlock(make_tier1().model_dump()), usage=_FakeUsage(input_tokens=300, output_tokens=30)),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=1))

    assert len(results) == 3
    assert usage.calls == 3
    assert usage.input_tokens == 600
    assert usage.output_tokens == 60


def test_score_many_scoring_error_contributes_zero_usage(monkeypatch, profile, rubric):
    """A batch with one persistently-failing item: that item's ScoringError
    contributes no usage, but the other items' usage still aggregates correctly —
    mirrors score_many's existing error-isolation test pattern."""
    postings = [make_posting(url=f"https://example.org/job/{i}") for i in range(3)]
    items = [(p, _PASSED_GATE) for p in postings]

    async def _fake_acall_llm(posting, profile, rubric):  # noqa: ARG001
        if posting.url.endswith("/job/1"):
            raise ValueError("simulated LLM failure")  # exhausts retry -> ScoringError, no usage set
        return make_tier1()

    monkeypatch.setattr(scoring_mod, "_acall_llm", _fake_acall_llm)

    results, usage = asyncio.run(scoring_mod.score_many(items, profile, rubric, max_concurrency=1))

    assert isinstance(results[1], ScoringError)
    # Stubbed _acall_llm never touches _last_usage, so even the 2 successful items
    # correctly report zero usage here — this test only locks in that a failing item
    # doesn't crash or corrupt the aggregate, not that stubs report nonzero usage.
    assert usage.calls == 0
    assert usage.input_tokens == 0


# ---------------------------------------------------------------------------
# _tier1_profile_view — Tier 1-only trim of operational/notification metadata
# ---------------------------------------------------------------------------

def test_tier1_profile_view_strips_email_on_match():
    profile = {"location": "Remote", "email_on_match": True}
    view = _tier1_profile_view(profile)
    assert "email_on_match" not in view
    assert view["location"] == "Remote"


def test_tier1_profile_view_strips_all_known_operational_keys_when_present():
    profile = {
        "location": "Remote",
        "user_id": "u1", "is_admin": True, "archived_at": "2026-01-01",
        "created_at": "2026-01-01", "updated_at": "2026-01-01",
        "email_on_match": True, "display_name": "Kalle",
    }
    view = _tier1_profile_view(profile)
    assert view == {"location": "Remote"}


def test_tier1_profile_view_preserves_scoring_relevant_keys():
    profile = {
        "location": "Sundsvall, Sweden",
        "seniority_level": "Senior",
        "comp_needs": "Market rate",
        "cause_priorities": ["global health"],
        "values_notes": "GWWC pledge",
        "career_goals": "Data leadership",
        "skills": "SQL, Python",
        "experience_inventory": "Work:\n- Head of Data",
        "hard_constraints": {"location": {}, "seniority": {}},
    }
    assert _tier1_profile_view(profile) == profile


def test_tier1_profile_view_noop_when_no_operational_keys_present():
    profile = {"location": "Remote", "skills": "SQL"}
    assert _tier1_profile_view(profile) == profile


def test_tier1_profile_view_does_not_mutate_input():
    profile = {"location": "Remote", "email_on_match": True}
    _tier1_profile_view(profile)
    assert "email_on_match" in profile  # original untouched


# ---------------------------------------------------------------------------
# _build_dimensions_text / _build_profile_text — pure prompt-piece builders,
# factored out of _call_llm/_acall_llm so scripts/measure_tier1_tokens.py can
# reuse the exact real text without copy-pasting it.
# ---------------------------------------------------------------------------

def test_build_dimensions_text_matches_rubric(rubric):
    text = _build_dimensions_text(rubric)
    for dim, cfg in rubric["dimensions"].items():
        assert f"- {dim} (weight {cfg['weight']}): {cfg['description']}" in text


def test_build_profile_text_is_yaml_dump_of_profile():
    profile = {"location": "Remote", "skills": "SQL"}
    text = _build_profile_text(profile)
    assert "location: Remote" in text
    assert "skills: SQL" in text


# ---------------------------------------------------------------------------
# End-to-end: the trim is actually wired into the real prompt sent to Anthropic,
# for BOTH the sync (_call_llm) and async (_acall_llm) real-call paths.
# ---------------------------------------------------------------------------

def test_acall_llm_prompt_excludes_stripped_profile_keys(monkeypatch, rubric):
    profile = {"location": "Remote", "email_on_match": True}
    fake_client = _FakeAsyncClient([_valid_tier1_response()])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    asyncio.run(scoring_mod._acall_llm(make_posting(), profile, rubric))

    prompt = fake_client.messages.last_kwargs["messages"][0]["content"]
    assert "email_on_match" not in prompt
    assert "location: Remote" in prompt


def test_call_llm_prompt_excludes_stripped_profile_keys(monkeypatch, rubric):
    profile = {"location": "Remote", "email_on_match": True}

    captured = {}

    class _FakeSyncMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _valid_tier1_response()

    class _FakeSyncClient:
        def __init__(self):
            self.messages = _FakeSyncMessages()

    monkeypatch.setattr(scoring_mod, "Anthropic", _FakeSyncClient)

    scoring_mod._call_llm(make_posting(), profile, rubric)

    prompt = captured["messages"][0]["content"]
    assert "email_on_match" not in prompt
    assert "location: Remote" in prompt


# ---------------------------------------------------------------------------
# Batched Tier 1 scoring — _build_batch_prompt / _merge_tool_use_blocks_list /
# _acall_llm_batch / score_many_batched. score()/_call_llm()/_invoke_with_retry()/
# ascore()/_acall_llm()/score_many() above are untouched by any of this.
# ---------------------------------------------------------------------------

def _batch_item_dict(posting_index: int, **tier1_kwargs) -> dict:
    d = make_tier1(**tier1_kwargs).model_dump()
    d["posting_index"] = posting_index
    return d


def _batch_response(scores: list[dict], *, usage=None) -> "_FakeResponse":
    return _FakeResponse(
        _FakeContentBlock({"scores": scores}, name=scoring_mod._TIER1_BATCH_TOOL_NAME),
        usage=usage,
    )


def test_build_batch_prompt_contains_all_postings_with_indices(rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"POSTING TEXT {i}") for i in range(3)]
    prompt = _build_batch_prompt(postings, {"location": "Remote"}, rubric)
    for i in range(3):
        assert f"=== POSTING {i} ===" in prompt
        assert f"POSTING TEXT {i}" in prompt


def test_build_batch_prompt_includes_independence_instruction(rubric):
    postings = [make_posting(raw_text="text")]
    prompt = _build_batch_prompt(postings, {"location": "Remote"}, rubric)
    assert "INDEPENDENTLY" in prompt
    assert "do NOT compare or rank" in prompt
    assert "posting_index" in prompt


def test_build_batch_prompt_excludes_stripped_profile_keys(rubric):
    postings = [make_posting(raw_text="text")]
    prompt = _build_batch_prompt(postings, {"location": "Remote", "email_on_match": True}, rubric)
    assert "email_on_match" not in prompt
    assert "location: Remote" in prompt


# --- _merge_tool_use_blocks_list -------------------------------------------

def test_merge_tool_use_blocks_list_single_block():
    response = _FakeResponse(_FakeContentBlock(
        {"scores": [{"posting_index": 0, "x": 1}]}, name="score_postings",
    ))
    merged = _merge_tool_use_blocks_list(response, "score_postings", "scores")
    assert merged["scores"] == [{"posting_index": 0, "x": 1}]


def test_merge_tool_use_blocks_list_concatenates_across_blocks():
    """The regression this variant exists for: .update()-based merging would keep
    only the LAST block's "scores" list, silently dropping every entry from
    earlier blocks. Concatenation must preserve all of them."""
    block1 = _FakeContentBlock({"scores": [{"posting_index": 0, "x": 1}]}, name="score_postings")
    block2 = _FakeContentBlock({"scores": [{"posting_index": 1, "x": 2}]}, name="score_postings")
    response = _FakeResponse([block1, block2])

    merged = _merge_tool_use_blocks_list(response, "score_postings", "scores")

    assert merged["scores"] == [{"posting_index": 0, "x": 1}, {"posting_index": 1, "x": 2}]


def test_merge_tool_use_blocks_list_three_way_split_no_entries_lost():
    blocks = [
        _FakeContentBlock({"scores": [{"posting_index": i, "x": i}]}, name="score_postings")
        for i in range(3)
    ]
    response = _FakeResponse(blocks)
    merged = _merge_tool_use_blocks_list(response, "score_postings", "scores")
    assert merged["scores"] == [{"posting_index": 0, "x": 0}, {"posting_index": 1, "x": 1}, {"posting_index": 2, "x": 2}]


def test_merge_tool_use_blocks_list_merges_non_list_keys_too():
    block1 = _FakeContentBlock({"scores": [{"a": 1}], "other": "x"}, name="score_postings")
    block2 = _FakeContentBlock({"scores": [{"a": 2}]}, name="score_postings")
    response = _FakeResponse([block1, block2])
    merged = _merge_tool_use_blocks_list(response, "score_postings", "scores")
    assert merged["scores"] == [{"a": 1}, {"a": 2}]
    assert merged["other"] == "x"


def test_merge_tool_use_blocks_list_ignores_non_matching_blocks():
    full = [{"posting_index": 0, "x": 1}]
    other = _FakeContentBlock({"scores": [{"unrelated": True}]}, name="some_other_tool")
    response = _FakeResponse([other, _FakeContentBlock({"scores": full}, name="score_postings")])
    merged = _merge_tool_use_blocks_list(response, "score_postings", "scores")
    assert merged["scores"] == full


def test_merge_tool_use_blocks_list_raises_index_error_when_no_match():
    response = _FakeResponse([_FakeContentBlock({"scores": []}, name="unrelated_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks_list(response, "score_postings", "scores")


def test_merge_tool_use_blocks_list_raises_index_error_on_empty_content():
    response = _FakeResponse([])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks_list(response, "score_postings", "scores")


# --- _tier1_batch_max_tokens -------------------------------------------------

def test_tier1_batch_max_tokens_pilot_value():
    assert scoring_mod._tier1_batch_max_tokens(5) == 8192


def test_tier1_batch_max_tokens_scales_linearly():
    assert scoring_mod._tier1_batch_max_tokens(10) == 16384


def test_tier1_batch_max_tokens_floored_at_single_call_default():
    assert scoring_mod._tier1_batch_max_tokens(1) == 2048


# --- _acall_llm_batch --------------------------------------------------------

def test_acall_llm_batch_uses_batch_tool_and_scaled_max_tokens(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(5)]
    scores = [_batch_item_dict(i) for i in range(5)]
    fake_client = _FakeAsyncClient([_batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    kwargs = fake_client.messages.last_kwargs
    assert kwargs["tools"][0]["name"] == "score_postings"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "score_postings"}
    assert kwargs["max_tokens"] == 8192


def test_acall_llm_batch_maps_results_by_posting_index_out_of_order(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(3)]
    scores = [
        _batch_item_dict(2, cause_mission_fit=0.9),
        _batch_item_dict(0, cause_mission_fit=0.1),
        _batch_item_dict(1, cause_mission_fit=0.5),
    ]
    fake_client = _FakeAsyncClient([_batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert results[0].cause_mission_fit.score == 0.1
    assert results[1].cause_mission_fit.score == 0.5
    assert results[2].cause_mission_fit.score == 0.9


def test_acall_llm_batch_missing_index_is_none(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(3)]
    scores = [_batch_item_dict(0), _batch_item_dict(2)]  # index 1 omitted
    fake_client = _FakeAsyncClient([_batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


def test_acall_llm_batch_out_of_range_index_ignored_not_raised(monkeypatch, profile, rubric):
    postings = [make_posting(url="https://example.org/job/0", raw_text="text0")]
    scores = [_batch_item_dict(0), _batch_item_dict(99)]  # 99 is out of range for a 1-posting batch
    fake_client = _FakeAsyncClient([_batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert len(results) == 1
    assert results[0] is not None


def test_acall_llm_batch_duplicate_posting_index_last_write_wins_no_corruption(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(2)]
    scores = [
        _batch_item_dict(0, cause_mission_fit=0.1),
        _batch_item_dict(0, cause_mission_fit=0.9),  # duplicate index 0 — last one wins
        _batch_item_dict(1, cause_mission_fit=0.5),
    ]
    fake_client = _FakeAsyncClient([_batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert results[0].cause_mission_fit.score == 0.9
    assert results[1].cause_mission_fit.score == 0.5  # untouched by the duplicate at index 0


def test_acall_llm_batch_scores_split_across_multiple_blocks(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(2)]
    block1 = _FakeContentBlock({"scores": [_batch_item_dict(0)]}, name=scoring_mod._TIER1_BATCH_TOOL_NAME)
    block2 = _FakeContentBlock({"scores": [_batch_item_dict(1)]}, name=scoring_mod._TIER1_BATCH_TOOL_NAME)
    fake_client = _FakeAsyncClient([_FakeResponse([block1, block2])])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert results[0] is not None
    assert results[1] is not None


def test_acall_llm_batch_retries_on_transient_error(monkeypatch, profile, rubric):
    postings = [make_posting(raw_text="text")]
    scores = [_batch_item_dict(0)]
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(scoring_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    fake_client = _FakeAsyncClient([_overloaded_error(), _batch_response(scores)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results = asyncio.run(scoring_mod._acall_llm_batch(postings, profile, rubric))

    assert results[0] is not None
    assert fake_client.messages.call_count == 2


def test_acall_llm_batch_sets_usage_once_per_batch(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(3)]
    scores = [_batch_item_dict(i) for i in range(3)]
    fake_client = _FakeAsyncClient([
        _batch_response(scores, usage=_FakeUsage(input_tokens=900, output_tokens=90)),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    async def _call_and_read_usage():
        await scoring_mod._acall_llm_batch(postings, profile, rubric)
        return scoring_mod._last_usage.get()

    usage = asyncio.run(_call_and_read_usage())
    assert usage.calls == 1  # ONE batch call for 3 postings, not 3
    assert usage.input_tokens == 900
    assert usage.output_tokens == 90


# --- score_many_batched -------------------------------------------------------

def test_score_many_batched_empty_items_returns_empty():
    results, usage = asyncio.run(score_many_batched([], {}, {"dimensions": {}}))
    assert results == []
    assert usage == TokenUsage()


def test_score_many_batched_happy_path_preserves_order(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(6)]
    items = [(p, _PASSED_GATE) for p in postings]

    scores1 = [_batch_item_dict(i) for i in range(3)]
    scores2 = [_batch_item_dict(i) for i in range(3)]
    fake_client = _FakeAsyncClient([_batch_response(scores1), _batch_response(scores2)])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=3, max_concurrency=1))

    assert len(results) == 6
    assert [r.posting.url for r in results] == [p.url for p in postings]
    assert all(not isinstance(r, ScoringError) for r in results)


def test_score_many_batched_missing_index_falls_back_individually_others_keep_batch_result(
    monkeypatch, profile, rubric,
):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(3)]
    items = [(p, _PASSED_GATE) for p in postings]

    batch_scores = [
        _batch_item_dict(0, cause_mission_fit=0.11),
        # index 1 omitted -> falls back to individual scoring
        _batch_item_dict(2, cause_mission_fit=0.33),
    ]
    fake_client = _FakeAsyncClient([_batch_response(batch_scores), _valid_tier1_response()])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=3, max_concurrency=1))

    assert len(results) == 3
    assert not any(isinstance(r, ScoringError) for r in results)
    assert results[0].scores.cause_mission_fit.score == 0.11  # kept its batch result
    assert results[2].scores.cause_mission_fit.score == 0.33  # kept its batch result
    assert fake_client.messages.call_count == 2  # 1 batch + 1 individual fallback for index 1


def test_score_many_batched_whole_batch_failure_falls_back_all(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(2)]
    items = [(p, _PASSED_GATE) for p in postings]

    fake_client = _FakeAsyncClient([
        _FakeResponse([]),  # no matching tool_use block -> IndexError -> ValueError, whole batch fails
        _valid_tier1_response(),
        _valid_tier1_response(),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=2, max_concurrency=1))

    assert len(results) == 2
    assert not any(isinstance(r, ScoringError) for r in results)
    assert fake_client.messages.call_count == 3  # 1 failed batch attempt + 2 individual fallbacks


def test_score_many_batched_fallback_also_fails_becomes_scoring_error(monkeypatch, profile, rubric):
    """A posting missing from the batch response that ALSO fails its individual
    fallback (both attempts) becomes a ScoringError at its position — exactly as
    score_many() already does, not a new failure mode."""
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(2)]
    items = [(p, _PASSED_GATE) for p in postings]

    batch_scores = [_batch_item_dict(0)]  # index 1 omitted
    fake_client = _FakeAsyncClient([
        _batch_response(batch_scores),
        _bad_request_error(),  # individual fallback attempt 1 for index 1
        _bad_request_error(),  # individual fallback attempt 2 (ascore's one-shot retry)
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=2, max_concurrency=1))

    assert not isinstance(results[0], ScoringError)
    assert isinstance(results[1], ScoringError)


def test_score_many_batched_aggregates_usage_across_batches(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(4)]
    items = [(p, _PASSED_GATE) for p in postings]

    fake_client = _FakeAsyncClient([
        _batch_response([_batch_item_dict(i) for i in range(2)], usage=_FakeUsage(input_tokens=1000, output_tokens=100)),
        _batch_response([_batch_item_dict(i) for i in range(2)], usage=_FakeUsage(input_tokens=1200, output_tokens=120)),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=2, max_concurrency=1))

    assert usage.calls == 2  # 2 BATCH calls, not 4 (one per posting)
    assert usage.input_tokens == 2200
    assert usage.output_tokens == 220


def test_score_many_batched_aggregates_usage_including_fallback_calls(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(2)]
    items = [(p, _PASSED_GATE) for p in postings]

    fake_client = _FakeAsyncClient([
        _batch_response([_batch_item_dict(0)], usage=_FakeUsage(input_tokens=1000, output_tokens=100)),
        _FakeResponse(
            _FakeContentBlock(make_tier1().model_dump()),
            usage=_FakeUsage(input_tokens=50, output_tokens=20),
        ),
    ])
    monkeypatch.setattr(scoring_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(score_many_batched(items, profile, rubric, batch_size=2, max_concurrency=1))

    assert usage.calls == 2  # 1 batch call + 1 individual fallback call
    assert usage.input_tokens == 1050
    assert usage.output_tokens == 120


def test_score_many_batched_respects_max_concurrency(monkeypatch, profile, rubric):
    postings = [make_posting(url=f"https://example.org/job/{i}", raw_text=f"text{i}") for i in range(20)]
    items = [(p, _PASSED_GATE) for p in postings]

    max_concurrency = 3
    current = 0
    peak = 0

    async def _fake_acall_llm_batch(batch_postings, profile, rubric):  # noqa: ARG001
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return [make_tier1() for _ in batch_postings]

    monkeypatch.setattr(scoring_mod, "_acall_llm_batch", _fake_acall_llm_batch)

    asyncio.run(score_many_batched(items, profile, rubric, batch_size=2, max_concurrency=max_concurrency))

    assert peak <= max_concurrency
    assert peak > 1  # sanity: batches actually ran concurrently, not serially
