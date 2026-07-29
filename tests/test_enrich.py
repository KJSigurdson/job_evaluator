"""Tests for enrich.py's response-parsing helpers — pure, no Anthropic API calls.

Covers the emphasize_in_cv/deemphasize list[str] guarantee: these must survive
parsing as real lists, including the defensive case where a model returns a
JSON-array-in-a-string instead of a native array. Also covers the multi-block
tool-response merge fix and the concurrent enrich_many path.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from anthropic import BadRequestError, OverloadedError, RateLimitError
from pydantic import ValidationError

from src import enrich as enrich_mod
from src.enrich import EnrichmentError, _coerce_list_field, _merge_tool_use_blocks, _parse_enrichment_output
from src.schemas import HardGateResult, ScoredPosting, Tier2EnrichmentOutput
from tests.conftest import make_posting, make_tier1

_PASSED_GATE = HardGateResult(passed=True, location_pass=True, seniority_pass=True)


def _tool_input(**overrides) -> dict:
    defaults = dict(
        org_summary="A charity evaluator.",
        why_fits="Strong mission alignment.",
        why_not_fits="No fundraising experience.",
        emphasize_in_cv=["BI leadership"],
        deemphasize=["IT portfolio work"],
    )
    defaults.update(overrides)
    return defaults


def _make_scored(url: str = "https://example.org/job/123") -> ScoredPosting:
    return ScoredPosting(
        posting=make_posting(url=url), gate=_PASSED_GATE, scores=make_tier1(), fit_score=0.8,
    )


# ---------------------------------------------------------------------------
# _coerce_list_field
# ---------------------------------------------------------------------------

def test_coerce_list_field_passes_through_real_list():
    assert _coerce_list_field(["a", "b"]) == ["a", "b"]


def test_coerce_list_field_parses_json_array_string():
    assert _coerce_list_field('["a", "b"]') == ["a", "b"]


def test_coerce_list_field_empty_json_array_string():
    assert _coerce_list_field("[]") == []


def test_coerce_list_field_leaves_malformed_string_unchanged():
    assert _coerce_list_field("not json at all") == "not json at all"


def test_coerce_list_field_leaves_non_array_json_string_unchanged():
    # A valid JSON string that parses to a non-list (e.g. an object or scalar) is not
    # our case to fix — leave it for normal list[str] validation to reject clearly.
    assert _coerce_list_field('{"a": 1}') == '{"a": 1}'
    assert _coerce_list_field("42") == "42"


def test_coerce_list_field_passes_through_non_string_non_list():
    assert _coerce_list_field(None) is None


# ---------------------------------------------------------------------------
# _parse_enrichment_output — end-to-end through Tier2EnrichmentOutput validation
# ---------------------------------------------------------------------------

def test_parse_enrichment_output_native_arrays_stay_lists():
    result = _parse_enrichment_output(_tool_input())
    assert result.emphasize_in_cv == ["BI leadership"]
    assert result.deemphasize == ["IT portfolio work"]
    assert isinstance(result.emphasize_in_cv, list)
    assert isinstance(result.deemphasize, list)


def test_parse_enrichment_output_coerces_json_array_string_to_list():
    tool_input = _tool_input(
        emphasize_in_cv='["BI leadership", "CRM ownership"]',
        deemphasize='["IT portfolio work"]',
    )
    result = _parse_enrichment_output(tool_input)
    assert result.emphasize_in_cv == ["BI leadership", "CRM ownership"]
    assert isinstance(result.emphasize_in_cv, list)
    assert result.deemphasize == ["IT portfolio work"]
    assert isinstance(result.deemphasize, list)


def test_parse_enrichment_output_empty_json_array_string_becomes_empty_list():
    tool_input = _tool_input(emphasize_in_cv="[]")
    result = _parse_enrichment_output(tool_input)
    assert result.emphasize_in_cv == []


def test_parse_enrichment_output_malformed_string_still_raises():
    tool_input = _tool_input(emphasize_in_cv="not a json array")
    with pytest.raises(ValidationError):
        _parse_enrichment_output(tool_input)


def test_parse_enrichment_output_other_fields_untouched():
    result = _parse_enrichment_output(_tool_input(org_summary="X"))
    assert result.org_summary == "X"


# ---------------------------------------------------------------------------
# Fake Anthropic response plumbing — shared by the merge-block and async tests below
# ---------------------------------------------------------------------------

class _FakeContentBlock:
    def __init__(self, input_data, *, type="tool_use", name="enrich_posting"):  # noqa: A002
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


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _overloaded_error() -> OverloadedError:
    """529 overloaded_error — transient Anthropic-side capacity condition, a distinct
    class from RateLimitError (429) that the pre-fix backoff loop didn't catch."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=529, request=request)
    return OverloadedError("overloaded", response=response, body=None)


def _bad_request_error() -> BadRequestError:
    """400 invalid_request — permanent; must fail fast, not retry through the loop."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError("invalid_request", response=response, body=None)


# ---------------------------------------------------------------------------
# _merge_tool_use_blocks — multi-block tool-response parsing (regression: a model
# occasionally splits one response across several tool_use blocks instead of one
# block with every key)
# ---------------------------------------------------------------------------

def test_merge_tool_use_blocks_single_block():
    full = _tool_input()
    response = _FakeResponse(_FakeContentBlock(full))
    assert _merge_tool_use_blocks(response, "enrich_posting") == full


def test_merge_tool_use_blocks_merges_multiple_blocks():
    full = _tool_input()
    blocks = [_FakeContentBlock({k: v}) for k, v in full.items()]
    response = _FakeResponse(blocks)

    merged = _merge_tool_use_blocks(response, "enrich_posting")

    assert merged == full


def test_merge_tool_use_blocks_ignores_non_matching_blocks():
    full = _tool_input()
    other_tool_block = _FakeContentBlock({"unrelated": "data"}, name="some_other_tool")
    response = _FakeResponse([other_tool_block, _FakeContentBlock(full)])

    assert _merge_tool_use_blocks(response, "enrich_posting") == full


def test_merge_tool_use_blocks_ignores_non_tool_use_blocks():
    full = _tool_input()
    text_block = _FakeContentBlock("some preamble text", type="text", name=None)
    response = _FakeResponse([text_block, _FakeContentBlock(full)])

    assert _merge_tool_use_blocks(response, "enrich_posting") == full


def test_merge_tool_use_blocks_raises_index_error_when_no_match():
    response = _FakeResponse([_FakeContentBlock({"x": 1}, name="unrelated_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(response, "enrich_posting")


def test_merge_tool_use_blocks_raises_index_error_on_empty_content():
    response = _FakeResponse([])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(response, "enrich_posting")


def test_acall_llm_succeeds_with_response_split_across_multiple_blocks(monkeypatch):
    """End-to-end regression for the real bug: a Tier 2 response split across
    several single-key tool_use blocks must still validate successfully via
    _acall_llm."""
    full = _tool_input()
    split_blocks = [_FakeContentBlock({k: v}) for k, v in full.items()]
    fake_client = _FakeAsyncClient([_FakeResponse(split_blocks)])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(enrich_mod._acall_llm(_make_scored(), {}))

    assert result.org_summary == full["org_summary"]
    assert result.emphasize_in_cv == full["emphasize_in_cv"]


# ---------------------------------------------------------------------------
# Rate-limit backoff — same pattern as scoring.py's _acall_llm, duplicated; one
# sanity test to confirm the wiring, not the full suite (already covered there).
# ---------------------------------------------------------------------------

def test_acall_llm_succeeds_after_rate_limit_retry(monkeypatch):
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    fake_client = _FakeAsyncClient([
        _rate_limit_error(), _FakeResponse(_FakeContentBlock(_tool_input())),
    ])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(enrich_mod._acall_llm(_make_scored(), {}))

    assert result is not None
    assert fake_client.messages.call_count == 2


def test_acall_llm_succeeds_after_overloaded_retries(monkeypatch):
    """529s are now retried by the same backoff loop as 429s."""
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    fake_client = _FakeAsyncClient([
        _overloaded_error(), _overloaded_error(), _FakeResponse(_FakeContentBlock(_tool_input())),
    ])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    result = asyncio.run(enrich_mod._acall_llm(_make_scored(), {}))

    assert result is not None
    assert fake_client.messages.call_count == 3  # 2 overloaded + 1 success


def test_persistent_overloaded_surfaces_as_enrichment_error(monkeypatch):
    """A 529 that survives _acall_llm's backoff is just a normal exception to
    aenrich's one-shot retry — no new error type; it still becomes EnrichmentError."""
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_MAX_ATTEMPTS", 2)

    # 2 attempts per _acall_llm call, 2 calls total (initial + one retry) = 4 errors.
    fake_client = _FakeAsyncClient([_overloaded_error() for _ in range(4)])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(EnrichmentError):
        asyncio.run(enrich_mod.aenrich(_make_scored(), {}))

    assert fake_client.messages.call_count == 4


def test_acall_llm_bad_request_not_retried_by_backoff(monkeypatch):
    """A 400 is permanent — the backoff loop re-raises it on the first attempt
    (call_count == 1), never consuming further attempts or the valid response."""
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(enrich_mod, "_RATE_LIMIT_MAX_DELAY_SECONDS", 0.001)

    fake_client = _FakeAsyncClient([_bad_request_error(), _FakeResponse(_FakeContentBlock(_tool_input()))])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    with pytest.raises(BadRequestError):
        asyncio.run(enrich_mod._acall_llm(_make_scored(), {}))

    assert fake_client.messages.call_count == 1


# ---------------------------------------------------------------------------
# Token usage aggregation — audit-only, additive to enrich_many's return value.
# Mirrors scoring.py's identical test coverage; see that file's comment on why
# set-and-read must happen within the same asyncio Task.
# ---------------------------------------------------------------------------

def test_acall_llm_sets_usage_from_real_response(monkeypatch):
    fake_client = _FakeAsyncClient([_FakeResponse(
        _FakeContentBlock(_tool_input()),
        usage=_FakeUsage(input_tokens=123, output_tokens=45, cache_read_input_tokens=7, cache_creation_input_tokens=9),
    )])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    async def _call_and_read_usage():
        await enrich_mod._acall_llm(_make_scored(), {})
        return enrich_mod._last_usage.get()

    usage = asyncio.run(_call_and_read_usage())
    assert usage.calls == 1
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45
    assert usage.cache_read_tokens == 7
    assert usage.cache_creation_tokens == 9


def test_acall_llm_usage_defaults_cache_fields_to_zero_when_none(monkeypatch):
    fake_client = _FakeAsyncClient([_FakeResponse(
        _FakeContentBlock(_tool_input()),
        usage=_FakeUsage(cache_read_input_tokens=None, cache_creation_input_tokens=None),
    )])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    async def _call_and_read_usage():
        await enrich_mod._acall_llm(_make_scored(), {})
        return enrich_mod._last_usage.get()

    usage = asyncio.run(_call_and_read_usage())
    assert usage.cache_read_tokens == 0
    assert usage.cache_creation_tokens == 0


def test_enrich_many_aggregates_usage_across_batch(monkeypatch):
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(3)]

    fake_client = _FakeAsyncClient([
        _FakeResponse(_FakeContentBlock(_tool_input()), usage=_FakeUsage(input_tokens=100, output_tokens=10)),
        _FakeResponse(_FakeContentBlock(_tool_input()), usage=_FakeUsage(input_tokens=200, output_tokens=20)),
        _FakeResponse(_FakeContentBlock(_tool_input()), usage=_FakeUsage(input_tokens=300, output_tokens=30)),
    ])
    monkeypatch.setattr(enrich_mod, "AsyncAnthropic", lambda: fake_client)

    results, usage = asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=1))

    assert len(results) == 3
    assert usage.calls == 3
    assert usage.input_tokens == 600
    assert usage.output_tokens == 60


def test_enrich_many_enrichment_error_contributes_zero_usage(monkeypatch):
    """A batch with one persistently-failing item: that item's EnrichmentError
    contributes no usage, but other items' usage still aggregates correctly —
    mirrors enrich_many's existing error-isolation test pattern."""
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(3)]

    async def _fake_acall_llm(scored, profile):  # noqa: ARG001
        if scored.posting.url.endswith("/job/1"):
            raise ValueError("simulated LLM failure")  # exhausts retry -> EnrichmentError, no usage set
        return Tier2EnrichmentOutput(**_tool_input())

    monkeypatch.setattr(enrich_mod, "_acall_llm", _fake_acall_llm)

    results, usage = asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=1))

    assert isinstance(results[1], EnrichmentError)
    assert usage.calls == 0
    assert usage.input_tokens == 0


# ---------------------------------------------------------------------------
# enrich_many — concurrent Tier 2 enrichment (mirrors scoring.py's score_many
# test coverage exactly)
# ---------------------------------------------------------------------------

def test_enrich_many_preserves_input_order(monkeypatch):
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(20)]

    async def _fake_acall_llm(scored, profile):  # noqa: ARG001
        return Tier2EnrichmentOutput(**_tool_input(org_summary=scored.posting.url))

    monkeypatch.setattr(enrich_mod, "_acall_llm", _fake_acall_llm)

    results, _usage = asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=5))
    assert [r.org_summary for r in results] == [item.posting.url for item in items]


def test_enrich_many_enrichment_error_does_not_abort_others(monkeypatch):
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(5)]

    async def _fake_acall_llm(scored, profile):  # noqa: ARG001
        if scored.posting.url.endswith("/job/2"):
            raise ValueError("simulated LLM failure")  # exhausts aenrich's retry -> EnrichmentError
        return Tier2EnrichmentOutput(**_tool_input())

    monkeypatch.setattr(enrich_mod, "_acall_llm", _fake_acall_llm)

    results, _usage = asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=5))

    assert len(results) == 5
    for i, result in enumerate(results):
        if i == 2:
            assert isinstance(result, EnrichmentError)
        else:
            assert not isinstance(result, EnrichmentError)


def test_enrich_many_non_enrichment_error_propagates_and_aborts(monkeypatch):
    """A non-EnrichmentError exception is a real bug, not an expected parse/rate-limit
    failure — enrich_many must not swallow it. Patches aenrich() itself (rather than
    _acall_llm, which aenrich's retry would convert to EnrichmentError either way) to
    exercise enrich_many's own catch-only-EnrichmentError contract directly."""
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(3)]

    async def _fake_aenrich(scored, profile):  # noqa: ARG001
        if scored.posting.url.endswith("/job/1"):
            raise RuntimeError("a real bug, not an enrichment failure")
        return Tier2EnrichmentOutput(**_tool_input())

    monkeypatch.setattr(enrich_mod, "aenrich", _fake_aenrich)

    with pytest.raises(RuntimeError, match="a real bug"):
        asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=5))


def test_enrich_many_respects_max_concurrency(monkeypatch):
    items = [_make_scored(url=f"https://example.org/job/{i}") for i in range(20)]

    max_concurrency = 3
    current = 0
    peak = 0

    async def _fake_acall_llm(scored, profile):  # noqa: ARG001
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)  # hold the slot briefly so overlaps are observable
        current -= 1
        return Tier2EnrichmentOutput(**_tool_input())

    monkeypatch.setattr(enrich_mod, "_acall_llm", _fake_acall_llm)

    asyncio.run(enrich_mod.enrich_many(items, {}, max_concurrency=max_concurrency))

    assert peak <= max_concurrency
    assert peak > 1  # sanity: concurrency actually happened, this isn't accidentally serial
