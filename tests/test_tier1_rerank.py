"""Tests for src/tier1_rerank.py: prompt content, response parsing, and the
orchestration flow (batch submit -> parse/validate -> one retry -> give up) against
a fake Anthropic Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import tier1_rerank
from src.models.tier1_rerank import Tier1RerankOutput
from src.tier1_rerank import (
    _merge_tool_use_blocks,
    _parse_scores,
    build_prompt,
    build_retry_prompt,
    rerank_tier1,
)


def _profile() -> dict:
    return {
        "career_goals": "Move into data leadership at an EA-aligned org.",
        "cause_priorities": "Global health",
        "hard_constraints": {"location": {"accept_fully_remote": True}, "seniority": {"accept_levels": ["senior"]}},
    }


def _job_card(job_id="job-1", **overrides) -> dict:
    defaults = dict(
        id=job_id, title="Head of Data", org="GiveDirectly",
        enrichment={"summary": "Lead the data team.", "seniority": ["senior"],
                    "cause_area": ["global health"], "skill_tags": ["SQL", "Python"]},
    )
    defaults.update(overrides)
    return defaults


def _valid_output_dict(job_ids=("job-1",)) -> dict:
    return {"scores": [{"job_id": jid, "score": 80, "reason": "Strong cause and skill fit."} for jid in job_ids]}


# ---------------------------------------------------------------------------
# build_prompt / build_retry_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_excludes_location_seniority_comp_from_profile_view():
    profile = {**_profile(), "location": "Sundsvall, Sweden", "seniority_level": "Senior", "comp_needs": "70000 SEK"}
    prompt = build_prompt(profile=profile, gate_constraints=profile["hard_constraints"], job_cards=[_job_card()])
    assert "Sundsvall" not in prompt
    assert "70000 SEK" not in prompt
    assert "seniority_level" not in prompt


def test_build_prompt_includes_profile_and_job_cards():
    prompt = build_prompt(profile=_profile(), gate_constraints=_profile()["hard_constraints"],
                           job_cards=[_job_card()])
    assert "Head of Data" in prompt
    assert "GiveDirectly" in prompt
    assert "Lead the data team." in prompt
    assert "job_id: job-1" in prompt
    assert "=== JOB job-1 ===" in prompt


def test_build_prompt_instructs_not_to_reapply_hard_gate():
    prompt = build_prompt(profile=_profile(), gate_constraints=_profile()["hard_constraints"], job_cards=[_job_card()])
    assert "already passed" in prompt.lower()
    assert "do not re-check" in prompt.lower() or "do not re-apply" in prompt.lower() or "not re-check" in prompt.lower()


def test_build_prompt_instructs_independent_scoring():
    prompt = build_prompt(profile=_profile(), gate_constraints=_profile()["hard_constraints"], job_cards=[_job_card()])
    assert "independently" in prompt.lower()
    assert "not compare or" in prompt.lower() or "do not compare" in prompt.lower()


def test_build_prompt_includes_all_job_cards():
    cards = [_job_card("job-1"), _job_card("job-2"), _job_card("job-3")]
    prompt = build_prompt(profile=_profile(), gate_constraints={}, job_cards=cards)
    for jid in ["job-1", "job-2", "job-3"]:
        assert f"=== JOB {jid} ===" in prompt


def test_build_retry_prompt_appends_validation_error():
    prompt = build_retry_prompt(profile=_profile(), gate_constraints={}, job_cards=[_job_card()],
                                 validation_error="score: field required")
    assert "previous attempt" in prompt.lower()
    assert "score: field required" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class _FakeContentBlock:
    def __init__(self, input_data, *, type="tool_use", name="score_jobs"):  # noqa: A002
        self.input = input_data
        self.type = type
        self.name = name


class _FakeMessage:
    def __init__(self, blocks):
        self.content = blocks if isinstance(blocks, list) else [blocks]
        self.stop_reason = "tool_use"


def test_merge_tool_use_blocks_single_block():
    full = _valid_output_dict()
    msg = _FakeMessage(_FakeContentBlock(full))
    assert _merge_tool_use_blocks(msg, "score_jobs") == full


def test_merge_tool_use_blocks_raises_when_no_match():
    msg = _FakeMessage([_FakeContentBlock({"x": 1}, name="other_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(msg, "score_jobs")


def test_parse_scores_valid():
    msg = _FakeMessage(_FakeContentBlock(_valid_output_dict(["job-1", "job-2"])))
    result = _parse_scores(msg)
    assert isinstance(result, Tier1RerankOutput)
    assert len(result.scores) == 2
    assert result.scores[0].job_id == "job-1"
    assert result.scores[0].score == 80


def test_parse_scores_invalid_raises_validation_error():
    msg = _FakeMessage(_FakeContentBlock({"scores": [{"job_id": "job-1", "score": 150, "reason": "x"}]}))
    with pytest.raises(ValidationError):  # score out of 0-100 range
        _parse_scores(msg)


# ---------------------------------------------------------------------------
# rerank_tier1() orchestration — fake Anthropic Batches client + FakeSupabaseClient
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, input_tokens=1000, output_tokens=200):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeSucceeded:
    type = "succeeded"

    def __init__(self, blocks, usage=None):
        self.message = _FakeMessage(blocks)
        self.message.usage = usage or _FakeUsage()


class _FakeErrored:
    def __init__(self, kind="errored"):
        self.type = kind


class _FakeResultItem:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _FakeBatch:
    def __init__(self, id, processing_status="ended"):  # noqa: A002
        self.id = id
        self.processing_status = processing_status


class _FakeBatchesAPI:
    def __init__(self, rounds: list[list]):
        self._rounds = list(rounds)
        self._counter = 0
        self._results_by_batch: dict[str, list] = {}
        self.created_batches: list[list] = []

    def create(self, requests):
        self._counter += 1
        batch_id = f"batch_{self._counter}"
        self.created_batches.append(list(requests))
        self._results_by_batch[batch_id] = self._rounds.pop(0)
        return _FakeBatch(batch_id)

    def retrieve(self, batch_id):
        return _FakeBatch(batch_id)

    def results(self, batch_id):
        return self._results_by_batch[batch_id]


class _FakeMessagesAPI:
    def __init__(self, batches_api):
        self.batches = batches_api


class _FakeAnthropicClient:
    def __init__(self, rounds: list[list]):
        self.messages = _FakeMessagesAPI(_FakeBatchesAPI(rounds))


def _users_input(user_ids=("u1",)) -> dict:
    return {
        uid: {"profile": _profile(), "gate_constraints": _profile()["hard_constraints"], "job_cards": [_job_card()]}
        for uid in user_ids
    }


def test_rerank_tier1_happy_path_all_succeed_round_one(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceeded(_FakeContentBlock(_valid_output_dict())))],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert "u1" in result
    assert result["u1"]["scores"][0].job_id == "job-1"
    assert len(anthropic_client.messages.batches.created_batches) == 1


def test_rerank_tier1_failure_then_successful_retry(fake_client):
    # Round 1's payload is missing required fields (score/reason) to force a retry.
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceeded(_FakeContentBlock({"scores": [{"job_id": "job-1"}]})))],  # missing score/reason
        [_FakeResultItem("u1", _FakeSucceeded(_FakeContentBlock(_valid_output_dict())))],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert "u1" in result
    assert len(anthropic_client.messages.batches.created_batches) == 2
    retry_request = anthropic_client.messages.batches.created_batches[1][0]
    retry_prompt = retry_request["params"]["messages"][0]["content"]
    assert "previous attempt" in retry_prompt.lower()


def test_rerank_tier1_fails_both_rounds_excluded_from_result(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeErrored())],
        [_FakeResultItem("u1", _FakeErrored())],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert result == {}


def test_rerank_tier1_mixed_users_one_succeeds_one_fails(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("u-good", _FakeSucceeded(_FakeContentBlock(_valid_output_dict()))),
            _FakeResultItem("u-bad", _FakeErrored()),
        ],
        [_FakeResultItem("u-bad", _FakeErrored())],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input(["u-good", "u-bad"]))

    assert "u-good" in result
    assert "u-bad" not in result


def test_rerank_tier1_empty_users_returns_empty_no_batch_call(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[])

    result = rerank_tier1(fake_client, anthropic_client, users={})

    assert result == {}
    assert anthropic_client.messages.batches.created_batches == []
    assert fake_client.table("run_metrics").rows == []


def test_rerank_tier1_writes_run_metrics_with_cost_from_actual_usage(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceeded(
            _FakeContentBlock(_valid_output_dict()), usage=_FakeUsage(input_tokens=2000, output_tokens=300),
        ))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input())

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["tier1_calls"] == 1
    assert metrics["estimated_cost_usd"] > 0
