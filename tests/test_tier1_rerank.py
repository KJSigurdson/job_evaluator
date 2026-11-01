"""Tests for src/tier1_rerank.py: prompt content, response parsing, the
orchestration flow (batch submit -> parse/validate -> one retry -> give up),
and crash-safety (batch_submissions/batch_results lifecycle, reprocess_staged())
against a fake Anthropic Batches client and the existing FakeSupabaseClient."""
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
# Response parsing — operates on plain dicts (see module docstring)
# ---------------------------------------------------------------------------

def _block(input_data, *, type="tool_use", name="score_jobs") -> dict:  # noqa: A002
    return {"type": type, "name": name, "input": input_data}


def _message(blocks) -> dict:
    return {"content": blocks if isinstance(blocks, list) else [blocks]}


def test_merge_tool_use_blocks_single_block():
    full = _valid_output_dict()
    msg = _message(_block(full))
    assert _merge_tool_use_blocks(msg, "score_jobs") == full


def test_merge_tool_use_blocks_raises_when_no_match():
    msg = _message([_block({"x": 1}, name="other_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(msg, "score_jobs")


def test_parse_scores_valid():
    msg = _message(_block(_valid_output_dict(["job-1", "job-2"])))
    result = _parse_scores(msg)
    assert isinstance(result, Tier1RerankOutput)
    assert len(result.scores) == 2
    assert result.scores[0].job_id == "job-1"
    assert result.scores[0].score == 80


def test_parse_scores_invalid_raises_validation_error():
    msg = _message(_block({"scores": [{"job_id": "job-1", "score": 150, "reason": "x"}]}))
    with pytest.raises(ValidationError):  # score out of 0-100 range
        _parse_scores(msg)


# ---------------------------------------------------------------------------
# rerank_tier1() orchestration — fake Anthropic Batches client + FakeSupabaseClient
# ---------------------------------------------------------------------------

class _FakeSucceededResult:
    def __init__(self, blocks, usage=None):
        blocks = blocks if isinstance(blocks, list) else [blocks]
        self._dump = {
            "type": "succeeded",
            "message": {
                "content": blocks,
                "usage": usage or {"input_tokens": 1000, "output_tokens": 200},
            },
        }

    def model_dump(self, mode="json"):
        return self._dump


class _FakeErroredResult:
    def __init__(self, kind="errored"):
        self._dump = {"type": kind}

    def model_dump(self, mode="json"):
        return self._dump


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
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert "u1" in result
    assert result["u1"]["scores"][0].job_id == "job-1"
    assert len(anthropic_client.messages.batches.created_batches) == 1


def test_rerank_tier1_failure_then_successful_retry(fake_client):
    # Round 1's payload is missing required fields (score/reason) to force a retry.
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceededResult(_block({"scores": [{"job_id": "job-1"}]})))],  # missing score/reason
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert "u1" in result
    assert len(anthropic_client.messages.batches.created_batches) == 2
    retry_request = anthropic_client.messages.batches.created_batches[1][0]
    retry_prompt = retry_request["params"]["messages"][0]["content"]
    assert "previous attempt" in retry_prompt.lower()


def test_rerank_tier1_fails_both_rounds_excluded_from_result(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeErroredResult())],
        [_FakeResultItem("u1", _FakeErroredResult())],
    ])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert result == {}


def test_rerank_tier1_mixed_users_one_succeeds_one_fails(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("u-good", _FakeSucceededResult(_block(_valid_output_dict()))),
            _FakeResultItem("u-bad", _FakeErroredResult()),
        ],
        [_FakeResultItem("u-bad", _FakeErroredResult())],
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
        [_FakeResultItem("u1", _FakeSucceededResult(
            _block(_valid_output_dict()), usage={"input_tokens": 2000, "output_tokens": 300},
        ))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input())

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["tier1_calls"] == 1
    assert metrics["estimated_cost_usd"] > 0


def test_rerank_tier1_run_metrics_stamped_with_run_id_stage_trigger(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input(), run_id="run-abc", trigger="scheduled")

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["run_id"] == "run-abc"
    assert metrics["stage"] == "tier1"
    assert metrics["trigger"] == "scheduled"


# ---------------------------------------------------------------------------
# Crash-safety: batch_submissions / batch_results lifecycle
# ---------------------------------------------------------------------------

def test_rerank_tier1_records_batch_submission_and_retrieval(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input())

    [submission] = fake_client.table("batch_submissions").rows
    assert submission["module"] == "tier1_rerank"
    assert submission["status"] == "retrieved"


def test_rerank_tier1_persists_raw_results_before_parsing(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input())

    [staged] = fake_client.table("batch_results").rows
    assert staged["custom_id"] == "u1"
    assert staged["raw"]["type"] == "succeeded"


def test_rerank_tier1_marks_batch_results_processed_after_parsing(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1(fake_client, anthropic_client, users=_users_input())

    [staged] = fake_client.table("batch_results").rows
    assert staged["processed"] is True


def test_rerank_tier1_resumes_outstanding_batch_instead_of_resubmitting(fake_client):
    fake_client.seed("batch_submissions", [{"batch_id": "orphaned_batch", "module": "tier1_rerank", "status": "submitted"}])
    anthropic_client = _FakeAnthropicClient(rounds=[])

    class _ResumableBatchesAPI(_FakeBatchesAPI):
        def retrieve(self, batch_id):
            assert batch_id == "orphaned_batch"
            return _FakeBatch(batch_id)

        def results(self, batch_id):
            assert batch_id == "orphaned_batch"
            return [_FakeResultItem("u1", _FakeSucceededResult(_block(_valid_output_dict())))]

    anthropic_client.messages.batches = _ResumableBatchesAPI(rounds=[])

    result = rerank_tier1(fake_client, anthropic_client, users=_users_input())

    assert "u1" in result
    assert anthropic_client.messages.batches.created_batches == []


# ---------------------------------------------------------------------------
# reprocess_staged — recovery path, no API call
# ---------------------------------------------------------------------------

def test_reprocess_staged_no_unprocessed_rows_is_noop(fake_client):
    assert tier1_rerank.reprocess_staged(fake_client, "batch_1") == {}


def test_reprocess_staged_parses_without_api_call(fake_client):
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "u1",
        "raw": _FakeSucceededResult(_block(_valid_output_dict())).model_dump(),
        "processed": False,
    }])

    result = tier1_rerank.reprocess_staged(fake_client, "batch_1")

    assert "u1" in result
    assert result["u1"]["scores"][0].job_id == "job-1"


def test_reprocess_staged_marks_rows_processed(fake_client):
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "u1",
        "raw": _FakeSucceededResult(_block(_valid_output_dict())).model_dump(),
        "processed": False,
    }])

    tier1_rerank.reprocess_staged(fake_client, "batch_1")

    [row] = fake_client.table("batch_results").rows
    assert row["processed"] is True
    assert row["raw"] is not None


def test_reprocess_staged_does_not_write_run_metrics(fake_client):
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "u1",
        "raw": _FakeSucceededResult(_block(_valid_output_dict())).model_dump(),
        "processed": False,
    }])

    tier1_rerank.reprocess_staged(fake_client, "batch_1")

    assert fake_client.table("run_metrics").rows == []


def test_reprocess_staged_failure_excluded_from_result(fake_client):
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "u1",
        "raw": _FakeErroredResult().model_dump(),
        "processed": False,
    }])

    result = tier1_rerank.reprocess_staged(fake_client, "batch_1")

    assert result == {}
