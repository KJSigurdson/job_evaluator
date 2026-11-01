"""Tests for src/tier2_enrich.py: prompt content, response parsing (incl. v1's
defensive list-field coercion), the orchestration flow (threshold filter ->
batch submit -> parse/validate -> one retry -> give up), and crash-safety
(context-wrapped batch_results staging, reprocess_staged()) against a fake
Anthropic Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import tier2_enrich
from src.schemas import Tier2EnrichmentOutput
from src.tier2_enrich import (
    _coerce_list_field,
    _merge_tool_use_blocks,
    _parse_tier2,
    build_prompt,
    build_retry_prompt,
    tier2_enrich as run_tier2_enrich,
)


def _profile() -> dict:
    return {
        "career_goals": "Move into data leadership at an EA-aligned org.",
        "experience_inventory": "Work: Data leader at OrgA",
        "hard_constraints": {"location": {}, "seniority": {}},
    }


def _job(job_id="job-1", **overrides) -> dict:
    defaults = dict(
        id=job_id, title="Head of Data", org="GiveDirectly", raw_description="Full posting text here.",
        enrichment={"summary": "Lead the data team.", "seniority": ["senior"],
                    "cause_area": ["global health"], "skill_tags": ["SQL", "Python"],
                    "work_arrangement": {"mode": "remote", "remote_scope": "global"}},
    )
    defaults.update(overrides)
    return defaults


def _valid_output_dict(**overrides) -> dict:
    defaults = dict(
        org_summary="GiveDirectly provides direct cash transfers.",
        why_fits="- Strong data leadership background\n- EA-aligned cause priorities",
        why_not_fits="- No prior direct-cash-transfer experience",
        emphasize_in_cv=["Led data & analytics function"],
        deemphasize=["Airline industry background"],
    )
    defaults.update(overrides)
    return defaults


def _candidate(user_id="u1", job_id="job-1", tier1_score=80, **overrides) -> dict:
    defaults = dict(
        user_id=user_id, profile=_profile(), job=_job(job_id), tier1_score=tier1_score,
        tier1_reason="Strong cause and skill fit.",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# build_prompt / build_retry_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_profile_and_job():
    prompt = build_prompt(profile=_profile(), job=_job(), tier1_score=80, tier1_reason="Great fit.")
    assert "Head of Data" in prompt
    assert "GiveDirectly" in prompt
    assert "Lead the data team." in prompt
    assert "80/100" in prompt
    assert "Great fit." in prompt


def test_build_prompt_includes_format_rules_for_bullets_and_phrases():
    prompt = build_prompt(profile=_profile(), job=_job(), tier1_score=80, tier1_reason="x")
    assert "- \"" in prompt or '"- "' in prompt
    assert "no markdown headers" in prompt.lower()
    assert "3-6 bullets" in prompt


def test_build_prompt_includes_full_untrimmed_profile():
    """Unlike tier1_rerank's prompt, Tier 2 doesn't strip location/seniority_level/
    comp_needs — matches v1's src/enrich.py, which yaml.dumps the whole profile."""
    profile = {**_profile(), "location": "Sundsvall, Sweden", "comp_needs": "70000 SEK"}
    prompt = build_prompt(profile=profile, job=_job(), tier1_score=80, tier1_reason="x")
    assert "Sundsvall" in prompt
    assert "70000 SEK" in prompt


def test_build_retry_prompt_appends_validation_error():
    prompt = build_retry_prompt(profile=_profile(), job=_job(), tier1_score=80, tier1_reason="x",
                                 validation_error="why_fits: field required")
    assert "previous attempt" in prompt.lower()
    assert "why_fits: field required" in prompt


# ---------------------------------------------------------------------------
# Response parsing — operates on plain dicts (see module docstring)
# ---------------------------------------------------------------------------

def _block(input_data, *, type="tool_use", name="enrich_posting") -> dict:  # noqa: A002
    return {"type": type, "name": name, "input": input_data}


def _message(blocks) -> dict:
    return {"content": blocks if isinstance(blocks, list) else [blocks]}


def test_merge_tool_use_blocks_single_block():
    full = _valid_output_dict()
    msg = _message(_block(full))
    assert _merge_tool_use_blocks(msg, "enrich_posting") == full


def test_merge_tool_use_blocks_raises_when_no_match():
    msg = _message([_block({"x": 1}, name="other_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(msg, "enrich_posting")


def test_parse_tier2_valid():
    msg = _message(_block(_valid_output_dict()))
    result = _parse_tier2(msg)
    assert isinstance(result, Tier2EnrichmentOutput)
    assert result.org_summary == _valid_output_dict()["org_summary"]


def test_parse_tier2_invalid_raises_validation_error():
    msg = _message(_block({"org_summary": "x"}))  # missing required fields
    with pytest.raises(ValidationError):
        _parse_tier2(msg)


def test_coerce_list_field_parses_json_array_in_string():
    assert _coerce_list_field('["a", "b"]') == ["a", "b"]


def test_coerce_list_field_passes_through_real_list():
    assert _coerce_list_field(["a", "b"]) == ["a", "b"]


def test_coerce_list_field_passes_through_malformed_string_unchanged():
    assert _coerce_list_field("not json") == "not json"


def test_parse_tier2_coerces_stringified_list_fields():
    payload = _valid_output_dict(emphasize_in_cv='["Led data function"]', deemphasize='["Airline background"]')
    msg = _message(_block(payload))
    result = _parse_tier2(msg)
    assert result.emphasize_in_cv == ["Led data function"]
    assert result.deemphasize == ["Airline background"]


# ---------------------------------------------------------------------------
# tier2_enrich() orchestration — fake Anthropic Batches client + FakeSupabaseClient
# ---------------------------------------------------------------------------

class _FakeSucceededResult:
    def __init__(self, blocks, usage=None):
        blocks = blocks if isinstance(blocks, list) else [blocks]
        self._dump = {
            "type": "succeeded",
            "message": {
                "content": blocks,
                "usage": usage or {"input_tokens": 1500, "output_tokens": 400},
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


def test_tier2_enrich_happy_path_all_succeed_round_one(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    assert result == {"considered": 1, "enriched": 1, "skipped_below_threshold": 0, "failed": 0}
    row = fake_client.table("matches_v2").rows[0]
    assert row["user_id"] == "u1"
    assert row["job_id"] == "job-1"
    assert row["tier1_score"] == 80
    assert row["tier2"]["org_summary"] == _valid_output_dict()["org_summary"]


def test_tier2_enrich_custom_id_stays_under_64_chars_with_real_uuid_length_ids(fake_client):
    """Regression: the Batches API rejects any custom_id over 64 characters.
    Two real UUIDs joined ("{user_id}:{job_id}") are 73 chars and get rejected
    outright — confirmed for real against production. custom_id must be
    short (index-based) regardless of how long user_id/job_id are."""
    long_user_id = "fd5eb2df-a9cf-4a62-9907-6bd98c8c9600"
    long_job_id = "87cd3639-4164-4110-a20c-04fc3f1b7d5a"
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])
    candidates = [_candidate(user_id=long_user_id, job_id=long_job_id)]

    run_tier2_enrich(fake_client, anthropic_client, candidates=candidates)

    [request] = anthropic_client.messages.batches.created_batches[0]
    assert len(request["custom_id"]) <= 64


def test_tier2_enrich_skips_below_threshold_candidates(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])
    candidates = [_candidate(job_id="job-1", tier1_score=80), _candidate(job_id="job-2", tier1_score=40)]

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=candidates, threshold=65)

    assert result["considered"] == 2
    assert result["enriched"] == 1
    assert result["skipped_below_threshold"] == 1
    assert len(anthropic_client.messages.batches.created_batches[0]) == 1  # only the qualifying one submitted


def test_tier2_enrich_no_qualifying_candidates_makes_no_batch_call(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[])
    candidates = [_candidate(tier1_score=10)]

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=candidates, threshold=65)

    assert result == {"considered": 1, "enriched": 0, "skipped_below_threshold": 1, "failed": 0}
    assert anthropic_client.messages.batches.created_batches == []
    assert fake_client.table("matches_v2").rows == []
    assert fake_client.table("run_metrics").rows == []


def test_tier2_enrich_failure_then_successful_retry(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block({"org_summary": "x"})))],  # missing fields
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    assert result["enriched"] == 1
    assert len(anthropic_client.messages.batches.created_batches) == 2
    retry_request = anthropic_client.messages.batches.created_batches[1][0]
    retry_prompt = retry_request["params"]["messages"][0]["content"]
    assert "previous attempt" in retry_prompt.lower()


def test_tier2_enrich_fails_both_rounds_never_written(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeErroredResult())],
        [_FakeResultItem("0", _FakeErroredResult())],
    ])

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    assert result["enriched"] == 0
    assert result["failed"] == 1
    assert fake_client.table("matches_v2").rows == []


def test_tier2_enrich_upserts_on_user_id_job_id_conflict(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    [payload] = fake_client.table("matches_v2").upsert_calls
    assert payload[0]["user_id"] == "u1"
    assert payload[0]["job_id"] == "job-1"


def test_tier2_enrich_writes_run_metrics_with_cost_from_actual_usage(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(
            _block(_valid_output_dict()), usage={"input_tokens": 3000, "output_tokens": 600},
        ))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["tier2_calls"] == 1
    assert metrics["estimated_cost_usd"] > 0


def test_tier2_enrich_run_metrics_stamped_with_run_id_stage_trigger(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()], run_id="run-abc", trigger="scheduled")

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["run_id"] == "run-abc"
    assert metrics["stage"] == "tier2"
    assert metrics["trigger"] == "scheduled"


def test_tier2_enrich_multiple_users_same_job_distinct_custom_ids(fake_client):
    """custom_id is index-based (see tier2_enrich()'s by_key), not derived from
    user_id/job_id — the Batches API caps custom_id at 64 characters, and two
    UUIDs joined together exceeds that (confirmed for real against production).
    This test's real point: two DIFFERENT users matched to the SAME job_id must
    still produce two independent matches_v2 rows, not collide."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict()))),
            _FakeResultItem("1", _FakeSucceededResult(_block(_valid_output_dict()))),
        ],
    ])
    candidates = [_candidate(user_id="u1", job_id="job-1"), _candidate(user_id="u2", job_id="job-1")]

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=candidates)

    assert result["enriched"] == 2
    rows_by_user = {r["user_id"]: r for r in fake_client.table("matches_v2").rows}
    assert set(rows_by_user) == {"u1", "u2"}


def test_tier2_enrich_run_metrics_written_even_if_final_write_fails(fake_client, monkeypatch):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(
            _block(_valid_output_dict()), usage={"input_tokens": 3000, "output_tokens": 600},
        ))],
    ])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated statement timeout")

    monkeypatch.setattr(tier2_enrich, "chunked_upsert", _boom)

    with pytest.raises(RuntimeError):
        run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["estimated_cost_usd"] > 0


# ---------------------------------------------------------------------------
# Crash-safety: batch_submissions / batch_results lifecycle (context-wrapped)
# ---------------------------------------------------------------------------

def test_tier2_enrich_records_batch_submission_and_retrieval(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    [submission] = fake_client.table("batch_submissions").rows
    assert submission["module"] == "tier2_enrich"
    assert submission["status"] == "retrieved"


def test_tier2_enrich_stages_context_alongside_raw_result(fake_client):
    """Unlike enrich_jobs.py/tier1_rerank.py, this module's custom_id is an
    anonymous index and can't identify the (user, job) pair on its own — the
    staged raw value must carry that context for recovery to work at all."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate(user_id="u1", job_id="job-1")])

    [staged] = fake_client.table("batch_results").rows
    assert staged["raw"]["llm_result"]["type"] == "succeeded"


def test_tier2_enrich_marks_batch_results_processed_after_matches_v2_write(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate(user_id="u1", job_id="job-1")])

    [staged] = fake_client.table("batch_results").rows
    assert staged["processed"] is True
    [match] = fake_client.table("matches_v2").rows
    assert match["user_id"] == "u1" and match["job_id"] == "job-1"
    assert staged["raw"]["context"]["user_id"] == "u1"
    assert staged["raw"]["context"]["job"]["id"] == "job-1"
    assert staged["raw"]["context"]["tier1_score"] == 80


def test_tier2_enrich_resumes_outstanding_batch_instead_of_resubmitting(fake_client):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    fake_client.seed("batch_submissions", [{
        "batch_id": "orphaned_batch", "module": "tier2_enrich", "status": "submitted",
        "created_at": now, "updated_at": now,
    }])
    anthropic_client = _FakeAnthropicClient(rounds=[])

    class _ResumableBatchesAPI(_FakeBatchesAPI):
        def retrieve(self, batch_id):
            assert batch_id == "orphaned_batch"
            return _FakeBatch(batch_id)

        def results(self, batch_id):
            assert batch_id == "orphaned_batch"
            return [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))]

    anthropic_client.messages.batches = _ResumableBatchesAPI(rounds=[])

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    assert result["enriched"] == 1
    assert anthropic_client.messages.batches.created_batches == []


# ---------------------------------------------------------------------------
# reprocess_staged — recovery path, no API call
# ---------------------------------------------------------------------------

def _staged_row(batch_id="batch_1", custom_id="0", *, context=None, llm_result=None, processed=False) -> dict:
    return {
        "batch_id": batch_id, "custom_id": custom_id,
        "raw": {
            "llm_result": llm_result or _FakeSucceededResult(_block(_valid_output_dict())).model_dump(),
            "context": context if context is not None else {
                "user_id": "u1", "job": _job(), "tier1_score": 80, "tier1_reason": "Great fit.", "profile": _profile(),
            },
        },
        "processed": processed,
    }


def test_reprocess_staged_no_unprocessed_rows_is_noop(fake_client):
    result = tier2_enrich.reprocess_staged(fake_client, "batch_1")
    assert result == {"selected": 0, "enriched": 0, "failed": 0}


def test_reprocess_staged_parses_and_writes_without_api_call(fake_client):
    fake_client.seed("batch_results", [_staged_row()])

    result = tier2_enrich.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 1, "failed": 0}
    row = fake_client.table("matches_v2").rows[0]
    assert row["user_id"] == "u1"
    assert row["job_id"] == "job-1"
    assert row["tier1_score"] == 80
    assert row["tier2"]["org_summary"] == _valid_output_dict()["org_summary"]


def test_reprocess_staged_marks_rows_processed(fake_client):
    fake_client.seed("batch_results", [_staged_row()])

    tier2_enrich.reprocess_staged(fake_client, "batch_1")

    [row] = fake_client.table("batch_results").rows
    assert row["processed"] is True
    assert row["raw"] is not None


def test_reprocess_staged_failure_marks_failed_not_written(fake_client):
    fake_client.seed("batch_results", [
        _staged_row(llm_result=_FakeErroredResult().model_dump()),
    ])

    result = tier2_enrich.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 0, "failed": 1}
    assert fake_client.table("matches_v2").rows == []


def test_reprocess_staged_does_not_write_run_metrics(fake_client):
    fake_client.seed("batch_results", [_staged_row()])

    tier2_enrich.reprocess_staged(fake_client, "batch_1")

    assert fake_client.table("run_metrics").rows == []


def test_reprocess_staged_only_processes_that_batch_id(fake_client):
    fake_client.seed("batch_results", [
        _staged_row(batch_id="batch_1", custom_id="0", context={
            "user_id": "u1", "job": _job("job-1"), "tier1_score": 80, "tier1_reason": "x", "profile": _profile(),
        }),
        _staged_row(batch_id="batch_2", custom_id="0", context={
            "user_id": "u2", "job": _job("job-2"), "tier1_score": 80, "tier1_reason": "x", "profile": _profile(),
        }),
    ])

    result = tier2_enrich.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 1, "failed": 0}
    [row] = fake_client.table("matches_v2").rows
    assert row["user_id"] == "u1"


def test_reprocess_staged_missing_context_logged_and_skipped(fake_client):
    """A staged row with no context (shouldn't happen in practice, but must not
    crash recovery) is marked processed and counted as failed, not written."""
    fake_client.seed("batch_results", [_staged_row(context=None)])
    # Overwrite to actually remove the context key entirely (context=None above
    # still sets the key to None via the dict, which .get() also treats as falsy).
    fake_client.table("batch_results").rows[0]["raw"]["context"] = None

    result = tier2_enrich.reprocess_staged(fake_client, "batch_1")

    assert result["enriched"] == 0
    assert result["failed"] == 1
    assert fake_client.table("batch_results").rows[0]["processed"] is True
