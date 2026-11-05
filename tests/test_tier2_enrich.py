"""Tests for src/tier2_enrich.py: prompt content, response parsing (incl. v1's
defensive list-field coercion), the orchestration flow (threshold filter ->
batch submit -> parse/validate -> one retry -> give up), and crash-safety
(context-wrapped batch_results staging, reprocess_staged()) against a fake
Anthropic Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import tier2_enrich
from src.batch_poll import validate_custom_id
from src.schemas import Tier2EnrichmentOutput
from src.tier2_enrich import (
    _batch_request,
    _coerce_list_field,
    _merge_tool_use_blocks,
    _parse_tier2,
    build_prompt,
    build_retry_prompt,
    fetch_candidates,
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


# ---------------------------------------------------------------------------
# _batch_request / custom_id — custom_id here is a plain enumeration index
# (str(i)), always valid under the Batches API's ^[a-zA-Z0-9_-]{1,64}$
# charset, but validate_custom_id() is still called at request-build time
# (both the live AND retry paths funnel through this one function) — see
# src.tier1_rerank's own f"{user_id}:{chunk_index}" incident.
# ---------------------------------------------------------------------------

def test_batch_request_accepts_an_index_custom_id():
    request = _batch_request("12", "prompt text")
    validate_custom_id(request["custom_id"])  # must not raise


def test_batch_request_raises_on_invalid_custom_id():
    with pytest.raises(ValueError, match="custom_id"):
        _batch_request("has a space", "prompt text")


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


def test_tier2_enrich_fractional_threshold_rounds_up_not_down(fake_client):
    """threshold=72.5 (e.g. a 0.725 rubric scaled by 100) must behave as if
    it were 73, not 72 — a candidate scoring exactly 72 must NOT qualify.
    This is a pure Python comparison (no DB query), so it wouldn't crash
    without the ceil, but it must still agree with fetch_candidates()'s own
    ceil'd DB filter rather than silently using a different effective bar."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])
    candidates = [_candidate(job_id="job-73", tier1_score=73), _candidate(job_id="job-72", tier1_score=72)]

    result = run_tier2_enrich(fake_client, anthropic_client, candidates=candidates, threshold=72.5)

    assert result["considered"] == 2
    assert result["enriched"] == 1
    assert result["skipped_below_threshold"] == 1
    assert len(anthropic_client.messages.batches.created_batches[0]) == 1  # only job-73 submitted


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


def test_tier2_enrich_upsert_payload_never_includes_status(fake_client):
    """matches_v2.status is frontend-owned (proposed/applied/screening/.../
    withdrawn) — this upsert's on_conflict=user_id,job_id means ANY key in
    this payload gets written on a conflict too, not just on a fresh
    insert, so 'status' must NEVER appear here: a crash-recovery re-apply
    of the exact same (user_id, job_id) pair must not reset a status the
    user has since changed via the frontend back to whatever this payload
    would say. New rows get their status entirely from the column's own
    DEFAULT (the migration's job, not this code's), confirmed 2026-11-04
    after matches_v2.status turned out to already exist live with no
    migration file ever capturing it."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])

    [payload] = fake_client.table("matches_v2").upsert_calls
    assert "status" not in payload[0]


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


def test_tier2_enrich_survives_run_metrics_insert_failure(fake_client, caplog):
    """insert_run_metrics() never raises — a schema-drift failure writing
    the observability row must not undo or block on matches_v2 rows already
    written above it."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])
    fake_client.table("run_metrics").raise_on_execute["insert"] = Exception("PGRST204: boom")

    with caplog.at_level("ERROR"):
        result = run_tier2_enrich(fake_client, anthropic_client, candidates=[_candidate()])  # must not raise

    assert result["enriched"] == 1
    assert fake_client.table("matches_v2").rows  # the real write survived
    assert fake_client.table("run_metrics").rows == []
    assert any(r.levelname == "ERROR" for r in caplog.records)


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


def test_reprocess_staged_upsert_payload_never_includes_status(fake_client):
    """Same invariant as the live path's sibling test — the replay path
    must be equally safe against resetting a real user's status on a
    recovery re-apply."""
    fake_client.seed("batch_results", [_staged_row()])

    tier2_enrich.reprocess_staged(fake_client, "batch_1")

    [payload] = fake_client.table("matches_v2").upsert_calls
    assert "status" not in payload[0]


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


# ---------------------------------------------------------------------------
# fetch_candidates() — reads tier1_verdicts directly (Phase 2); valid,
# >= threshold, not already matched. Null profile_content_hash on either
# side must always be treated as stale, never a match.
# ---------------------------------------------------------------------------

def _verdict(job_id="job-1", user_id="u1", score=80, job_enriched_at="t1",
             profile_content_hash="hash-abc", reason="good fit") -> dict:
    return {
        "user_id": user_id, "job_id": job_id, "score": score,
        "rationale": {"reason": reason},
        "job_enriched_at": job_enriched_at, "profile_content_hash": profile_content_hash,
    }


def test_fetch_candidates_returns_valid_above_threshold_verdict(fake_client):
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=80)])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])

    candidates = fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=65)

    assert len(candidates) == 1
    assert candidates[0]["tier1_score"] == 80
    assert candidates[0]["tier1_reason"] == "good fit"
    assert candidates[0]["job"]["id"] == "job-1"


def test_fetch_candidates_excludes_below_threshold(fake_client):
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=40)])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])

    assert fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=65) == []


@pytest.mark.parametrize("threshold,expected_int", [(65.0, 65), (70.0, 70), (75.0, 75), (72.5, 73)])
def test_fetch_candidates_sends_an_int_to_the_score_filter(fake_client, threshold, expected_int):
    """Regression for a real incident: tier1_verdicts.score is an INTEGER
    column, and a fractional threshold (user.rubric["threshold"] * 100,
    e.g. 0.725 * 100 = 72.5) sent as a .gte() filter value gets stringified
    into the request URL as "72.5" — Postgres rejects that outright trying
    to parse it as an integer (22P02 invalid input syntax for type integer).
    65.0/70.0/75.0 are the exact production scoring_weights.insert_threshold
    values (0.65/0.7/0.75 * 100) confirmed live in an earlier turn; 72.5 is
    the fractional case ceil() exists for (0.725 * 100 -> 73, rounding UP
    so ">=" semantics against an integer column are preserved exactly)."""
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=expected_int)])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])

    fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=threshold)

    [(col, value)] = fake_client.table("tier1_verdicts").gte_calls
    assert col == "score"
    assert value == expected_int
    assert isinstance(value, int) and not isinstance(value, bool)


def test_fetch_candidates_excludes_already_matched_job(fake_client):
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=80)])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])
    fake_client.seed("matches_v2", [{"user_id": "u1", "job_id": "job-1"}])

    assert fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=65) == []


def test_fetch_candidates_excludes_stale_job_enriched_at(fake_client):
    """Job re-enriched since this verdict was scored — the verdict no longer
    means anything, must not surface as a Tier 2 candidate."""
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=80, job_enriched_at="t1")])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t2"}])  # re-enriched since

    assert fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=65) == []


def test_fetch_candidates_excludes_profile_hash_mismatch(fake_client):
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=80, profile_content_hash="hash-OLD")])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])

    assert fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-NEW", threshold=65) == []


def test_fetch_candidates_excludes_deleted_job(fake_client):
    fake_client.seed("tier1_verdicts", [_verdict("job-deleted", score=80)])

    assert fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash="hash-abc", threshold=65) == []


@pytest.mark.parametrize("verdict_hash,current_hash", [
    (None, "hash-abc"),  # verdict's own hash unknown
    ("hash-abc", None),  # caller's current hash unknown (profile not yet hashed)
    (None, None),        # both unknown — None != None would otherwise look like "matches"
])
def test_fetch_candidates_null_profile_hash_is_always_invalid(fake_client, caplog, verdict_hash, current_hash):
    fake_client.seed("tier1_verdicts", [_verdict("job-1", score=80, profile_content_hash=verdict_hash)])
    fake_client.seed("jobs", [{**_job("job-1"), "enriched_at": "t1"}])

    with caplog.at_level("WARNING"):
        candidates = fetch_candidates(fake_client, user_id="u1", profile=_profile(), profile_content_hash=current_hash, threshold=65)

    assert candidates == []
    assert any("null profile_content_hash" in r.message for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)
