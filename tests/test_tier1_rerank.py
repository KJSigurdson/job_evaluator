"""Tests for src/tier1_rerank.py: prompt content, response parsing, chunking,
verdict writing/validity, and the single-batch-per-run orchestration (batch
submit -> parse/validate -> one retry -> drop) against a fake Anthropic
Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src import tier1_rerank
from src.batch_poll import validate_custom_id
from src.models.tier1_rerank import Tier1RerankOutput
from src.tier1_rerank import (
    TIER1_MAX_PER_USER,
    _batch_request,
    _build_manifest,
    _chunk_user_requests,
    _merge_tool_use_blocks,
    _parse_scores,
    _write_verdicts,
    apply_drained_batch,
    build_prompt,
    build_retry_prompt,
    fetch_valid_verdict_job_ids,
    reprocess_staged,
    rerank_tier1_batch,
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
        enriched_at="2026-10-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return defaults


def _seed_manifest(fake_client, batch_id, *, entries=None, jobs=None, user_id="u1",
                    profile_content_hash="hash-abc", custom_id="u1_0", module="tier1_rerank"):
    """Seed a batch_submissions row carrying the request-time manifest
    _write_verdicts/apply_drained_batch/reprocess_staged read via
    fetch_batch_manifest — i.e. simulate "this is what was persisted
    alongside the batch at submission time", independent of whatever `jobs`/
    `profiles` currently say. entries, if given, overrides the single
    custom_id/user_id/jobs shortcut with an arbitrary {custom_id: {...}} dict."""
    manifest = entries if entries is not None else {
        custom_id: {"user_id": user_id, "profile_content_hash": profile_content_hash, "jobs": jobs or {}},
    }
    fake_client.seed("batch_submissions", [{
        "batch_id": batch_id, "module": module, "status": "retrieved", "manifest": manifest,
    }])


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
# Response parsing — operates on plain dicts
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
# _chunk_user_requests — custom_id = f"{user_id}_{chunk_index}", at most
# _CHUNK_SIZE (40) job cards per chunk
# ---------------------------------------------------------------------------

def test_chunk_user_requests_single_chunk_under_cap():
    user_requests = {"u1": {"profile": _profile(), "gate_constraints": {}, "job_cards": [_job_card("job-1")]}}
    chunks = _chunk_user_requests(user_requests)
    assert list(chunks.keys()) == ["u1_0"]
    assert chunks["u1_0"][0] == "u1"
    assert len(chunks["u1_0"][1]) == 1


def test_chunk_user_requests_splits_at_40():
    cards = [_job_card(f"job-{i}") for i in range(85)]
    user_requests = {"u1": {"profile": _profile(), "gate_constraints": {}, "job_cards": cards}}
    chunks = _chunk_user_requests(user_requests)
    assert set(chunks.keys()) == {"u1_0", "u1_1", "u1_2"}
    assert len(chunks["u1_0"][1]) == 40
    assert len(chunks["u1_1"][1]) == 40
    assert len(chunks["u1_2"][1]) == 5


def test_chunk_user_requests_multiple_users_namespaced_independently():
    user_requests = {
        "u1": {"profile": _profile(), "gate_constraints": {}, "job_cards": [_job_card("job-1")]},
        "u2": {"profile": _profile(), "gate_constraints": {}, "job_cards": [_job_card("job-2")]},
    }
    chunks = _chunk_user_requests(user_requests)
    assert set(chunks.keys()) == {"u1_0", "u2_0"}
    assert chunks["u1_0"][0] == "u1"
    assert chunks["u2_0"][0] == "u2"


def test_chunk_user_requests_double_digit_chunk_index_is_a_valid_custom_id():
    """Regression for the real incident: a custom_id separator that happens
    to work for single-digit chunk indices can still break the Batches
    API's charset — ':' did, in fact. 450 cards -> chunk indices 0..11, so
    this exercises double digits too, and every single produced custom_id
    must pass the API's own ^[a-zA-Z0-9_-]{1,64}$ constraint, not just a
    hand-picked few."""
    cards = [_job_card(f"job-{i}") for i in range(450)]
    user_requests = {"u1": {"profile": _profile(), "gate_constraints": {}, "job_cards": cards}}

    chunks = _chunk_user_requests(user_requests)

    assert set(chunks.keys()) == {f"u1_{i}" for i in range(12)}  # ceil(450/40) = 12
    assert "u1_10" in chunks and "u1_11" in chunks
    for custom_id in chunks:
        validate_custom_id(custom_id)  # must not raise for ANY produced id


def test_batch_request_uses_underscore_separator_not_colon():
    """The specific incident: f"{user_id}:{chunk_index}" (colon) made the
    WHOLE batch .create() call fail with a 400 invalid_request_error, not
    just the one bad request."""
    request = _batch_request("u1_0", "prompt text")
    assert request["custom_id"] == "u1_0"
    assert ":" not in request["custom_id"]


def test_batch_request_raises_on_invalid_custom_id():
    """_batch_request() is the one chokepoint both the live AND retry paths
    funnel through (rerank_tier1_batch calls it for round 1 and round 2
    alike) — validating here, at request-build time, covers both without
    needing a separate check in the retry code path."""
    with pytest.raises(ValueError, match="custom_id"):
        _batch_request("u1:0", "prompt text")


def test_tier1_max_per_user_is_200():
    assert TIER1_MAX_PER_USER == 200


# ---------------------------------------------------------------------------
# _build_manifest — the request-time snapshot persisted alongside a batch
# submission (see src.batch_poll.resume_or_submit's manifest param)
# ---------------------------------------------------------------------------

def test_build_manifest_captures_job_enriched_at_and_profile_hash():
    user_requests = {
        "u1": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-abc",
               "job_cards": [_job_card("job-1", enriched_at="t1"), _job_card("job-2", enriched_at="t2")]},
    }
    chunks = _chunk_user_requests(user_requests)

    manifest = _build_manifest(chunks, user_requests)

    assert manifest == {"u1_0": {"user_id": "u1", "profile_content_hash": "hash-abc", "jobs": {"job-1": "t1", "job-2": "t2"}}}


def test_build_manifest_namespaces_each_chunk_independently():
    user_requests = {
        "u1": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-u1",
               "job_cards": [_job_card("job-1", enriched_at="t1")]},
        "u2": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-u2",
               "job_cards": [_job_card("job-2", enriched_at="t2")]},
    }
    chunks = _chunk_user_requests(user_requests)

    manifest = _build_manifest(chunks, user_requests)

    assert manifest["u1_0"]["profile_content_hash"] == "hash-u1"
    assert manifest["u2_0"]["profile_content_hash"] == "hash-u2"


# ---------------------------------------------------------------------------
# _write_verdicts — job_enriched_at/profile_content_hash come from the
# batch's own REQUEST-TIME manifest (fetch_batch_manifest), never a fresh
# `jobs`/`profiles` read at write time.
# ---------------------------------------------------------------------------

def _score(job_id="job-1", score=80, reason="good fit"):
    from src.models.tier1_rerank import JobScore
    return JobScore(job_id=job_id, score=score, reason=reason)


def test_write_verdicts_upserts_using_batch_manifest(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})

    written = _write_verdicts(fake_client, {"u1": [_score("job-1", 80, "good fit")]}, "batch_1")

    assert written == 1
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["user_id"] == "u1"
    assert row["job_id"] == "job-1"
    assert row["score"] == 80
    assert row["rationale"] == {"reason": "good fit"}
    assert row["job_enriched_at"] == "2026-10-01T00:00:00+00:00"
    assert row["profile_content_hash"] == "hash-abc"
    assert row["batch_id"] == "batch_1"


def test_write_verdicts_ignores_current_jobs_table_entirely(fake_client):
    """The defining behavior change: even if `jobs` has a DIFFERENT (newer)
    enriched_at for this job right now, the verdict must record what the
    manifest says was sent, not current state. See also
    test_apply_drained_batch_uses_request_time_manifest_not_current_job_state
    for the end-to-end drainer version of this same guarantee."""
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-20T00:00:00+00:00"}])

    _write_verdicts(fake_client, {"u1": [_score("job-1", 80, "good fit")]}, "batch_1")

    [row] = fake_client.table("tier1_verdicts").rows
    assert row["job_enriched_at"] == "2026-10-01T00:00:00+00:00"


def test_write_verdicts_writes_below_threshold_scores_too(fake_client):
    """Every score is upserted, not just ones clearing a threshold — Tier 2
    threshold filtering reads this table later, it doesn't gate writes here."""
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})

    _write_verdicts(fake_client, {"u1": [_score("job-1", 12, "poor fit")]}, "batch_1")

    [row] = fake_client.table("tier1_verdicts").rows
    assert row["score"] == 12


def test_write_verdicts_skips_score_missing_from_manifest(fake_client):
    """A score whose job_id isn't in this batch's manifest at all (shouldn't
    happen — the manifest is built from the same job_cards the request was
    built from) is skipped rather than guessed at from current `jobs` state."""
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})

    written = _write_verdicts(fake_client, {"u1": [_score("some-other-job")]}, "batch_1")

    assert written == 0
    assert fake_client.table("tier1_verdicts").rows == []


def test_write_verdicts_no_manifest_for_batch_skips_everything(fake_client):
    """A batch_id with no batch_submissions row at all (or no manifest
    column set) yields no verdicts — never falls back to current `jobs`/
    `profiles` state."""
    written = _write_verdicts(fake_client, {"u1": [_score("job-1")]}, "batch_unknown")

    assert written == 0
    assert fake_client.table("tier1_verdicts").rows == []


def test_write_verdicts_noop_on_empty_input(fake_client):
    assert _write_verdicts(fake_client, {}, "batch_1") == 0


# ---------------------------------------------------------------------------
# fetch_valid_verdict_job_ids — valid iff job_enriched_at AND
# profile_content_hash both still match current state
# ---------------------------------------------------------------------------

def test_fetch_valid_verdict_job_ids_valid_when_both_match(fake_client):
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-1", "job_enriched_at": "2026-10-01T00:00:00+00:00",
        "profile_content_hash": "hash-abc",
    }])
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-01T00:00:00+00:00"}])

    assert fetch_valid_verdict_job_ids(fake_client, "u1", "hash-abc") == {"job-1"}


def test_fetch_valid_verdict_job_ids_invalid_when_job_reenriched(fake_client):
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-1", "job_enriched_at": "2026-10-01T00:00:00+00:00",
        "profile_content_hash": "hash-abc",
    }])
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-15T00:00:00+00:00"}])  # re-enriched since

    assert fetch_valid_verdict_job_ids(fake_client, "u1", "hash-abc") == set()


def test_fetch_valid_verdict_job_ids_invalid_when_profile_changed(fake_client):
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-1", "job_enriched_at": "2026-10-01T00:00:00+00:00",
        "profile_content_hash": "hash-abc",
    }])
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-01T00:00:00+00:00"}])

    assert fetch_valid_verdict_job_ids(fake_client, "u1", "hash-NEW") == set()


def test_fetch_valid_verdict_job_ids_empty_when_no_verdicts(fake_client):
    assert fetch_valid_verdict_job_ids(fake_client, "u1", "hash-abc") == set()


@pytest.mark.parametrize("verdict_hash,current_hash", [
    (None, "hash-abc"),  # verdict's own stored hash is unknown
    ("hash-abc", None),  # caller's current hash is unknown (profile not yet hashed)
    (None, None),        # both unknown — None != None would otherwise look like "matches"
])
def test_fetch_valid_verdict_job_ids_null_hash_is_always_invalid(fake_client, caplog, verdict_hash, current_hash):
    fake_client.seed("tier1_verdicts", [{
        "user_id": "u1", "job_id": "job-1", "job_enriched_at": "2026-10-01T00:00:00+00:00",
        "profile_content_hash": verdict_hash,
    }])
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-01T00:00:00+00:00"}])

    with caplog.at_level("WARNING"):
        result = fetch_valid_verdict_job_ids(fake_client, "u1", current_hash)

    assert result == set()
    assert any("null profile_content_hash" in r.message for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)


# ---------------------------------------------------------------------------
# rerank_tier1_batch() orchestration — fake Anthropic Batches client (same
# "always ended immediately" fake as test_enrich_jobs.py, compatible with
# src.batch_poll.resume_or_submit) + FakeSupabaseClient
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


class _FakeResultItem:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _FakeBatch:
    def __init__(self, id, processing_status="ended"):  # noqa: A002
        self.id = id
        self.processing_status = processing_status


class _FakeBatchesAPI:
    """*rounds* is a list of "result item lists" — one per expected .create()
    call, in order. Each .create() call reports processing_status='ended'
    immediately, matching src.batch_poll's "ended on first retrieve" happy
    path (no real polling loop exercised here)."""

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


def _user_requests(user_ids=("u1",), job_cards=None) -> dict:
    return {
        uid: {"profile": _profile(), "gate_constraints": _profile()["hard_constraints"],
              "profile_content_hash": "hash-abc", "job_cards": job_cards or [_job_card()]}
        for uid in user_ids
    }


def test_rerank_tier1_batch_happy_path_single_user_single_chunk(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())

    assert result == {"status": "complete", "scored": 1, "failed": 0}
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["user_id"] == "u1"
    assert row["job_id"] == "job-1"
    # From the request-time manifest built off _job_card()'s/_user_requests()'s
    # own defaults — NOT a DB read (there's no `jobs`/`profiles` seeded at all
    # in this test).
    assert row["job_enriched_at"] == "2026-10-01T00:00:00+00:00"
    assert row["profile_content_hash"] == "hash-abc"
    assert len(anthropic_client.messages.batches.created_batches) == 1  # ONE batch for everyone


def test_rerank_tier1_batch_covers_multiple_users_in_one_batch(fake_client):
    user_requests = {
        "u1": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-u1",
               "job_cards": [_job_card("job-1")]},
        "u2": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-u2",
               "job_cards": [_job_card("job-2")]},
    }
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict(["job-1"])))),
            _FakeResultItem("u2_0", _FakeSucceededResult(_block(_valid_output_dict(["job-2"])))),
        ],
    ])

    result = rerank_tier1_batch(fake_client, anthropic_client, user_requests=user_requests)

    assert result["scored"] == 2
    [single_batch] = anthropic_client.messages.batches.created_batches
    assert len(single_batch) == 2  # both users' chunks in the SAME batch
    user_ids = {row["user_id"] for row in fake_client.table("tier1_verdicts").rows}
    assert user_ids == {"u1", "u2"}


def test_rerank_tier1_batch_multiple_chunks_for_one_user_in_one_batch(fake_client):
    cards = [_job_card("job-1", enriched_at="t1"), _job_card("job-2", enriched_at="t1")]
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict(["job-1"])))),
            _FakeResultItem("u1_1", _FakeSucceededResult(_block(_valid_output_dict(["job-2"])))),
        ],
    ])

    result = rerank_tier1_batch(
        fake_client, anthropic_client,
        user_requests={"u1": {"profile": _profile(), "gate_constraints": {}, "profile_content_hash": "hash-abc", "job_cards": cards}},
    )
    # Force two chunks by monkeypatching _CHUNK_SIZE would be cleaner, but the
    # test above (test_chunk_user_requests_splits_at_40) already covers the
    # actual chunk-size boundary; here we just confirm two results merge into
    # one user's verdict set correctly within a single batch.
    assert result["scored"] == 2


def test_rerank_tier1_batch_empty_user_requests_still_checks_for_outstanding(fake_client):
    """Spec F: a user with zero unscored candidates contributes nothing to
    user_requests — but the batch call must still happen (not short-circuit)
    so an outstanding batch from a prior crashed run can still be resumed."""
    anthropic_client = _FakeAnthropicClient(rounds=[])

    result = rerank_tier1_batch(fake_client, anthropic_client, user_requests={})

    assert result == {"status": "complete", "scored": 0, "failed": 0}
    assert anthropic_client.messages.batches.created_batches == []


def test_rerank_tier1_batch_failure_then_successful_retry(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block({"scores": [{"job_id": "job-1", "score": 999, "reason": "bad"}]})))],
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    result = rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())

    assert result == {"status": "complete", "scored": 1, "failed": 0}
    assert len(anthropic_client.messages.batches.created_batches) == 2
    # The retry batch's own request must carry a VALID custom_id too — the
    # retry path funnels through the same _batch_request() as round 1 (see
    # test_batch_request_raises_on_invalid_custom_id), but this confirms it
    # end-to-end through the real retry call site.
    [retry_batch] = anthropic_client.messages.batches.created_batches[1:]
    for request in retry_batch:
        validate_custom_id(request["custom_id"])
    # The retried chunk's verdict must carry the RETRY batch's own batch_id
    # (and therefore its own manifest) — not round 1's, whose chunk for this
    # custom_id is the one that failed to parse.
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["batch_id"] == "batch_2"
    assert row["job_enriched_at"] == "2026-10-01T00:00:00+00:00"


def test_rerank_tier1_batch_fails_both_rounds_dropped_not_partial(fake_client):
    bad_block = _block({"scores": [{"job_id": "job-1", "score": 999, "reason": "bad"}]})
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(bad_block))],
        [_FakeResultItem("u1_0", _FakeSucceededResult(bad_block))],
    ])

    result = rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())

    assert result == {"status": "complete", "scored": 0, "failed": 1}
    assert fake_client.table("tier1_verdicts").rows == []


def test_rerank_tier1_batch_records_batch_submission_and_retrieval(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())

    [submission] = fake_client.table("batch_submissions").rows
    assert submission["module"] == "tier1_rerank"
    assert submission["status"] == "retrieved"
    # The request-time manifest is persisted alongside the submission, not
    # derived at write time — see _build_manifest/record_batch_submitted.
    assert submission["manifest"] == {"u1_0": {"user_id": "u1", "profile_content_hash": "hash-abc", "jobs": {"job-1": "2026-10-01T00:00:00+00:00"}}}


def test_rerank_tier1_batch_persists_raw_results_and_marks_processed(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())

    [staged] = fake_client.table("batch_results").rows
    assert staged["custom_id"] == "u1_0"
    assert staged["raw"]["type"] == "succeeded"
    assert staged["processed"] is True


def test_rerank_tier1_batch_writes_run_metrics(fake_client):
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])

    rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests(), run_id="run-abc", trigger="scheduled")

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["stage"] == "tier1"
    assert metrics["run_id"] == "run-abc"
    assert metrics["trigger"] == "scheduled"
    assert metrics["tier1_calls"] == 1
    assert metrics["estimated_cost_usd"] > 0


def test_rerank_tier1_batch_survives_run_metrics_insert_failure(fake_client, caplog):
    """insert_run_metrics() never raises — a schema-drift failure writing
    the observability row must not undo or block on the verdicts already
    written above it (see src.supabase_client.insert_run_metrics's own
    tests for the real incident this guards)."""
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("u1_0", _FakeSucceededResult(_block(_valid_output_dict())))],
    ])
    fake_client.table("run_metrics").raise_on_execute["insert"] = Exception("PGRST204: boom")

    with caplog.at_level("ERROR"):
        result = rerank_tier1_batch(fake_client, anthropic_client, user_requests=_user_requests())  # must not raise

    assert result == {"status": "complete", "scored": 1, "failed": 0}
    assert fake_client.table("tier1_verdicts").rows  # the real write survived
    assert fake_client.table("run_metrics").rows == []
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_rerank_tier1_batch_pending_writes_zero_count_run_metrics_and_stops(fake_client, monkeypatch):
    from src.batch_poll import BatchOutcome

    monkeypatch.setattr(
        tier1_rerank, "resume_or_submit",
        lambda *a, **k: BatchOutcome(status="pending", batch_id="batch_x", request_counts={"processing": 1}),
    )

    result = rerank_tier1_batch(fake_client, anthropic_client=object(), user_requests=_user_requests())

    assert result["status"] == "pending"
    assert result["batch_id"] == "batch_x"
    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["stage"] == "tier1"
    assert metrics["tier1_calls"] == 0
    assert fake_client.table("tier1_verdicts").rows == []


# ---------------------------------------------------------------------------
# apply_drained_batch() — drainer-only, verdicts ONLY, no retry
# ---------------------------------------------------------------------------

def test_apply_drained_batch_writes_verdicts(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    results = {"u1_0": _FakeSucceededResult(_block(_valid_output_dict())).model_dump()}

    result = apply_drained_batch(fake_client, results, batch_id="batch_1")

    assert result == {"scored": 1, "failed": 0}
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["user_id"] == "u1"
    assert row["batch_id"] == "batch_1"


def test_apply_drained_batch_uses_request_time_manifest_not_current_job_state(fake_client):
    """The core correctness fix this change exists for: the drainer applies
    results in a SEPARATE process, potentially hours after submission (the
    daily pipeline submitted the batch; its own poll budget ran out; the
    hourly drainer picks it up later). If job-1 gets re-enriched in that
    window, the verdict must still record what the LLM actually scored
    against (the OLD, request-time enriched_at from the manifest) — not
    whatever `jobs` says right now. And because job_enriched_at then no
    longer matches jobs.enriched_at, fetch_valid_verdict_job_ids must treat
    the resulting verdict as INVALID, not accidentally valid."""
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})  # request-time (OLD)
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-20T00:00:00+00:00"}])  # re-enriched since (current)
    results = {"u1_0": _FakeSucceededResult(_block(_valid_output_dict())).model_dump()}

    result = apply_drained_batch(fake_client, results, batch_id="batch_1")

    assert result == {"scored": 1, "failed": 0}
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["job_enriched_at"] == "2026-10-01T00:00:00+00:00"  # from the manifest, not current jobs.enriched_at

    assert fetch_valid_verdict_job_ids(fake_client, "u1", "hash-abc") == set()


def test_apply_drained_batch_parse_failure_dropped_no_retry(fake_client):
    bad = _FakeSucceededResult(_block({"scores": [{"job_id": "job-1", "score": 999, "reason": "bad"}]})).model_dump()

    result = apply_drained_batch(fake_client, {"u1_0": bad}, batch_id="batch_1")

    assert result == {"scored": 0, "failed": 1}


def test_apply_drained_batch_marks_processed(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    fake_client.seed("batch_results", [{"batch_id": "batch_1", "custom_id": "u1_0", "raw": {}, "processed": False}])
    results = {"u1_0": _FakeSucceededResult(_block(_valid_output_dict())).model_dump()}

    apply_drained_batch(fake_client, results, batch_id="batch_1")

    [row] = fake_client.table("batch_results").rows
    assert row["processed"] is True


def test_apply_drained_batch_no_manifest_skips_verdicts_instead_of_guessing(fake_client):
    """A batch with no manifest persisted (shouldn't happen for tier1_rerank
    going forward, but possible for one submitted before the manifest column
    existed) must never fall back to reading current `jobs`/`profiles` —
    that's exactly the write-time-state bug this change fixes."""
    fake_client.seed("jobs", [{"id": "job-1", "enriched_at": "2026-10-01T00:00:00+00:00"}])
    fake_client.seed("profiles", [{"user_id": "u1", "profile_content_hash": "hash-abc"}])
    results = {"u1_0": _FakeSucceededResult(_block(_valid_output_dict())).model_dump()}

    result = apply_drained_batch(fake_client, results, batch_id="batch_no_manifest")

    assert result == {"scored": 0, "failed": 0}
    assert fake_client.table("tier1_verdicts").rows == []


# ---------------------------------------------------------------------------
# reprocess_staged() — same-process recovery, no new API call, no retry
# ---------------------------------------------------------------------------

def test_reprocess_staged_no_unprocessed_rows_is_noop(fake_client):
    assert reprocess_staged(fake_client, "batch_1") == {"selected": 0, "scored": 0, "failed": 0}


def test_reprocess_staged_writes_verdicts_without_api_call(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    raw = _FakeSucceededResult(_block(_valid_output_dict())).model_dump()
    fake_client.seed("batch_results", [{"batch_id": "batch_1", "custom_id": "u1_0", "raw": raw, "processed": False}])

    result = reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "scored": 1, "failed": 0}
    [row] = fake_client.table("tier1_verdicts").rows
    assert row["user_id"] == "u1"


def test_reprocess_staged_does_not_write_run_metrics(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    raw = _FakeSucceededResult(_block(_valid_output_dict())).model_dump()
    fake_client.seed("batch_results", [{"batch_id": "batch_1", "custom_id": "u1_0", "raw": raw, "processed": False}])

    reprocess_staged(fake_client, "batch_1")

    assert fake_client.table("run_metrics").rows == []


def test_reprocess_staged_marks_rows_processed(fake_client):
    _seed_manifest(fake_client, "batch_1", jobs={"job-1": "2026-10-01T00:00:00+00:00"})
    raw = _FakeSucceededResult(_block(_valid_output_dict())).model_dump()
    fake_client.seed("batch_results", [{"batch_id": "batch_1", "custom_id": "u1_0", "raw": raw, "processed": False}])

    reprocess_staged(fake_client, "batch_1")

    [row] = fake_client.table("batch_results").rows
    assert row["processed"] is True
