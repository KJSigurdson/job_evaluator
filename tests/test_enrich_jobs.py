"""Tests for src/enrich_jobs.py: prompt content, response parsing, and the
orchestration flow (batch submit -> parse/validate -> one retry -> stale) against a
fake Anthropic Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import enrich_jobs
from src.enrich_jobs import (
    _merge_tool_use_blocks,
    _parse_enrichment,
    build_prompt,
    build_retry_prompt,
    enrich_jobs as run_enrich_jobs,
)
from src.models.job_enrichment import JobEnrichment


def _valid_enrichment_dict(**overrides) -> dict:
    defaults = dict(
        summary="A data leadership role at a global health charity.",
        seniority=["senior"],
        cause_area=["global health"],
        skill_tags=["python", "sql"],
        salary=None,
        work_arrangement={
            "mode": "remote", "remote_scope": "unspecified", "eligible_regions": [],
            "timezone_constraint": None, "restriction_type": None,
        },
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# build_prompt / build_retry_prompt — content regression tests, not quality
# review (that's the explicit human gate before this ever runs for real)
# ---------------------------------------------------------------------------

def test_build_prompt_includes_job_fields():
    prompt = build_prompt(title="Head of Data", org="GiveDirectly", location_field="Remote, Global",
                           description="We need a data leader.")
    assert "Head of Data" in prompt
    assert "GiveDirectly" in prompt
    assert "Remote, Global" in prompt
    assert "We need a data leader." in prompt


def test_build_prompt_handles_missing_location_field():
    prompt = build_prompt(title="Analyst", org="Org", location_field=None, description="text")
    assert "(not stated)" in prompt


def test_build_prompt_includes_seniority_keyword_mapping():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "lead" in prompt.lower() and "senior" in prompt.lower()
    assert "principal" in prompt.lower() and "director" in prompt.lower()


def test_build_prompt_instructs_remote_is_not_automatically_global():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "NOT" in prompt and "global" in prompt.lower()
    assert "unspecified" in prompt.lower()


def test_build_prompt_instructs_scanning_full_description_for_authorization():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "authorized to work" in prompt.lower() or "work-authorization" in prompt.lower()


def test_build_prompt_includes_shorthand_examples():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "US-remote" in prompt
    assert "Remote, EU" in prompt


def test_build_prompt_mentions_restriction_type_distinction():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "authorization" in prompt.lower() and "timezone" in prompt.lower()


def test_build_retry_prompt_appends_validation_error():
    retry = build_retry_prompt(
        title="x", org="y", location_field=None, description="z",
        validation_error="seniority: field required",
    )
    assert "seniority: field required" in retry
    assert "previous attempt" in retry.lower()


def test_build_retry_prompt_still_contains_base_prompt_content():
    retry = build_retry_prompt(title="Head of Data", org="y", location_field=None, description="z", validation_error="err")
    assert "Head of Data" in retry


# ---------------------------------------------------------------------------
# _merge_tool_use_blocks / _parse_enrichment
# ---------------------------------------------------------------------------

class _FakeContentBlock:
    def __init__(self, input_data, *, type="tool_use", name="enrich_job"):  # noqa: A002
        self.input = input_data
        self.type = type
        self.name = name


class _FakeMessage:
    def __init__(self, blocks):
        self.content = blocks if isinstance(blocks, list) else [blocks]
        self.stop_reason = "tool_use"


def test_merge_tool_use_blocks_single_block():
    full = _valid_enrichment_dict()
    msg = _FakeMessage(_FakeContentBlock(full))
    assert _merge_tool_use_blocks(msg, "enrich_job") == full


def test_merge_tool_use_blocks_merges_split_blocks():
    full = _valid_enrichment_dict()
    blocks = [_FakeContentBlock({k: v}) for k, v in full.items()]
    msg = _FakeMessage(blocks)
    assert _merge_tool_use_blocks(msg, "enrich_job") == full


def test_merge_tool_use_blocks_raises_when_no_match():
    msg = _FakeMessage([_FakeContentBlock({"x": 1}, name="other_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(msg, "enrich_job")


def test_parse_enrichment_valid():
    msg = _FakeMessage(_FakeContentBlock(_valid_enrichment_dict()))
    result = _parse_enrichment(msg)
    assert isinstance(result, JobEnrichment)
    assert result.summary == _valid_enrichment_dict()["summary"]


def test_parse_enrichment_invalid_raises_validation_error():
    msg = _FakeMessage(_FakeContentBlock({"summary": "x"}))  # missing work_arrangement
    with pytest.raises(ValidationError):
        _parse_enrichment(msg)


# ---------------------------------------------------------------------------
# enrich_jobs() orchestration — fake Anthropic Batches client + FakeSupabaseClient
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
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
    """*rounds* is a list of "result item lists" — one per expected .create() call,
    in order. Each .create() call is assigned the next round's results, served back
    by .results() for that batch's id, and instantly reports processing_status='ended'
    so tests don't sleep through the real polling loop."""

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


def _job_row(job_id="job-1", **overrides) -> dict:
    defaults = dict(
        id=job_id, title="Head of Data", org="GiveDirectly",
        location_field="Remote, Global", raw_description="We need a data leader.",
        status="pending_enrich",
        canonical_url=f"https://example.org/{job_id}", source_board="80000_hours",
        content_hash="irrelevant",
    )
    defaults.update(overrides)
    return defaults


def test_enrich_jobs_happy_path_all_succeed_round_one(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary == {"selected": 1, "enriched": 1, "stale": 0, "embeddings_skipped": True}
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["enrichment"]["summary"] == _valid_enrichment_dict()["summary"]
    assert row["embedding"] is None
    assert len(anthropic_client.messages.batches.created_batches) == 1  # no retry needed


def test_enrich_jobs_failure_then_successful_retry(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceeded(_FakeContentBlock({"summary": "incomplete"})))],  # round 1: invalid
        [_FakeResultItem("job-1", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict())))],   # round 2: valid
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["enriched"] == 1
    assert summary["stale"] == 0
    assert fake_client.table("jobs").rows[0]["status"] == "enriched"
    assert len(anthropic_client.messages.batches.created_batches) == 2
    # The retry request's prompt must contain the validation error from round 1.
    retry_request = anthropic_client.messages.batches.created_batches[1][0]
    retry_prompt = retry_request["params"]["messages"][0]["content"]
    assert "previous attempt" in retry_prompt.lower()


def test_enrich_jobs_fails_both_rounds_marks_stale_never_writes_partial(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row(raw_description="original text")])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeErrored())],
        [_FakeResultItem("job-1", _FakeErrored())],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary == {"selected": 1, "enriched": 0, "stale": 1, "embeddings_skipped": True}
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "stale"
    assert "enrichment" not in row or row.get("enrichment") is None  # never partially written
    assert len(anthropic_client.messages.batches.created_batches) == 2


def test_enrich_jobs_mixed_batch_one_succeeds_one_permanently_fails(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row("job-good"), _job_row("job-bad")])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("job-good", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict()))),
            _FakeResultItem("job-bad", _FakeErrored()),
        ],
        [_FakeResultItem("job-bad", _FakeErrored())],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["enriched"] == 1
    assert summary["stale"] == 1
    rows_by_id = {r["id"]: r for r in fake_client.table("jobs").rows}
    assert rows_by_id["job-good"]["status"] == "enriched"
    assert rows_by_id["job-bad"]["status"] == "stale"


def test_enrich_jobs_dry_run_makes_no_batch_and_writes_nothing(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row(), _job_row("job-2")])
    anthropic_client = _FakeAnthropicClient(rounds=[])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client, dry_run=True)

    assert summary["selected"] == 2
    assert summary["enriched"] == 0
    assert anthropic_client.messages.batches.created_batches == []
    assert fake_client.table("run_metrics").rows == []


def test_enrich_jobs_respects_limit(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row(f"job-{i}") for i in range(5)])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem(f"job-{i}", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict()))) for i in range(2)],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client, limit=2)

    assert summary["selected"] == 2
    assert len(anthropic_client.messages.batches.created_batches[0]) == 2


def test_enrich_jobs_only_selects_pending_enrich_status(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [
        _job_row("job-pending", status="pending_enrich"),
        _job_row("job-done", status="enriched"),
        _job_row("job-stale", status="stale"),
    ])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-pending", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["selected"] == 1


def test_enrich_jobs_writes_run_metrics_with_cost_from_actual_usage(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceeded(
            _FakeContentBlock(_valid_enrichment_dict()), usage=_FakeUsage(input_tokens=1000, output_tokens=200),
        ))],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["jobs_enriched"] == 1
    assert metrics["tier1_calls"] == 1
    assert metrics["estimated_cost_usd"] > 0


def test_enrich_jobs_embeddings_used_when_key_present(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(enrich_jobs, "embed_texts", lambda texts: [[0.1] * 1024 for _ in texts])
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["embeddings_skipped"] is False
    row = fake_client.table("jobs").rows[0]
    assert row["embedding"] == [0.1] * 1024


def test_enrich_jobs_enriched_and_stale_updates_are_separate_upsert_calls(fake_client, monkeypatch):
    """NULL-fill-avoidance check: enriched-job payloads and stale-job payloads have
    different key sets, so they must never be mixed into one heterogeneous upsert
    call (same lesson as src/ingest.py's changed vs unchanged split)."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row("job-good"), _job_row("job-bad")])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("job-good", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict()))),
            _FakeResultItem("job-bad", _FakeErrored()),
        ],
        [_FakeResultItem("job-bad", _FakeErrored())],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    upsert_calls = fake_client.table("jobs").upsert_calls
    assert len(upsert_calls) == 2
    enriched_call = next(c for c in upsert_calls if "enrichment" in c[0])
    stale_call = next(c for c in upsert_calls if "enrichment" not in c[0])
    assert set(enriched_call[0].keys()) == {
        "id", "canonical_url", "source_board", "content_hash", "enrichment", "embedding", "enriched_at", "status",
    }
    assert set(stale_call[0].keys()) == {"id", "canonical_url", "source_board", "content_hash", "status"}


def test_enrich_jobs_enriched_update_carries_not_null_columns_through(fake_client, monkeypatch):
    """Regression: canonical_url/source_board/content_hash are NOT NULL with no
    default on `jobs` — Postgres validates NOT NULL on an upsert's INSERT branch
    even though this row already exists and only ON CONFLICT DO UPDATE actually
    fires (confirmed the hard way against production: omitting them here crashed
    with 'null value in column "canonical_url" violates not-null constraint')."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row(canonical_url="https://example.org/x", source_board="80000_hours", content_hash="abc123")])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceeded(_FakeContentBlock(_valid_enrichment_dict())))],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [payload] = fake_client.table("jobs").upsert_calls
    assert payload[0]["canonical_url"] == "https://example.org/x"
    assert payload[0]["source_board"] == "80000_hours"
    assert payload[0]["content_hash"] == "abc123"
