"""Tests for src/enrich_jobs.py: prompt content, response parsing, the
orchestration flow (batch submit -> parse/validate -> one retry -> stale), and
crash-safety (batch_submissions/batch_results lifecycle, reprocess_staged())
against a fake Anthropic Batches client and the existing FakeSupabaseClient."""
from __future__ import annotations

import httpx
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


def test_build_prompt_includes_posting_type_classification():
    prompt = build_prompt(title="x", org="y", location_field=None, description="z")
    assert "posting_type" in prompt
    for value in ["job", "internship", "fellowship", "volunteer", "funding", "program", "speculative"]:
        assert value in prompt


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
# _merge_tool_use_blocks / _parse_enrichment — operate on plain dicts (see
# module docstring: results are normalized via .model_dump(mode="json") right
# after retrieval, so the live path and reprocess_staged()'s replay path share
# identical parsing code with no dual-shape handling)
# ---------------------------------------------------------------------------

def _block(input_data, *, type="tool_use", name="enrich_job") -> dict:  # noqa: A002
    return {"type": type, "name": name, "input": input_data}


def _message(blocks) -> dict:
    return {"content": blocks if isinstance(blocks, list) else [blocks]}


def test_merge_tool_use_blocks_single_block():
    full = _valid_enrichment_dict()
    msg = _message(_block(full))
    assert _merge_tool_use_blocks(msg, "enrich_job") == full


def test_merge_tool_use_blocks_merges_split_blocks():
    full = _valid_enrichment_dict()
    blocks = [_block({k: v}) for k, v in full.items()]
    msg = _message(blocks)
    assert _merge_tool_use_blocks(msg, "enrich_job") == full


def test_merge_tool_use_blocks_raises_when_no_match():
    msg = _message([_block({"x": 1}, name="other_tool")])
    with pytest.raises(IndexError):
        _merge_tool_use_blocks(msg, "enrich_job")


def test_parse_enrichment_valid():
    msg = _message(_block(_valid_enrichment_dict()))
    result = _parse_enrichment(msg)
    assert isinstance(result, JobEnrichment)
    assert result.summary == _valid_enrichment_dict()["summary"]


def test_parse_enrichment_invalid_raises_validation_error():
    msg = _message(_block({"summary": "x"}))  # missing work_arrangement
    with pytest.raises(ValidationError):
        _parse_enrichment(msg)


# ---------------------------------------------------------------------------
# enrich_jobs() orchestration — fake Anthropic Batches client + FakeSupabaseClient
# ---------------------------------------------------------------------------

class _FakeSucceededResult:
    def __init__(self, blocks, usage=None):
        blocks = blocks if isinstance(blocks, list) else [blocks]
        self._dump = {
            "type": "succeeded",
            "message": {
                "content": blocks,
                "usage": usage or {"input_tokens": 100, "output_tokens": 50},
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
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
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
        [_FakeResultItem("job-1", _FakeSucceededResult(_block({"summary": "incomplete"})))],  # round 1: invalid
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],   # round 2: valid
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
        [_FakeResultItem("job-1", _FakeErroredResult())],
        [_FakeResultItem("job-1", _FakeErroredResult())],
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
            _FakeResultItem("job-good", _FakeSucceededResult(_block(_valid_enrichment_dict()))),
            _FakeResultItem("job-bad", _FakeErroredResult()),
        ],
        [_FakeResultItem("job-bad", _FakeErroredResult())],
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
        [_FakeResultItem(f"job-{i}", _FakeSucceededResult(_block(_valid_enrichment_dict()))) for i in range(2)],
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
        [_FakeResultItem("job-pending", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["selected"] == 1


def test_enrich_jobs_writes_run_metrics_with_cost_from_actual_usage(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceededResult(
            _block(_valid_enrichment_dict()), usage={"input_tokens": 1000, "output_tokens": 200},
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
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["embeddings_skipped"] is False
    row = fake_client.table("jobs").rows[0]
    assert row["embedding"] == [0.1] * 1024


def test_enrich_jobs_voyage_failure_does_not_block_enrichment_write(fake_client, monkeypatch):
    """A Voyage outage must not lose the enrichment (and Anthropic spend) that
    already succeeded — embedding stays NULL, status still flips to enriched."""
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")

    def _raise(texts):
        raise httpx.HTTPError("voyage is down")

    monkeypatch.setattr(enrich_jobs, "embed_texts", _raise)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["enriched"] == 1
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["embedding"] is None


# ---------------------------------------------------------------------------
# embed_texts — chunking at Voyage's per-request cap
# ---------------------------------------------------------------------------

class _FakeVoyageResponse:
    def __init__(self, n: int):
        self._n = n

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"embedding": [0.5] * 1024} for _ in range(self._n)]}


def test_embed_texts_single_call_for_small_input(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["input"])
        return _FakeVoyageResponse(len(json["input"]))

    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(enrich_jobs.httpx, "post", fake_post)

    result = enrich_jobs.embed_texts([f"text {i}" for i in range(5)])

    assert len(calls) == 1
    assert len(result) == 5


def test_embed_texts_chunks_at_voyage_batch_size(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["input"])
        return _FakeVoyageResponse(len(json["input"]))

    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(enrich_jobs.httpx, "post", fake_post)

    texts = [f"text {i}" for i in range(300)]
    result = enrich_jobs.embed_texts(texts)

    assert len(calls) == 3  # 128 + 128 + 44
    assert [len(c) for c in calls] == [128, 128, 44]
    assert len(result) == 300


# ---------------------------------------------------------------------------
# backfill_embeddings
# ---------------------------------------------------------------------------

def test_backfill_embeddings_noop_without_key(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row(status="enriched", embedding=None)])

    result = enrich_jobs.backfill_embeddings(fake_client)

    assert result == {"selected": 0, "embedded": 0, "failed": 0}
    assert fake_client.table("jobs").upsert_calls == []


def test_backfill_embeddings_noop_when_nothing_missing(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    fake_client.seed("jobs", [_job_row(status="enriched", embedding=[0.1] * 1024)])

    result = enrich_jobs.backfill_embeddings(fake_client)

    assert result == {"selected": 0, "embedded": 0, "failed": 0}


def test_backfill_embeddings_embeds_and_writes_not_null_safe_payload(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(enrich_jobs, "embed_texts", lambda texts: [[0.2] * 1024 for _ in texts])
    fake_client.seed("jobs", [_job_row(
        status="enriched", embedding=None,
        canonical_url="https://example.org/x", source_board="80000_hours", content_hash="abc",
        enrichment={"summary": "A great role."},
    )])

    result = enrich_jobs.backfill_embeddings(fake_client)

    assert result == {"selected": 1, "embedded": 1, "failed": 0}
    row = fake_client.table("jobs").rows[0]
    assert row["embedding"] == [0.2] * 1024
    [payload] = fake_client.table("jobs").upsert_calls
    assert payload[0]["canonical_url"] == "https://example.org/x"
    assert payload[0]["source_board"] == "80000_hours"
    assert payload[0]["content_hash"] == "abc"


def test_backfill_embeddings_voyage_failure_writes_nothing(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")

    def _raise(texts):
        raise httpx.HTTPError("voyage is down")

    monkeypatch.setattr(enrich_jobs, "embed_texts", _raise)
    fake_client.seed("jobs", [_job_row(status="enriched", embedding=None)])

    result = enrich_jobs.backfill_embeddings(fake_client)

    assert result == {"selected": 1, "embedded": 0, "failed": 1}
    assert fake_client.table("jobs").upsert_calls == []


def test_enrich_jobs_enriched_and_stale_updates_are_separate_upsert_calls(fake_client, monkeypatch):
    """NULL-fill-avoidance check: enriched-job payloads and stale-job payloads have
    different key sets, so they must never be mixed into one heterogeneous upsert
    call (same lesson as src/ingest.py's changed vs unchanged split)."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row("job-good"), _job_row("job-bad")])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [
            _FakeResultItem("job-good", _FakeSucceededResult(_block(_valid_enrichment_dict()))),
            _FakeResultItem("job-bad", _FakeErroredResult()),
        ],
        [_FakeResultItem("job-bad", _FakeErroredResult())],
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
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [payload] = fake_client.table("jobs").upsert_calls
    assert payload[0]["canonical_url"] == "https://example.org/x"
    assert payload[0]["source_board"] == "80000_hours"
    assert payload[0]["content_hash"] == "abc123"


# ---------------------------------------------------------------------------
# Crash-safety: batch_submissions / batch_results lifecycle
# ---------------------------------------------------------------------------

def test_enrich_jobs_records_batch_submission_and_retrieval(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [submission] = fake_client.table("batch_submissions").rows
    assert submission["module"] == "enrich_jobs"
    assert submission["status"] == "retrieved"  # updated after successful .results()


def test_enrich_jobs_persists_raw_results_before_parsing(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))],
    ])

    run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [staged] = fake_client.table("batch_results").rows
    assert staged["custom_id"] == "job-1"
    assert staged["raw"]["type"] == "succeeded"
    assert staged["processed"] is False  # left as staged — this module's own tables don't mark it processed


def test_enrich_jobs_resumes_outstanding_batch_instead_of_resubmitting(fake_client, monkeypatch):
    """A prior run that submitted a batch but crashed before finishing polling
    leaves a 'submitted' batch_submissions row. The next run must resume
    polling that batch_id, never submit fresh (duplicate, already-paid-for) work."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    fake_client.seed("batch_submissions", [{"batch_id": "orphaned_batch", "module": "enrich_jobs", "status": "submitted"}])
    anthropic_client = _FakeAnthropicClient(rounds=[])  # .create() must never be called

    class _ResumableBatchesAPI(_FakeBatchesAPI):
        def retrieve(self, batch_id):
            assert batch_id == "orphaned_batch"
            return _FakeBatch(batch_id)

        def results(self, batch_id):
            assert batch_id == "orphaned_batch"
            return [_FakeResultItem("job-1", _FakeSucceededResult(_block(_valid_enrichment_dict())))]

    anthropic_client.messages.batches = _ResumableBatchesAPI(rounds=[])

    summary = run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    assert summary["enriched"] == 1
    assert anthropic_client.messages.batches.created_batches == []  # never submitted fresh work


def test_enrich_jobs_run_metrics_written_even_if_final_jobs_write_fails(fake_client, monkeypatch):
    """The whole point of moving run_metrics before the final write: a crash in
    that write must not lose the cost record."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    anthropic_client = _FakeAnthropicClient(rounds=[
        [_FakeResultItem("job-1", _FakeSucceededResult(
            _block(_valid_enrichment_dict()), usage={"input_tokens": 1000, "output_tokens": 200},
        ))],
    ])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated statement timeout")

    monkeypatch.setattr(enrich_jobs, "_write_enrichment", _boom)

    with pytest.raises(RuntimeError):
        run_enrich_jobs(client=fake_client, anthropic_client=anthropic_client)

    [metrics] = fake_client.table("run_metrics").rows
    assert metrics["jobs_enriched"] == 1
    assert metrics["estimated_cost_usd"] > 0


# ---------------------------------------------------------------------------
# reprocess_staged — recovery path, no API call
# ---------------------------------------------------------------------------

def test_reprocess_staged_no_unprocessed_rows_is_noop(fake_client):
    result = enrich_jobs.reprocess_staged(fake_client, "batch_1")
    assert result == {"selected": 0, "enriched": 0, "stale": 0}


def test_reprocess_staged_parses_and_writes_without_api_call(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "job-1",
        "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(),
        "processed": False,
    }])

    result = enrich_jobs.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 1, "stale": 0}
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["enrichment"]["summary"] == _valid_enrichment_dict()["summary"]


def test_reprocess_staged_marks_rows_processed(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "job-1",
        "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(),
        "processed": False,
    }])

    enrich_jobs.reprocess_staged(fake_client, "batch_1")

    [row] = fake_client.table("batch_results").rows
    assert row["processed"] is True
    assert row["raw"] is not None  # NOT NULL column carried through, not dropped


def test_reprocess_staged_failure_marks_stale_no_second_api_retry(fake_client, monkeypatch):
    """No retry-with-error-appended here — that would need a new LLM call,
    defeating "without re-calling the API". A parse failure on the staged raw
    data is terminal."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "job-1",
        "raw": _FakeErroredResult().model_dump(),
        "processed": False,
    }])

    result = enrich_jobs.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 0, "stale": 1}
    assert fake_client.table("jobs").rows[0]["status"] == "stale"


def test_reprocess_staged_does_not_write_run_metrics(fake_client, monkeypatch):
    """The normal path already wrote run_metrics before the crash that
    necessitated recovery — writing it again here would double-count cost."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row()])
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "job-1",
        "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(),
        "processed": False,
    }])

    enrich_jobs.reprocess_staged(fake_client, "batch_1")

    assert fake_client.table("run_metrics").rows == []


def test_reprocess_staged_only_processes_that_batch_id(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("jobs", [_job_row("job-1"), _job_row("job-2")])
    fake_client.seed("batch_results", [
        {"batch_id": "batch_1", "custom_id": "job-1",
         "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(), "processed": False},
        {"batch_id": "batch_2", "custom_id": "job-2",
         "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(), "processed": False},
    ])

    result = enrich_jobs.reprocess_staged(fake_client, "batch_1")

    assert result == {"selected": 1, "enriched": 1, "stale": 0}
    rows_by_id = {r["id"]: r for r in fake_client.table("jobs").rows}
    assert rows_by_id["job-1"]["status"] == "enriched"
    assert rows_by_id["job-2"]["status"] == "pending_enrich"  # untouched — different batch_id


def test_reprocess_staged_missing_job_logged_and_skipped(fake_client, caplog):
    """A staged result for a job_id that no longer exists (e.g. deleted) must
    not crash the recovery run."""
    fake_client.seed("batch_results", [{
        "batch_id": "batch_1", "custom_id": "nonexistent-job",
        "raw": _FakeSucceededResult(_block(_valid_enrichment_dict())).model_dump(),
        "processed": False,
    }])

    result = enrich_jobs.reprocess_staged(fake_client, "batch_1")

    assert result["enriched"] == 0
    assert result["stale"] == 0
