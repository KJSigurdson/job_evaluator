"""Tests for src/supabase_client.py's fetch_all() pagination helper.

Regression coverage for a real bug: PostgREST caps a single response at
db.max_rows (1000) with no error/warning — a plain query.execute() on a query
matching more rows than that silently truncates. fetch_all() must page through
via .range() until every row is collected.
"""
from __future__ import annotations

from src.supabase_client import (
    chunked_insert,
    chunked_upsert,
    fetch_all,
    find_outstanding_batch,
    persist_batch_results,
    record_batch_retrieved,
    record_batch_submitted,
)


def test_fetch_all_single_page_under_page_size(fake_client):
    fake_client.seed("jobs", [{"id": i} for i in range(5)])

    rows = fetch_all(fake_client.table("jobs").select("id"), page_size=1000)

    assert len(rows) == 5


def test_fetch_all_pages_past_the_cap(fake_client):
    fake_client.seed("jobs", [{"id": i} for i in range(250)])

    rows = fetch_all(fake_client.table("jobs").select("id"), page_size=100)

    assert len(rows) == 250
    assert sorted(r["id"] for r in rows) == list(range(250))


def test_fetch_all_exact_multiple_of_page_size(fake_client):
    """A result count that's an exact multiple of page_size must not trigger
    one extra, wastefully-empty request loop — the boundary case most likely
    to have an off-by-one."""
    fake_client.seed("jobs", [{"id": i} for i in range(200)])

    rows = fetch_all(fake_client.table("jobs").select("id"), page_size=100)

    assert len(rows) == 200


def test_fetch_all_empty_table(fake_client):
    assert fetch_all(fake_client.table("jobs").select("id"), page_size=1000) == []


def test_fetch_all_respects_prior_filters(fake_client):
    fake_client.seed("jobs", [{"id": i, "status": "enriched" if i % 2 == 0 else "stale"} for i in range(10)])

    rows = fetch_all(fake_client.table("jobs").select("id").eq("status", "enriched"), page_size=3)

    assert len(rows) == 5
    assert all(r["id"] % 2 == 0 for r in rows)


# ---------------------------------------------------------------------------
# chunked_upsert / chunked_insert
# ---------------------------------------------------------------------------

def test_chunked_upsert_splits_into_multiple_calls(fake_client):
    rows = [{"id": i, "val": i} for i in range(250)]

    chunked_upsert(fake_client.table("jobs"), rows, on_conflict="id", chunk_size=100)

    upsert_calls = fake_client.table("jobs").upsert_calls
    assert len(upsert_calls) == 3
    assert [len(c) for c in upsert_calls] == [100, 100, 50]
    assert len(fake_client.table("jobs").rows) == 250


def test_chunked_upsert_all_rows_written():
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    rows = [{"id": i} for i in range(1200)]

    chunked_upsert(client.table("jobs"), rows, on_conflict="id", chunk_size=500)

    assert len(client.table("jobs").rows) == 1200
    assert len(client.table("jobs").upsert_calls) == 3


def test_chunked_upsert_empty_rows_no_calls(fake_client):
    chunked_upsert(fake_client.table("jobs"), [], on_conflict="id")
    assert fake_client.table("jobs").upsert_calls == []


def test_chunked_insert_splits_into_multiple_calls(fake_client):
    rows = [{"id": i} for i in range(250)]

    chunked_insert(fake_client.table("jobs"), rows, chunk_size=100)

    insert_calls = fake_client.table("jobs").insert_calls
    assert len(insert_calls) == 3
    assert [len(c) for c in insert_calls] == [100, 100, 50]
    assert len(fake_client.table("jobs").rows) == 250


# ---------------------------------------------------------------------------
# Batch lifecycle: record_batch_submitted / record_batch_retrieved /
# find_outstanding_batch / persist_batch_results
# ---------------------------------------------------------------------------

def test_record_batch_submitted_writes_submitted_row(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")

    [row] = fake_client.table("batch_submissions").rows
    assert row["batch_id"] == "batch_1"
    assert row["module"] == "enrich_jobs"
    assert row["status"] == "submitted"


def test_find_outstanding_batch_returns_none_when_no_submissions(fake_client):
    assert find_outstanding_batch(fake_client, "enrich_jobs") is None


def test_find_outstanding_batch_finds_submitted_batch(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")

    assert find_outstanding_batch(fake_client, "enrich_jobs") == "batch_1"


def test_find_outstanding_batch_ignores_retrieved_batches(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")
    record_batch_retrieved(fake_client, "batch_1")

    assert find_outstanding_batch(fake_client, "enrich_jobs") is None


def test_find_outstanding_batch_scoped_to_module(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")

    assert find_outstanding_batch(fake_client, "tier1_rerank") is None


def test_find_outstanding_batch_multiple_outstanding_returns_none(fake_client, caplog):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")
    record_batch_submitted(fake_client, "batch_2", "enrich_jobs")

    assert find_outstanding_batch(fake_client, "enrich_jobs") is None


def test_record_batch_retrieved_updates_status(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")

    record_batch_retrieved(fake_client, "batch_1")

    [row] = fake_client.table("batch_submissions").rows
    assert row["status"] == "retrieved"


def test_persist_batch_results_writes_one_row_per_custom_id(fake_client):
    persist_batch_results(fake_client, "batch_1", {
        "job-1": {"type": "succeeded", "message": {"content": []}},
        "job-2": {"type": "errored"},
    })

    rows = fake_client.table("batch_results").rows
    assert len(rows) == 2
    by_custom_id = {r["custom_id"]: r for r in rows}
    assert by_custom_id["job-1"]["batch_id"] == "batch_1"
    assert by_custom_id["job-1"]["raw"] == {"type": "succeeded", "message": {"content": []}}
    assert by_custom_id["job-1"]["processed"] is False
    assert by_custom_id["job-2"]["raw"] == {"type": "errored"}


def test_persist_batch_results_empty_results_no_call(fake_client):
    persist_batch_results(fake_client, "batch_1", {})
    assert fake_client.table("batch_results").upsert_calls == []


def test_persist_batch_results_is_idempotent_on_resume(fake_client):
    """A resumed run re-persisting the same batch's results must not crash on
    the unique(batch_id, custom_id) constraint."""
    persist_batch_results(fake_client, "batch_1", {"job-1": {"type": "succeeded"}})
    persist_batch_results(fake_client, "batch_1", {"job-1": {"type": "succeeded"}})

    rows = fake_client.table("batch_results").rows
    assert len(rows) == 1
