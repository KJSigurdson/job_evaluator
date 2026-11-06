"""Tests for src/supabase_client.py's fetch_all() pagination helper.

Regression coverage for a real bug: PostgREST caps a single response at
db.max_rows (1000) with no error/warning — a plain query.execute() on a query
matching more rows than that silently truncates. fetch_all() must page through
via .range() until every row is collected.
"""
from __future__ import annotations

import httpx

from src.supabase_client import (
    _IN_FILTER_CHUNK_SIZE,
    chunked_insert,
    chunked_upsert,
    fetch_all,
    fetch_all_in,
    fetch_batch_manifest,
    find_outstanding_batch,
    insert_run_metrics,
    mark_batch_results_processed,
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
# fetch_all — no offset=/limit= accumulation across pages (2026-11-06)
#
# FakeSupabaseClient's own .range() just overwrites an attribute each call,
# so it can't reproduce this bug at all — these tests use a minimal stub
# that reproduces the REAL postgrest-py 1.0.2 behavior instead: .range()
# does `self.params = self.params.add(...)`, and httpx.QueryParams.add()
# APPENDS a same-named key rather than replacing it. Confirmed for real in
# a production log: a second .range() call on the same (mutated-in-place)
# query object left `...&offset=0&offset=1000&limit=1000&limit=1000` in the
# request URL for a `jobs` table page 2 fetch.
# ---------------------------------------------------------------------------

class _AccumulatingRangeQuery:
    """Reproduces postgrest-py's actual .range() implementation exactly
    (self.params = self.params.add(...)) so a regression here is caught by
    the same mechanism that caused the real bug, not a simplified fake.

    Tracks which page is next via len(params_per_call) — a shared MUTABLE
    list — rather than a plain int counter attribute: fetch_all()'s fix is
    copy.copy() per page, a shallow copy, so an int attribute (immutable)
    would NOT be shared across copies and every copy would see page 0
    forever (confirmed the hard way — this exact mistake hung the test
    suite in an infinite loop). The list IS shared by reference under a
    shallow copy, same as the real postgrest builder's `session`/`headers`
    staying shared on purpose — this fake must honor the same shallow-copy
    semantics it's meant to exercise.
    """

    def __init__(self, pages: list[list[dict]]):
        self.params = httpx.QueryParams()
        self._pages = pages
        self.params_per_call: list[httpx.QueryParams] = []

    def range(self, start: int, end: int) -> "_AccumulatingRangeQuery":
        self.params = self.params.add("offset", start)
        self.params = self.params.add("limit", end - start + 1)
        return self

    def execute(self):
        idx = len(self.params_per_call)
        self.params_per_call.append(self.params)
        data = self._pages[idx] if idx < len(self._pages) else []
        return type("_Resp", (), {"data": data})()


def test_fetch_all_each_page_request_has_exactly_one_offset_and_one_limit():
    pages = [[{"id": i} for i in range(1000)], [{"id": i} for i in range(1000, 1500)]]
    query = _AccumulatingRangeQuery(pages)

    rows = fetch_all(query, page_size=1000)

    assert len(rows) == 1500
    assert len(query.params_per_call) == 2
    for params in query.params_per_call:
        assert len(params.get_list("offset")) == 1
        assert len(params.get_list("limit")) == 1


def test_fetch_all_second_page_offset_and_limit_values_are_correct():
    """Not just 'exactly one' — the single remaining value per page must be
    THIS page's, not the first page's (which .add()'s append-only semantics
    would otherwise leave as the first, lower-priority-but-still-present
    entry if a naive fix just deduped instead of using a fresh query)."""
    pages = [[{"id": i} for i in range(1000)], [{"id": i} for i in range(1000, 1300)]]
    query = _AccumulatingRangeQuery(pages)

    fetch_all(query, page_size=1000)

    first_page_params, second_page_params = query.params_per_call
    assert first_page_params.get("offset") == "0"
    assert first_page_params.get("limit") == "1000"
    assert second_page_params.get("offset") == "1000"
    assert second_page_params.get("limit") == "1000"


def test_fetch_all_three_pages_no_accumulation_on_any_page():
    pages = [[{"id": i} for i in range(1000)], [{"id": i} for i in range(1000, 2000)], [{"id": i} for i in range(2000, 2100)]]
    query = _AccumulatingRangeQuery(pages)

    rows = fetch_all(query, page_size=1000)

    assert len(rows) == 2100
    assert len(query.params_per_call) == 3
    for params in query.params_per_call:
        assert len(params.get_list("offset")) == 1
        assert len(params.get_list("limit")) == 1


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
# fetch_all_in — chunked .in_() filter, merged results. Regression coverage
# for a real incident: src.closure.close_jobs() re-fetched 1,665 job_ids via
# a single .in_("id", job_ids), building a ~65KB GET URL that Cloudflare
# answered with a 520 before PostgREST ever saw it.
# ---------------------------------------------------------------------------

def test_in_filter_chunk_size_is_100():
    """Deliberately small — this is a URL-length limit (every PostgREST
    filter, .in_() included, is query-string-encoded), NOT the 500-row
    _WRITE_CHUNK_SIZE used by chunked_upsert/chunked_insert's POST bodies."""
    assert _IN_FILTER_CHUNK_SIZE == 100


def test_fetch_all_in_merges_results_across_chunks(fake_client):
    fake_client.seed("jobs", [{"id": i} for i in range(250)])

    rows = fetch_all_in(fake_client.table("jobs"), "id", list(range(250)), chunk_size=100)

    assert sorted(r["id"] for r in rows) == list(range(250))


def test_fetch_all_in_never_exceeds_chunk_size_per_request(fake_client):
    fake_client.seed("jobs", [{"id": i} for i in range(250)])

    fetch_all_in(fake_client.table("jobs"), "id", list(range(250)), chunk_size=100)

    sizes = [size for col, size in fake_client.table("jobs").in_call_sizes if col == "id"]
    assert sizes == [100, 100, 50]


def test_fetch_all_in_default_chunk_size_is_in_filter_chunk_size(fake_client):
    fake_client.seed("jobs", [{"id": i} for i in range(250)])

    fetch_all_in(fake_client.table("jobs"), "id", list(range(250)))

    sizes = [size for col, size in fake_client.table("jobs").in_call_sizes if col == "id"]
    assert all(size <= _IN_FILTER_CHUNK_SIZE for size in sizes)
    assert len(sizes) == 3  # 250 / 100, rounded up


def test_fetch_all_in_empty_values_no_request(fake_client):
    assert fetch_all_in(fake_client.table("jobs"), "id", []) == []
    assert fake_client.table("jobs").in_call_sizes == []


def test_fetch_all_in_respects_select_cols(fake_client):
    fake_client.seed("jobs", [{"id": "job-1", "title": "Head of Data", "org": "GiveDirectly"}])

    [row] = fetch_all_in(fake_client.table("jobs"), "id", ["job-1"], select_cols="id,title")

    # The fake doesn't project columns (mirrors other tests in this file),
    # but the call itself must not raise with a narrower select_cols.
    assert row["id"] == "job-1"


def test_fetch_all_in_applies_filter_fn_to_every_chunk(fake_client):
    fake_client.seed("jobs", [
        {"id": "job-1", "status": "pending_enrich"},
        {"id": "job-2", "status": "enriched"},
    ])

    rows = fetch_all_in(
        fake_client.table("jobs"), "id", ["job-1", "job-2"],
        filter_fn=lambda q: q.eq("status", "pending_enrich"),
    )

    assert [r["id"] for r in rows] == ["job-1"]


# ---------------------------------------------------------------------------
# mark_batch_results_processed — same URL-length concern as fetch_all_in,
# since .in_() is query-string-encoded for UPDATE too, not just SELECT.
# ---------------------------------------------------------------------------

def test_mark_batch_results_processed_default_chunk_is_in_filter_chunk_size(fake_client):
    custom_ids = [f"job-{i}" for i in range(250)]
    fake_client.seed("batch_results", [{"batch_id": "batch_1", "custom_id": cid, "processed": False} for cid in custom_ids])

    mark_batch_results_processed(fake_client, "batch_1", custom_ids)

    sizes = [size for col, size in fake_client.table("batch_results").in_call_sizes if col == "custom_id"]
    assert sizes == [100, 100, 50]
    assert all(r["processed"] is True for r in fake_client.table("batch_results").rows)


def test_mark_batch_results_processed_empty_list_no_calls(fake_client):
    mark_batch_results_processed(fake_client, "batch_1", [])
    assert fake_client.table("batch_results").in_call_sizes == []


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
    assert row["manifest"] is None


def test_record_batch_submitted_persists_manifest(fake_client):
    manifest = {"u1_0": {"user_id": "u1", "profile_content_hash": "hash-abc", "jobs": {"job-1": "t1"}}}
    record_batch_submitted(fake_client, "batch_1", "tier1_rerank", manifest=manifest)

    [row] = fake_client.table("batch_submissions").rows
    assert row["manifest"] == manifest


def test_fetch_batch_manifest_returns_persisted_manifest(fake_client):
    manifest = {"u1_0": {"user_id": "u1", "profile_content_hash": "hash-abc", "jobs": {"job-1": "t1"}}}
    record_batch_submitted(fake_client, "batch_1", "tier1_rerank", manifest=manifest)

    assert fetch_batch_manifest(fake_client, "batch_1") == manifest


def test_fetch_batch_manifest_none_when_batch_unknown(fake_client):
    assert fetch_batch_manifest(fake_client, "batch_unknown") is None


def test_fetch_batch_manifest_none_when_submitted_without_one(fake_client):
    record_batch_submitted(fake_client, "batch_1", "enrich_jobs")

    assert fetch_batch_manifest(fake_client, "batch_1") is None


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


# ---------------------------------------------------------------------------
# insert_run_metrics — NEVER raises. Regression for a real incident:
# close_jobs() closed 1,665 jobs for real (4 chunked upserts, all 200 OK),
# then its OWN run_metrics insert 400'd with PGRST204 ("Could not find the
# 'closure_summary' column" — a migration that existed in the repo but
# hadn't actually been applied) and that exception aborted the run before
# enrich/matching ever started, even though the closure work itself had
# already fully committed. Observability must never be able to do that.
# ---------------------------------------------------------------------------

def test_insert_run_metrics_writes_row_on_success(fake_client):
    insert_run_metrics(fake_client, {"stage": "ingest", "jobs_ingested": 5})

    [row] = fake_client.table("run_metrics").rows
    assert row == {"stage": "ingest", "jobs_ingested": 5}


def test_insert_run_metrics_swallows_failure_and_logs_error(fake_client, caplog):
    """The fake client has no real schema, so it can never catch a drift bug
    like PGRST204 on its own — raise_on_execute is how a test simulates
    "this specific .insert() call fails against real Postgres", which is
    exactly what happened in production here."""
    fake_client.table("run_metrics").raise_on_execute["insert"] = Exception(
        "PGRST204: Could not find the 'closure_summary' column of 'run_metrics' in the schema cache"
    )
    payload = {"stage": "closure", "closure_summary": {"closed": 1665}}

    with caplog.at_level("ERROR"):
        insert_run_metrics(fake_client, payload)  # must not raise

    assert fake_client.table("run_metrics").rows == []
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "closure_summary" in errors[0].message  # full payload logged, not just the exception
    assert "1665" in errors[0].message
