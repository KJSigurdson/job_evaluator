"""Tests for seen_store.py against the FakeSupabaseClient.

record_verdict() only queues a payload (no network call); upsert_verdicts() flushes
everything queued in one (or a few chunked) batched upsert — see pipeline.py's
per-user loop, which calls record_verdict per posting and upsert_verdicts once
per user.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.dedup import canonical_url
from src.seen_store import _CHUNK_SIZE, fetch_global_first_seen, load_seen_urls, record_verdict, upsert_verdicts

_URL = "https://example.org/job/42?utm_source=newsletter"
_URL2 = "https://example.org/job/99"


def _record_and_flush(client, seen_map, user_id, url, verdict, fit_score, *, dry_run: bool = False) -> None:
    """Convenience helper mirroring the old single-item record_verdict(...) call —
    queue one verdict and flush it immediately, for tests that don't care about batching."""
    pending: list[dict] = []
    record_verdict(pending, seen_map, user_id, url, verdict, fit_score)
    upsert_verdicts(client, pending, dry_run=dry_run)


# ---------------------------------------------------------------------------
# load_seen_urls
# ---------------------------------------------------------------------------

def test_load_seen_urls_empty(fake_client):
    assert load_seen_urls(fake_client, "u1") == {}


def test_load_seen_urls_scoped_to_user(fake_client):
    _record_and_flush(fake_client, {}, "u1", _URL, "gated_out", None)
    _record_and_flush(fake_client, {}, "u2", _URL2, "gated_out", None)

    assert set(load_seen_urls(fake_client, "u1").keys()) == {canonical_url(_URL)}
    assert set(load_seen_urls(fake_client, "u2").keys()) == {canonical_url(_URL2)}


# ---------------------------------------------------------------------------
# record_verdict — pure queuing, no network I/O
# ---------------------------------------------------------------------------

def test_record_verdict_does_not_write(fake_client):
    pending: list[dict] = []
    record_verdict(pending, {}, "u1", _URL, "gated_out", None)
    assert fake_client.table("seen").rows == []
    assert fake_client.table("seen").upsert_calls == []


def test_record_verdict_appends_payload_with_first_seen():
    pending: list[dict] = []
    record_verdict(pending, {}, "u1", _URL, "gated_out", None)
    assert len(pending) == 1
    payload = pending[0]
    assert payload["user_id"] == "u1"
    assert payload["canonical_url"] == canonical_url(_URL)
    assert payload["verdict"] == "gated_out"
    assert payload["fit_score"] is None
    assert payload["first_seen"]


def test_record_verdict_new_url_gets_a_fresh_first_seen():
    pending: list[dict] = []
    record_verdict(pending, {}, "u1", _URL, "gated_out", None)
    assert "first_seen" in pending[0]
    assert pending[0]["first_seen"] is not None


def test_record_verdict_already_seen_url_preserves_exact_first_seen():
    """first_seen must ALWAYS be present in the payload — never omitted — even when
    the URL was already seen, so every row in a batch has a uniform key set (see
    module docstring: PostgREST NULL-fills any column missing from a row in a bulk
    upsert; a heterogeneous batch would NULL out an already-seen row's first_seen)."""
    original_first_seen = "2020-01-01T00:00:00+00:00"
    seen_map = {canonical_url(_URL): {"first_seen": original_first_seen}}
    pending: list[dict] = []
    record_verdict(pending, seen_map, "u1", _URL, "below_threshold", 0.5)

    assert "first_seen" in pending[0]
    assert pending[0]["first_seen"] == original_first_seen


def test_record_verdict_uniform_keys_across_mixed_batch(fake_client):
    """A pending list containing one brand-new URL and one already-seen URL must
    produce payloads with the IDENTICAL key set — no PostgREST NULL-fill regardless
    of chunk composition."""
    seen_map = {canonical_url(_URL2): {"first_seen": "2020-01-01T00:00:00+00:00"}}
    pending: list[dict] = []
    record_verdict(pending, seen_map, "u1", _URL, "gated_out", None)        # brand-new
    record_verdict(pending, seen_map, "u1", _URL2, "below_threshold", 0.5)  # already seen

    assert len(pending) == 2
    assert set(pending[0].keys()) == set(pending[1].keys())

    upsert_verdicts(fake_client, pending)

    call = fake_client.table("seen").upsert_calls[0]
    assert len(call) == 2
    assert set(call[0].keys()) == set(call[1].keys())
    for row in call:
        assert row["first_seen"] is not None


def test_record_verdict_mixed_batch_regression_matches_production_crash():
    """Regression for the exact production scenario: seed seen_map with one
    pre-existing URL, then record_verdict for it AND a brand-new URL into the same
    pending list. Both payloads must have non-null first_seen, and the already-seen
    one must match its original value exactly (not a fresh timestamp, not NULL)."""
    original_first_seen = "2019-06-15T12:00:00+00:00"
    seen_map = {canonical_url(_URL): {"first_seen": original_first_seen, "verdict": "gated_out", "fit_score": None}}
    pending: list[dict] = []

    record_verdict(pending, seen_map, "u1", _URL, "below_threshold", 0.6)   # pre-existing
    record_verdict(pending, seen_map, "u1", _URL2, "gated_out", None)       # brand-new

    existing_payload = next(p for p in pending if p["canonical_url"] == canonical_url(_URL))
    new_payload = next(p for p in pending if p["canonical_url"] == canonical_url(_URL2))

    assert existing_payload["first_seen"] is not None
    assert existing_payload["first_seen"] == original_first_seen
    assert new_payload["first_seen"] is not None
    assert new_payload["first_seen"] != original_first_seen


def test_record_verdict_updates_seen_map_in_memory():
    seen_map: dict[str, dict] = {}
    pending: list[dict] = []
    record_verdict(pending, seen_map, "u1", _URL, "gated_out", None)
    assert canonical_url(_URL) in seen_map


def test_record_verdict_strips_tracking_params_for_key():
    pending: list[dict] = []
    record_verdict(pending, {}, "u1", _URL, "gated_out", None)
    assert pending[0]["canonical_url"] == canonical_url(_URL)


def test_record_verdict_multiple_calls_accumulate_in_same_list():
    pending: list[dict] = []
    record_verdict(pending, {}, "u1", _URL, "gated_out", None)
    record_verdict(pending, {}, "u1", _URL2, "below_threshold", 0.6)
    assert len(pending) == 2
    assert {p["canonical_url"] for p in pending} == {canonical_url(_URL), canonical_url(_URL2)}


# ---------------------------------------------------------------------------
# upsert_verdicts — batched network write
# ---------------------------------------------------------------------------

def test_upsert_verdicts_writes_all_queued_rows(fake_client):
    seen_map: dict[str, dict] = {}
    pending: list[dict] = []
    record_verdict(pending, seen_map, "u1", _URL, "gated_out", None)
    record_verdict(pending, seen_map, "u1", _URL2, "below_threshold", 0.6)

    upsert_verdicts(fake_client, pending)

    loaded = load_seen_urls(fake_client, "u1")
    assert loaded[canonical_url(_URL)]["verdict"] == "gated_out"
    assert loaded[canonical_url(_URL2)]["verdict"] == "below_threshold"


def test_upsert_verdicts_many_postings_issue_one_upsert_call(fake_client):
    """Regression: 50 postings for one user must produce ONE upsert() call (one
    Supabase round-trip), not 50 — the whole point of batching."""
    seen_map: dict[str, dict] = {}
    pending: list[dict] = []
    for i in range(50):
        record_verdict(pending, seen_map, "u1", f"https://example.org/job/{i}", "gated_out", None)

    upsert_verdicts(fake_client, pending)

    assert len(fake_client.table("seen").upsert_calls) == 1
    assert len(fake_client.table("seen").upsert_calls[0]) == 50
    assert len(fake_client.table("seen").rows) == 50


def test_upsert_verdicts_chunks_at_chunk_size(fake_client):
    seen_map: dict[str, dict] = {}
    pending: list[dict] = []
    n = _CHUNK_SIZE + 200
    for i in range(n):
        record_verdict(pending, seen_map, "u1", f"https://example.org/job/{i}", "gated_out", None)

    upsert_verdicts(fake_client, pending)

    calls = fake_client.table("seen").upsert_calls
    assert len(calls) == 2
    assert len(calls[0]) == _CHUNK_SIZE
    assert len(calls[1]) == 200
    assert len(fake_client.table("seen").rows) == n


def test_upsert_verdicts_empty_pending_makes_no_call(fake_client):
    upsert_verdicts(fake_client, [])
    assert fake_client.table("seen").upsert_calls == []


def test_upsert_verdicts_dry_run_writes_nothing(fake_client):
    seen_map: dict[str, dict] = {}
    pending: list[dict] = []
    record_verdict(pending, seen_map, "u1", _URL, "gated_out", None)

    upsert_verdicts(fake_client, pending, dry_run=True)

    assert fake_client.table("seen").rows == []
    assert fake_client.table("seen").upsert_calls == []


def test_upsert_verdicts_on_conflict_is_user_id_canonical_url(fake_client):
    """Re-flushing the same (user_id, canonical_url) updates in place — doesn't duplicate."""
    seen_map: dict[str, dict] = {}
    pending1: list[dict] = []
    record_verdict(pending1, seen_map, "u1", _URL, "gated_out", None)
    upsert_verdicts(fake_client, pending1)

    pending2: list[dict] = []
    record_verdict(pending2, seen_map, "u1", _URL, "inserted", 0.9)
    upsert_verdicts(fake_client, pending2)

    rows = fake_client.table("seen").rows
    assert len(rows) == 1
    assert rows[0]["verdict"] == "inserted"


# ---------------------------------------------------------------------------
# first_seen preservation across separate upsert_verdicts flushes (i.e. across runs)
# ---------------------------------------------------------------------------

def test_first_seen_preserved_across_separate_flushes(fake_client):
    seen_map: dict[str, dict] = {}
    pending1: list[dict] = []
    record_verdict(pending1, seen_map, "u1", _URL, "gated_out", None)
    upsert_verdicts(fake_client, pending1)
    first_seen_1 = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]["first_seen"]

    # Simulate the next run: seen_map reloaded fresh from the DB, URL already present.
    seen_map_2 = load_seen_urls(fake_client, "u1")
    pending2: list[dict] = []
    record_verdict(pending2, seen_map_2, "u1", _URL, "below_threshold", 0.6)
    upsert_verdicts(fake_client, pending2)
    first_seen_2 = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]["first_seen"]

    assert first_seen_1 == first_seen_2


def test_two_users_can_record_same_url_independently(fake_client):
    _record_and_flush(fake_client, {}, "u1", _URL, "gated_out", None)
    _record_and_flush(fake_client, {}, "u2", _URL, "inserted", 0.9)

    assert load_seen_urls(fake_client, "u1")[canonical_url(_URL)]["verdict"] == "gated_out"
    assert load_seen_urls(fake_client, "u2")[canonical_url(_URL)]["verdict"] == "inserted"


# ---------------------------------------------------------------------------
# fetch_global_first_seen
# ---------------------------------------------------------------------------

def test_global_first_seen_empty_when_no_rows(fake_client):
    assert fetch_global_first_seen(fake_client) == {}


def test_global_first_seen_takes_earliest_across_users(fake_client):
    earlier = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    later = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    fake_client.seed("seen", [
        {"user_id": "u1", "canonical_url": canonical_url(_URL), "verdict": "gated_out",
         "fit_score": None, "first_seen": later},
        {"user_id": "u2", "canonical_url": canonical_url(_URL), "verdict": "gated_out",
         "fit_score": None, "first_seen": earlier},
    ])

    result = fetch_global_first_seen(fake_client)
    assert result[canonical_url(_URL)].isoformat() == earlier


def test_global_first_seen_per_url(fake_client):
    ts1 = datetime.now(timezone.utc).isoformat()
    ts2 = datetime.now(timezone.utc).isoformat()
    fake_client.seed("seen", [
        {"user_id": "u1", "canonical_url": canonical_url(_URL), "verdict": "gated_out",
         "fit_score": None, "first_seen": ts1},
        {"user_id": "u1", "canonical_url": canonical_url(_URL2), "verdict": "gated_out",
         "fit_score": None, "first_seen": ts2},
    ])

    result = fetch_global_first_seen(fake_client)
    assert set(result.keys()) == {canonical_url(_URL), canonical_url(_URL2)}
