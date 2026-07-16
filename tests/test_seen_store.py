"""Tests for seen_store.py against the FakeSupabaseClient."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.dedup import canonical_url
from src.seen_store import fetch_global_first_seen, load_seen_urls, record_verdict

_URL = "https://example.org/job/42?utm_source=newsletter"
_URL2 = "https://example.org/job/99"


# ---------------------------------------------------------------------------
# load_seen_urls
# ---------------------------------------------------------------------------

def test_load_seen_urls_empty(fake_client):
    assert load_seen_urls(fake_client, "u1") == {}


def test_load_seen_urls_scoped_to_user(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    record_verdict(fake_client, {}, "u2", _URL2, "gated_out", None)

    assert set(load_seen_urls(fake_client, "u1").keys()) == {canonical_url(_URL)}
    assert set(load_seen_urls(fake_client, "u2").keys()) == {canonical_url(_URL2)}


# ---------------------------------------------------------------------------
# record_verdict
# ---------------------------------------------------------------------------

def test_record_adds_entry_with_first_seen(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    row = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]
    assert row["verdict"] == "gated_out"
    assert row["fit_score"] is None
    assert row["first_seen"]


def test_record_strips_tracking_params_for_key(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    assert canonical_url(_URL) in load_seen_urls(fake_client, "u1")


def test_record_preserves_first_seen_on_update(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    first_seen_1 = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]["first_seen"]

    record_verdict(fake_client, seen_map, "u1", _URL, "below_threshold", 0.6)
    first_seen_2 = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]["first_seen"]

    assert first_seen_1 == first_seen_2


def test_record_updates_verdict_and_fit_score(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    record_verdict(fake_client, seen_map, "u1", _URL, "inserted", 0.82)

    row = load_seen_urls(fake_client, "u1")[canonical_url(_URL)]
    assert row["verdict"] == "inserted"
    assert row["fit_score"] == pytest.approx(0.82)


def test_record_updates_in_memory_seen_map(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None)
    assert canonical_url(_URL) in seen_map


def test_dry_run_does_not_write(fake_client):
    seen_map = {}
    record_verdict(fake_client, seen_map, "u1", _URL, "gated_out", None, dry_run=True)
    assert load_seen_urls(fake_client, "u1") == {}
    assert seen_map == {}


def test_two_users_can_record_same_url_independently(fake_client):
    record_verdict(fake_client, {}, "u1", _URL, "gated_out", None)
    record_verdict(fake_client, {}, "u2", _URL, "inserted", 0.9)

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
