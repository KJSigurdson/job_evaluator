"""Unit tests for seen_cache.py. load/save are the only IO; all other functions are pure."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.seen_cache import is_seen, load_seen, prune, record, save_seen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty() -> dict:
    return {"version": 1, "entries": {}}


def _cache_with(url: str, verdict: str, fit_score, days_old: int = 0) -> dict:
    """Return a cache dict containing a single entry."""
    first_seen = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    from src.dedup import canonical_url
    return {
        "version": 1,
        "entries": {
            canonical_url(url): {
                "first_seen": first_seen,
                "verdict": verdict,
                "fit_score": fit_score,
            }
        },
    }


_URL = "https://example.org/job/42?utm_source=newsletter"
_URL2 = "https://example.org/job/99"


# ---------------------------------------------------------------------------
# is_seen — pure
# ---------------------------------------------------------------------------

def test_is_seen_miss_on_empty_cache():
    assert not is_seen(_URL, _empty())


def test_is_seen_hit_after_record():
    data = _empty()
    record(data, _URL, "gated_out", None)
    assert is_seen(_URL, data)


def test_is_seen_strips_tracking_params():
    """URL with UTM params should match a record made for the clean URL."""
    from src.dedup import canonical_url
    clean = canonical_url(_URL)           # strips utm_source
    data = _empty()
    data["entries"][clean] = {"first_seen": datetime.now(timezone.utc).isoformat(),
                               "verdict": "gated_out", "fit_score": None}
    assert is_seen(_URL, data)            # queried with UTM — should still match


def test_is_seen_different_url_is_miss():
    data = _empty()
    record(data, _URL, "gated_out", None)
    assert not is_seen(_URL2, data)


# ---------------------------------------------------------------------------
# record — pure
# ---------------------------------------------------------------------------

def test_record_adds_entry():
    data = _empty()
    record(data, _URL, "gated_out", None)
    from src.dedup import canonical_url
    assert canonical_url(_URL) in data["entries"]


def test_record_sets_verdict_and_fit_score():
    data = _empty()
    record(data, _URL, "below_threshold", 0.62)
    from src.dedup import canonical_url
    entry = data["entries"][canonical_url(_URL)]
    assert entry["verdict"] == "below_threshold"
    assert entry["fit_score"] == pytest.approx(0.62)


def test_record_null_fit_score_for_gated_out():
    data = _empty()
    record(data, _URL, "gated_out", None)
    from src.dedup import canonical_url
    assert data["entries"][canonical_url(_URL)]["fit_score"] is None


def test_record_preserves_first_seen_on_update():
    data = _empty()
    record(data, _URL, "gated_out", None)
    from src.dedup import canonical_url
    original_ts = data["entries"][canonical_url(_URL)]["first_seen"]
    record(data, _URL, "below_threshold", 0.5)
    assert data["entries"][canonical_url(_URL)]["first_seen"] == original_ts


def test_record_updates_verdict_on_second_call():
    data = _empty()
    record(data, _URL, "gated_out", None)
    record(data, _URL, "inserted", 0.82)
    from src.dedup import canonical_url
    assert data["entries"][canonical_url(_URL)]["verdict"] == "inserted"


def test_record_round_trip_via_is_seen():
    data = _empty()
    assert not is_seen(_URL, data)
    record(data, _URL, "inserted", 0.9)
    assert is_seen(_URL, data)


# ---------------------------------------------------------------------------
# load_seen — IO
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_empty(tmp_path):
    result = load_seen(tmp_path / "does_not_exist.json")
    assert result == {"version": 1, "entries": {}}


def test_load_corrupt_file_returns_empty_and_does_not_raise(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text("not valid json {{{{")
    result = load_seen(p)
    assert result == {"version": 1, "entries": {}}


def test_load_wrong_shape_returns_empty(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text(json.dumps({"not_entries": {}}))
    result = load_seen(p)
    assert result == {"version": 1, "entries": {}}


# ---------------------------------------------------------------------------
# save_seen + load_seen — IO round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    path = tmp_path / "seen.json"
    data = _empty()
    record(data, _URL, "gated_out", None)
    save_seen(path, data)

    loaded = load_seen(path)
    assert is_seen(_URL, loaded)
    from src.dedup import canonical_url
    assert loaded["entries"][canonical_url(_URL)]["verdict"] == "gated_out"


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "sub" / "dir" / "seen.json"
    save_seen(path, _empty())
    assert path.exists()


def test_save_is_atomic_valid_json(tmp_path):
    """File should be valid JSON even if we read it immediately after save."""
    path = tmp_path / "seen.json"
    save_seen(path, _empty())
    content = path.read_text()
    parsed = json.loads(content)
    assert "entries" in parsed


# ---------------------------------------------------------------------------
# prune — pure
# ---------------------------------------------------------------------------

def test_prune_removes_old_entries():
    data = _cache_with(_URL, "gated_out", None, days_old=200)
    result = prune(data, max_age_days=180)
    assert not is_seen(_URL, result)


def test_prune_keeps_recent_entries():
    data = _cache_with(_URL, "gated_out", None, days_old=10)
    result = prune(data, max_age_days=180)
    assert is_seen(_URL, result)


def test_prune_one_day_inside_max_age_is_kept():
    data = _cache_with(_URL, "gated_out", None, days_old=179)
    result = prune(data, max_age_days=180)
    assert is_seen(_URL, result)


def test_prune_keeps_some_drops_others():
    data = _empty()
    record(data, _URL,  "gated_out",      None)
    record(data, _URL2, "below_threshold", 0.6)
    # backdate _URL to 200 days ago
    from src.dedup import canonical_url
    data["entries"][canonical_url(_URL)]["first_seen"] = (
        datetime.now(timezone.utc) - timedelta(days=200)
    ).isoformat()

    result = prune(data, max_age_days=180)
    assert not is_seen(_URL, result)
    assert is_seen(_URL2, result)


def test_prune_returns_new_dict():
    data = _cache_with(_URL, "gated_out", None, days_old=10)
    result = prune(data)
    assert result is not data


def test_prune_empty_cache_returns_empty():
    result = prune(_empty())
    assert result == {"version": 1, "entries": {}}
