"""Tests for matches_store.py: upsert column allowlist and existing-URL fetch."""
from __future__ import annotations

from datetime import date

import pytest

from src.dedup import canonical_url
from src.matches_store import fetch_existing_match_urls, upsert_match
from src.schemas import MatchRow

_FORBIDDEN_COLUMNS = {"status", "user_notes", "discarded"}


def _match_row(**kwargs) -> MatchRow:
    defaults = dict(
        title="Head of Data",
        org="GiveDirectly",
        org_summary="A charity evaluator.",
        source="80k",
        url="https://example.org/job/1",
        canonical_url=canonical_url("https://example.org/job/1"),
        date_found=date.today(),
        fit_score=0.85,
        dimension_scores={"cause_mission_fit": 0.9, "role_function_fit": 0.8},
        why_fits="Strong mission alignment.",
        why_not_fits="No fundraising experience.",
        emphasize_in_cv=["BI leadership"],
        deemphasize=["IT portfolio work"],
    )
    defaults.update(kwargs)
    return MatchRow(**defaults)


# ---------------------------------------------------------------------------
# upsert_match — never writes user-owned columns
# ---------------------------------------------------------------------------

def test_upsert_never_writes_forbidden_columns(fake_client):
    upsert_match(fake_client, "u1", _match_row())
    row = fake_client.table("matches").rows[0]
    assert not (_FORBIDDEN_COLUMNS & row.keys())


def test_upsert_sets_user_id(fake_client):
    upsert_match(fake_client, "u1", _match_row())
    row = fake_client.table("matches").rows[0]
    assert row["user_id"] == "u1"


def test_upsert_payload_uses_dimension_scores_key(fake_client):
    upsert_match(fake_client, "u1", _match_row(dimension_scores={"cause_mission_fit": 0.42}))
    row = fake_client.table("matches").rows[0]
    assert row["dimension_scores"] == {"cause_mission_fit": 0.42}
    assert "scores" not in row


def test_upsert_includes_url(fake_client):
    upsert_match(fake_client, "u1", _match_row())
    row = fake_client.table("matches").rows[0]
    assert row["url"] == "https://example.org/job/1"


def test_upsert_re_encounter_does_not_touch_forbidden_columns(fake_client):
    """Even a direct re-upsert (bypassing the pipeline's skip-set) must never clobber
    user-owned tracking columns — MatchRow structurally has no such fields."""
    row = _match_row()
    upsert_match(fake_client, "u1", row)

    # Simulate the app having set a status on this row after the first upsert.
    fake_client.table("matches").rows[0]["status"] = "Applied"

    upsert_match(fake_client, "u1", row)  # re-encounter, same canonical_url
    assert fake_client.table("matches").rows[0]["status"] == "Applied"


def test_upsert_dry_run_does_not_write(fake_client):
    upsert_match(fake_client, "u1", _match_row(), dry_run=True)
    assert fake_client.table("matches").rows == []


def test_upsert_on_conflict_updates_not_duplicates(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)
    upsert_match(fake_client, "u1", row)
    assert len(fake_client.table("matches").rows) == 1


# ---------------------------------------------------------------------------
# fetch_existing_match_urls
# ---------------------------------------------------------------------------

def test_fetch_existing_match_urls_empty(fake_client):
    assert fetch_existing_match_urls(fake_client, "u1") == set()


def test_fetch_existing_match_urls_scoped_to_user(fake_client):
    upsert_match(fake_client, "u1", _match_row(url="https://example.org/a", canonical_url=canonical_url("https://example.org/a")))
    upsert_match(fake_client, "u2", _match_row(url="https://example.org/b", canonical_url=canonical_url("https://example.org/b")))

    assert fetch_existing_match_urls(fake_client, "u1") == {canonical_url("https://example.org/a")}
    assert fetch_existing_match_urls(fake_client, "u2") == {canonical_url("https://example.org/b")}
