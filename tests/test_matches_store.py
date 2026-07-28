"""Tests for matches_store.py: upsert column allowlist and existing-URL fetch."""
from __future__ import annotations

from datetime import date

import pytest

from src.dedup import canonical_url
from src.matches_store import fetch_existing_match_urls, update_enrichment, upsert_match
from src.schemas import MatchRow

_FORBIDDEN_COLUMNS = {"status", "user_notes", "discarded"}
_ENRICHMENT_COLUMNS = {"why_fits", "why_not_fits", "emphasize_in_cv", "deemphasize"}


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


def test_upsert_payload_writes_real_lists_for_cv_guidance(fake_client):
    """Regression: emphasize_in_cv/deemphasize map to jsonb columns and must be
    written as actual Python lists, not a stringified JSON array — the client
    serialises real lists to jsonb automatically; a string would land as raw text."""
    upsert_match(fake_client, "u1", _match_row(
        emphasize_in_cv=["BI leadership", "CRM ownership"],
        deemphasize=["IT portfolio work"],
    ))
    row = fake_client.table("matches").rows[0]

    assert row["emphasize_in_cv"] == ["BI leadership", "CRM ownership"]
    assert isinstance(row["emphasize_in_cv"], list)
    assert not isinstance(row["emphasize_in_cv"], str)

    assert row["deemphasize"] == ["IT portfolio work"]
    assert isinstance(row["deemphasize"], list)
    assert not isinstance(row["deemphasize"], str)


def test_upsert_payload_writes_empty_list_for_empty_cv_guidance(fake_client):
    upsert_match(fake_client, "u1", _match_row(emphasize_in_cv=[], deemphasize=[]))
    row = fake_client.table("matches").rows[0]
    assert row["emphasize_in_cv"] == []
    assert row["deemphasize"] == []


def test_upsert_payload_writes_null_for_absent_cv_guidance(fake_client):
    upsert_match(fake_client, "u1", _match_row(emphasize_in_cv=None, deemphasize=None))
    row = fake_client.table("matches").rows[0]
    assert row["emphasize_in_cv"] is None
    assert row["deemphasize"] is None


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


# ---------------------------------------------------------------------------
# update_enrichment — targeted update, structurally limited to the 4 enrichment columns
# ---------------------------------------------------------------------------

def test_update_enrichment_payload_contains_only_the_four_target_keys(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)

    update_enrichment(
        fake_client, "u1", row.canonical_url,
        why_fits="- new bullet one\n- new bullet two",
        why_not_fits="- new gap",
        emphasize_in_cv=["short phrase"],
        deemphasize=["other phrase"],
    )

    assert len(fake_client.table("matches").update_calls) == 1
    payload = fake_client.table("matches").update_calls[0]
    assert set(payload.keys()) == _ENRICHMENT_COLUMNS


def test_update_enrichment_uses_update_not_upsert(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)
    upsert_calls_before = len(fake_client.table("matches").upsert_calls)

    update_enrichment(
        fake_client, "u1", row.canonical_url,
        why_fits="- a", why_not_fits="- b", emphasize_in_cv=["x"], deemphasize=["y"],
    )

    assert len(fake_client.table("matches").upsert_calls) == upsert_calls_before


def test_update_enrichment_writes_the_new_values(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)

    update_enrichment(
        fake_client, "u1", row.canonical_url,
        why_fits="- reformatted why", why_not_fits="- reformatted why not",
        emphasize_in_cv=["reformatted emphasize"], deemphasize=["reformatted deemphasize"],
    )

    updated = fake_client.table("matches").rows[0]
    assert updated["why_fits"] == "- reformatted why"
    assert updated["why_not_fits"] == "- reformatted why not"
    assert updated["emphasize_in_cv"] == ["reformatted emphasize"]
    assert updated["deemphasize"] == ["reformatted deemphasize"]


def test_update_enrichment_never_touches_other_columns(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)
    fake_client.table("matches").rows[0]["status"] = "Applied"

    update_enrichment(
        fake_client, "u1", row.canonical_url,
        why_fits="- a", why_not_fits="- b", emphasize_in_cv=["x"], deemphasize=["y"],
    )

    updated = fake_client.table("matches").rows[0]
    assert updated["status"] == "Applied"
    assert updated["fit_score"] == row.fit_score
    assert updated["dimension_scores"] == row.dimension_scores
    assert updated["org_summary"] == row.org_summary


def test_update_enrichment_scoped_to_user_and_canonical_url(fake_client):
    """Two users can have a match at the same canonical_url — updating one must not
    touch the other's row."""
    shared_url = "https://example.org/shared"
    row1 = _match_row(url=shared_url, canonical_url=canonical_url(shared_url))
    row2 = _match_row(url=shared_url, canonical_url=canonical_url(shared_url))
    upsert_match(fake_client, "u1", row1)
    upsert_match(fake_client, "u2", row2)

    update_enrichment(
        fake_client, "u1", canonical_url(shared_url),
        why_fits="- only for u1", why_not_fits="- only for u1",
        emphasize_in_cv=["u1 only"], deemphasize=["u1 only"],
    )

    rows_by_user = {r["user_id"]: r for r in fake_client.table("matches").rows}
    assert rows_by_user["u1"]["why_fits"] == "- only for u1"
    assert rows_by_user["u2"]["why_fits"] == row2.why_fits


def test_update_enrichment_dry_run_does_not_write(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)

    update_enrichment(
        fake_client, "u1", row.canonical_url,
        why_fits="- a", why_not_fits="- b", emphasize_in_cv=["x"], deemphasize=["y"],
        dry_run=True,
    )

    assert fake_client.table("matches").update_calls == []
    assert fake_client.table("matches").rows[0]["why_fits"] == row.why_fits


def test_update_enrichment_no_match_is_a_noop(fake_client):
    """Targeted .update() on a URL that doesn't exist yet must not create a row —
    unlike .upsert(), a non-matching filter is simply a no-op."""
    update_enrichment(
        fake_client, "u1", "https://example.org/never-existed",
        why_fits="- a", why_not_fits="- b", emphasize_in_cv=["x"], deemphasize=["y"],
    )
    assert fake_client.table("matches").rows == []
