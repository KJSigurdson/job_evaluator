"""Tests for quota_store.py: search_quota status write-back."""
from __future__ import annotations

import pytest

from src.quota_store import set_status


def test_set_status_upserts_row(fake_client):
    set_status(fake_client, "u1", "running")
    row = fake_client.table("search_quota").rows[0]
    assert row["user_id"] == "u1"
    assert row["status"] == "running"


def test_set_status_updates_not_duplicates(fake_client):
    set_status(fake_client, "u1", "running")
    set_status(fake_client, "u1", "complete")
    rows = fake_client.table("search_quota").rows
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"


def test_set_status_completed_sets_completed_at(fake_client):
    set_status(fake_client, "u1", "complete", completed=True)
    row = fake_client.table("search_quota").rows[0]
    assert row["completed_at"]


def test_set_status_not_completed_omits_completed_at(fake_client):
    set_status(fake_client, "u1", "running")
    row = fake_client.table("search_quota").rows[0]
    assert "completed_at" not in row


def test_set_status_rejects_invalid_status(fake_client):
    with pytest.raises(ValueError):
        set_status(fake_client, "u1", "bogus_status")


def test_set_status_all_valid_statuses_accepted(fake_client):
    for status in ("not_started", "queued", "running", "complete", "failed"):
        set_status(fake_client, "u1", status)
        assert fake_client.table("search_quota").rows[0]["status"] == status
