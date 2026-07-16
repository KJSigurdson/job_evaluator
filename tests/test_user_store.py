"""Tests for user_store.py: profile+weights merge and dict-shape building."""
from __future__ import annotations

import logging

import pytest

from src.user_store import _build_profile, _build_rubric, fetch_users


def _profile_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1",
        experience=["Built a BI function"],
        skills=["sql", "python"],
        career_goals="Data leadership",
        cause_priorities=["global health"],
        location="Berlin, Germany",
        location_constraints="Remote preferred",
        seniority_level="Senior",
        comp_needs="Market rate",
        values_notes="GWWC pledge",
    )
    defaults.update(kwargs)
    return defaults


def _weights_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1",
        cause_mission_fit=0.25,
        role_function_fit=0.25,
        location_compatibility=0.15,
        seniority_match=0.10,
        comp_adequacy=0.10,
        values_alignment=0.10,
        skill_growth=0.05,
        location_rule={"accept_fully_remote": True, "accept_sweden_hybrid": False, "accept_onsite_locations": []},
        seniority_rule={"min_years_experience": 5},
        insert_threshold=0.75,
        near_miss_floor=0.65,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# fetch_users
# ---------------------------------------------------------------------------

def test_fetch_users_merges_profile_and_weights(fake_client):
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])

    users = fetch_users(fake_client)

    assert len(users) == 1
    assert users[0].user_id == "u1"


def test_fetch_users_skips_profile_with_no_weights(fake_client, caplog):
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1")])

    with caplog.at_level(logging.WARNING):
        users = fetch_users(fake_client)

    assert [u.user_id for u in users] == ["u1"]
    assert "u2" in caplog.text


def test_fetch_users_empty_when_no_profiles(fake_client):
    assert fetch_users(fake_client) == []


def test_fetch_users_multiple_users(fake_client):
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1"), _weights_row(user_id="u2")])

    users = fetch_users(fake_client)
    assert {u.user_id for u in users} == {"u1", "u2"}


# ---------------------------------------------------------------------------
# _build_profile
# ---------------------------------------------------------------------------

def test_build_profile_uses_weights_row_for_hard_constraints():
    prow = _profile_row(location_constraints="descriptive text, not gate-shaped")
    wrow = _weights_row()
    profile = _build_profile(prow, wrow)

    assert profile["hard_constraints"]["location"] == wrow["location_rule"]
    assert profile["hard_constraints"]["seniority"] == wrow["seniority_rule"]
    # descriptive field carried through separately for LLM context, untouched
    assert profile["location_constraints"] == "descriptive text, not gate-shaped"


def test_build_profile_defaults_missing_location_rule_to_empty_dict():
    prow = _profile_row()
    wrow = _weights_row(location_rule=None)
    profile = _build_profile(prow, wrow)
    assert profile["hard_constraints"]["location"] == {}


# ---------------------------------------------------------------------------
# _build_rubric
# ---------------------------------------------------------------------------

def test_build_rubric_shape():
    rubric = _build_rubric(_weights_row())
    assert rubric["threshold"] == 0.75
    assert rubric["near_miss_min"] == 0.65
    assert set(rubric["dimensions"].keys()) == {
        "cause_mission_fit", "role_function_fit", "location_compatibility",
        "seniority_match", "comp_adequacy", "values_alignment", "skill_growth",
    }
    assert rubric["dimensions"]["cause_mission_fit"]["weight"] == 0.25
    assert rubric["dimensions"]["cause_mission_fit"]["description"]


def test_build_rubric_warns_when_weights_dont_sum_to_one(caplog):
    wrow = _weights_row(cause_mission_fit=0.9)  # now sums well above 1.0
    with caplog.at_level(logging.WARNING):
        _build_rubric(wrow)
    assert "sum to" in caplog.text
