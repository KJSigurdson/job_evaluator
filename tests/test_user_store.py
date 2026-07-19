"""Tests for user_store.py: profile+weights merge and dict-shape building."""
from __future__ import annotations

import json
import logging

import pytest

from src.user_store import (
    _build_profile,
    _build_rubric,
    _parse_rule,
    _render_experience_text,
    _render_skills_text,
    fetch_users,
)


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
    # location_rule/seniority_rule are `text` columns in real Postgres — PostgREST
    # returns them as JSON strings, not parsed objects. Stored as strings here to match.
    defaults = dict(
        user_id="u1",
        cause_mission_fit=0.25,
        role_function_fit=0.25,
        location_compatibility=0.15,
        seniority_match=0.10,
        comp_adequacy=0.10,
        values_alignment=0.10,
        skill_growth=0.05,
        location_rule=json.dumps({"accept_fully_remote": True, "accept_sweden_hybrid": False, "accept_onsite_locations": []}),
        seniority_rule=json.dumps({"min_years_experience": 5}),
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
# fetch_users(only_user_id=...) — one-off single-user scoping
# ---------------------------------------------------------------------------

def test_fetch_users_only_user_id_returns_only_that_user(fake_client):
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1"), _weights_row(user_id="u2")])

    users = fetch_users(fake_client, only_user_id="u1")
    assert [u.user_id for u in users] == ["u1"]


def test_fetch_users_only_user_id_scopes_experiences(fake_client):
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1"), _weights_row(user_id="u2")])
    fake_client.seed("experiences", [
        {"id": "e1", "user_id": "u1", "kind": "work", "title": "u1's job",
         "organization": "Org1", "start_date": "2020-01-01", "end_date": None,
         "proficiency": None, "sort_order": 0, "notes": None},
        {"id": "e2", "user_id": "u2", "kind": "work", "title": "u2's job",
         "organization": "Org2", "start_date": "2020-01-01", "end_date": None,
         "proficiency": None, "sort_order": 0, "notes": None},
    ])

    [user] = fetch_users(fake_client, only_user_id="u1")
    assert "u1's job" in user.profile["experience_inventory"]
    assert "u2's job" not in user.profile["experience_inventory"]


def test_fetch_users_only_user_id_not_found_returns_empty(fake_client):
    fake_client.seed("profiles", [_profile_row(user_id="u1")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1")])

    assert fetch_users(fake_client, only_user_id="nonexistent") == []


def test_fetch_users_none_only_user_id_is_unaffected(fake_client):
    """only_user_id=None (the default) must produce identical results to omitting it."""
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1"), _weights_row(user_id="u2")])

    assert {u.user_id for u in fetch_users(fake_client, only_user_id=None)} == {"u1", "u2"}


# ---------------------------------------------------------------------------
# _build_profile
# ---------------------------------------------------------------------------

def test_build_profile_uses_weights_row_for_hard_constraints():
    prow = _profile_row(location_constraints="descriptive text, not gate-shaped")
    wrow = _weights_row()
    profile = _build_profile(prow, wrow)

    # location_rule/seniority_rule are stored as JSON strings (real PostgREST shape);
    # _build_profile must parse them into dicts before gate.py ever sees them.
    assert profile["hard_constraints"]["location"] == json.loads(wrow["location_rule"])
    assert profile["hard_constraints"]["seniority"] == json.loads(wrow["seniority_rule"])
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


# ---------------------------------------------------------------------------
# _parse_rule — location_rule/seniority_rule are `text` columns holding JSON
# ---------------------------------------------------------------------------

def test_parse_rule_valid_json_object_string():
    assert _parse_rule('{"accept_fully_remote": true}') == {"accept_fully_remote": True}


def test_parse_rule_none_returns_empty_dict():
    assert _parse_rule(None) == {}


def test_parse_rule_empty_string_returns_empty_dict():
    assert _parse_rule("") == {}


def test_parse_rule_whitespace_only_string_returns_empty_dict():
    assert _parse_rule("   ") == {}


def test_parse_rule_null_literal_string_returns_empty_dict():
    assert _parse_rule("null") == {}


def test_parse_rule_dict_passthrough():
    d = {"min_years_experience": 5}
    assert _parse_rule(d) is d


def test_parse_rule_malformed_json_returns_empty_dict_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = _parse_rule("{not valid json", user_id="u1", field="location_rule")
    assert result == {}
    assert "u1" in caplog.text
    assert "location_rule" in caplog.text


def test_parse_rule_non_object_json_returns_empty_dict_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = _parse_rule("[1, 2, 3]", user_id="u1", field="seniority_rule")
    assert result == {}
    assert "u1" in caplog.text
    assert "seniority_rule" in caplog.text


def test_build_profile_parses_json_string_rules_end_to_end():
    """The exact bug: real PostgREST returns text columns as strings, not dicts.
    gate.py calls .get() on hard_constraints values, so an unparsed string crashes it —
    this asserts _build_profile hands gate.py a dict either way."""
    prow = _profile_row()
    wrow = _weights_row(
        location_rule='{"accept_fully_remote": true, "accept_onsite_locations": ["Sundsvall"]}',
        seniority_rule='{"min_years_experience": 5}',
    )
    profile = _build_profile(prow, wrow)

    assert isinstance(profile["hard_constraints"]["location"], dict)
    assert isinstance(profile["hard_constraints"]["seniority"], dict)
    assert profile["hard_constraints"]["location"]["accept_fully_remote"] is True
    assert profile["hard_constraints"]["seniority"]["min_years_experience"] == 5
    # gate.py calls .get() on this — must not raise
    profile["hard_constraints"]["location"].get("accept_fully_remote")


def test_build_profile_malformed_rule_string_degrades_to_permissive_empty_dict():
    prow = _profile_row()
    wrow = _weights_row(location_rule="{not valid json")
    profile = _build_profile(prow, wrow)
    assert profile["hard_constraints"]["location"] == {}


# ---------------------------------------------------------------------------
# fetch_users — bulk experiences/experience_achievements fetch
# ---------------------------------------------------------------------------

def _experience(**kwargs) -> dict:
    defaults = dict(
        id="exp-1", user_id="u1", kind="work", title="Head of BI",
        organization="Evidensia", start_date="2019-01-01", end_date="2024-01-01",
        proficiency=None, sort_order=0, notes=None,
    )
    defaults.update(kwargs)
    return defaults


def _achievement(**kwargs) -> dict:
    defaults = dict(
        id="ach-1", experience_id="exp-1", user_id="u1",
        description="Grew team from 1 to 7", sort_order=0,
    )
    defaults.update(kwargs)
    return defaults


def test_fetch_users_renders_structured_experience_into_profile(fake_client):
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])
    fake_client.seed("experiences", [_experience()])
    fake_client.seed("experience_achievements", [_achievement()])

    [user] = fetch_users(fake_client)

    assert "Head of BI" in user.profile["experience_inventory"]
    assert "Evidensia" in user.profile["experience_inventory"]
    assert "Grew team from 1 to 7" in user.profile["experience_inventory"]


def test_fetch_users_only_queries_experiences_for_processed_users(fake_client):
    """A profile with no scoring_weights row is never processed, so it shouldn't be
    part of the bulk experiences query (verified indirectly: its experiences, if any,
    must not leak into another user's rendered profile)."""
    fake_client.seed("profiles", [_profile_row(user_id="u1"), _profile_row(user_id="u2")])
    fake_client.seed("scoring_weights", [_weights_row(user_id="u1")])  # u2 has no weights
    fake_client.seed("experiences", [_experience(user_id="u2", title="Should not appear")])

    [user] = fetch_users(fake_client)
    assert user.user_id == "u1"
    assert "Should not appear" not in str(user.profile["experience_inventory"])


# ---------------------------------------------------------------------------
# _render_experience_text / _render_skills_text — grouping, ordering, fallback
# ---------------------------------------------------------------------------

def test_render_experience_text_groups_by_kind_in_fixed_order():
    rows = [
        _experience(id="e1", kind="education", title="MSc Economics", sort_order=0),
        _experience(id="e2", kind="work", title="Head of BI", sort_order=0),
        _experience(id="e3", kind="extracurricular", title="Board member", sort_order=0),
    ]
    text = _render_experience_text(rows, {})
    assert text.index("Work:") < text.index("Education:") < text.index("Extracurricular:")


def test_render_experience_text_orders_entries_by_sort_order():
    rows = [
        _experience(id="e1", title="Second Job", sort_order=1),
        _experience(id="e2", title="First Job", sort_order=0),
    ]
    text = _render_experience_text(rows, {})
    assert text.index("First Job") < text.index("Second Job")


def test_render_experience_text_nests_achievements_under_their_experience():
    rows = [_experience(id="e1", title="Head of BI")]
    achievements = {"e1": [_achievement(experience_id="e1", description="Built the BI function")]}
    text = _render_experience_text(rows, achievements)
    assert text.index("Head of BI") < text.index("Built the BI function")


def test_render_experience_text_includes_notes():
    rows = [_experience(id="e1", notes="Promoted twice")]
    text = _render_experience_text(rows, {})
    assert "Promoted twice" in text


def test_render_experience_text_handles_missing_organization():
    rows = [_experience(id="e1", title="Freelance Consultant", organization=None)]
    text = _render_experience_text(rows, {})
    assert "Freelance Consultant (2019-01-01–2024-01-01)" in text


def test_render_experience_text_present_when_no_end_date():
    rows = [_experience(id="e1", end_date=None)]
    text = _render_experience_text(rows, {})
    assert "Present" in text


def test_render_experience_text_none_when_no_work_kind_rows():
    rows = [_experience(id="e1", kind="skill", title="SQL")]
    assert _render_experience_text(rows, {}) is None


def test_render_experience_text_none_for_empty_list():
    assert _render_experience_text([], {}) is None


def test_render_skills_text_groups_by_kind_in_fixed_order():
    rows = [
        _experience(id="s1", kind="language", title="Swedish", proficiency="fluent", sort_order=0),
        _experience(id="s2", kind="skill", title="Stakeholder management", proficiency="expert", sort_order=0),
        _experience(id="s3", kind="software_skill", title="Power BI", proficiency="expert", sort_order=0),
    ]
    text = _render_skills_text(rows)
    assert text.index("Skills:") < text.index("Software:") < text.index("Languages:")
    assert "Power BI (expert)" in text
    assert "Swedish (fluent)" in text


def test_render_skills_text_orders_entries_by_sort_order():
    rows = [
        _experience(id="s1", kind="skill", title="Second Skill", proficiency="working_proficiency", sort_order=1),
        _experience(id="s2", kind="skill", title="First Skill", proficiency="expert", sort_order=0),
    ]
    text = _render_skills_text(rows)
    assert text.index("First Skill") < text.index("Second Skill")


def test_render_skills_text_none_when_no_skill_kind_rows():
    rows = [_experience(id="e1", kind="work")]
    assert _render_skills_text(rows) is None


# ---------------------------------------------------------------------------
# Per-field fallback to free-text profiles.experience / profiles.skills
# ---------------------------------------------------------------------------

def test_build_profile_falls_back_to_free_text_when_no_experiences_rows():
    prow = _profile_row(experience="Free-text experience blob", skills="Free-text skills blob")
    profile = _build_profile(prow, _weights_row(), experiences=[])
    assert profile["experience_inventory"] == "Free-text experience blob"
    assert profile["skills"] == "Free-text skills blob"


def test_build_profile_structured_experience_with_free_text_skills_fallback():
    """Structured work rows exist but no structured skill-kind rows → experience is
    rendered, skills falls back to the free-text profiles.skills column."""
    prow = _profile_row(skills="Free-text skills blob")
    experiences = [_experience(id="e1", kind="work", title="Head of BI")]

    profile = _build_profile(prow, _weights_row(), experiences=experiences)

    assert "Head of BI" in profile["experience_inventory"]
    assert profile["skills"] == "Free-text skills blob"


def test_build_profile_structured_skills_with_free_text_experience_fallback():
    prow = _profile_row(experience="Free-text experience blob")
    experiences = [_experience(id="s1", kind="skill", title="SQL", proficiency="expert")]

    profile = _build_profile(prow, _weights_row(), experiences=experiences)

    assert profile["experience_inventory"] == "Free-text experience blob"
    assert "SQL (expert)" in profile["skills"]
