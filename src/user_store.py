"""Fetch users from Supabase and build the profile/rubric dicts gate.py, scoring.py,
and enrich.py expect (they take plain dicts — previously loaded from profile.yaml /
rubric.yaml, now built from `profiles` + `scoring_weights` rows).

A user is only processed if they have BOTH a `profiles` row and a `scoring_weights`
row — scoring is impossible without weights, so a profile with no weights row is
skipped (logged, not fatal).
"""
from __future__ import annotations

import logging

from src.schemas import UserContext

log = logging.getLogger(__name__)

_DIMENSIONS = (
    "cause_mission_fit",
    "role_function_fit",
    "location_compatibility",
    "seniority_match",
    "comp_adequacy",
    "values_alignment",
    "skill_growth",
)

# Static prompt text for each dimension — shared across all users' Tier 1 LLM calls.
# Deliberately user-agnostic (unlike the old single-user rubric.yaml, which hardcoded
# Sweden-specific phrasing); per-user weighting comes from scoring_weights, per-user
# constraints from the profile itself.
_DIMENSION_DESCRIPTIONS = {
    "cause_mission_fit": "Alignment with the candidate's EA-aligned cause priorities.",
    "role_function_fit": "Match on job function to the candidate's skills and career goals.",
    "location_compatibility": "Remote/hybrid/on-site compatibility with the candidate's location constraints.",
    "seniority_match": "Role seniority consistent with the candidate's experience level.",
    "comp_adequacy": "Compensation adequacy relative to the candidate's stated comp needs.",
    "values_alignment": "Org culture and mission alignment with the candidate's values.",
    "skill_growth": "Opportunity to close the candidate's acknowledged skill gaps.",
}


def fetch_users(client) -> list[UserContext]:
    """Fetch every user with a profile row and matching scoring_weights row."""
    profile_rows = client.table("profiles").select("*").execute().data or []
    weights_rows = client.table("scoring_weights").select("*").execute().data or []
    weights_by_user = {row["user_id"]: row for row in weights_rows}

    users: list[UserContext] = []
    for prow in profile_rows:
        uid = prow["user_id"]
        wrow = weights_by_user.get(uid)
        if wrow is None:
            log.warning("Skipping user %s: profile row with no scoring_weights row", uid)
            continue
        users.append(UserContext(
            user_id=uid,
            profile=_build_profile(prow, wrow),
            rubric=_build_rubric(wrow),
        ))

    log.info("Loaded %d user(s) with profile + weights", len(users))
    return users


def _build_profile(prow: dict, wrow: dict) -> dict:
    """Map `profiles` + `scoring_weights` rows onto the shape gate.py / scoring.py /
    enrich.py expect.

    gate.py reads profile["hard_constraints"]["location"] as
    {accept_fully_remote, accept_sweden_hybrid, accept_onsite_locations} — that's the
    shape scoring_weights.location_rule holds (the gate's structured pass/fail config).
    profiles.location_constraints is separate: free-form descriptive context (e.g.
    "prefers remote or Sweden-based hybrid") folded into the LLM prompt, not used by
    the gate itself. Same split for seniority: scoring_weights.seniority_rule is the
    (currently gate-inert — see gate._check_seniority) structured rule;
    profiles.seniority_level is descriptive context.
    """
    return {
        "location": prow.get("location"),
        "location_constraints": prow.get("location_constraints"),
        "seniority_level": prow.get("seniority_level"),
        "comp_needs": prow.get("comp_needs"),
        "cause_priorities": prow.get("cause_priorities"),
        "values_notes": prow.get("values_notes"),
        "career_goals": prow.get("career_goals"),
        "skills": prow.get("skills"),
        "experience_inventory": prow.get("experience"),
        "hard_constraints": {
            "location": wrow.get("location_rule") or {},
            "seniority": wrow.get("seniority_rule"),
        },
    }


def _build_rubric(row: dict) -> dict:
    """Map a `scoring_weights` row onto the shape scoring.py expects."""
    weight_sum = sum(float(row[dim]) for dim in _DIMENSIONS)
    if abs(weight_sum - 1.0) > 0.01:
        log.warning(
            "scoring_weights for user %s: dimension weights sum to %.3f, expected 1.0",
            row.get("user_id"), weight_sum,
        )

    return {
        "threshold": row["insert_threshold"],
        "near_miss_min": row["near_miss_floor"],
        "dimensions": {
            dim: {"weight": row[dim], "description": _DIMENSION_DESCRIPTIONS[dim]}
            for dim in _DIMENSIONS
        },
    }
