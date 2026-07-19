"""Fetch users from Supabase and build the profile/rubric dicts gate.py, scoring.py,
and enrich.py expect (they take plain dicts — previously loaded from profile.yaml /
rubric.yaml, now built from `profiles` + `scoring_weights` rows, enriched with
structured `experiences` / `experience_achievements` rows where available).

A user is only processed if they have BOTH a `profiles` row and a `scoring_weights`
row — scoring is impossible without weights, so a profile with no weights row is
skipped (logged, not fatal).
"""
from __future__ import annotations

import logging

from src.schemas import UserContext

log = logging.getLogger(__name__)

_WORK_KINDS = ("work", "education", "extracurricular")
_SKILL_KINDS = ("skill", "software_skill", "language")

_EXPERIENCE_HEADERS = {"work": "Work", "education": "Education", "extracurricular": "Extracurricular"}
_SKILL_HEADERS = {"skill": "Skills", "software_skill": "Software", "language": "Languages"}

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


def fetch_users(client, only_user_id: str | None = None) -> list[UserContext]:
    """Fetch every user with a profile row and matching scoring_weights row.

    only_user_id — scope every query (profiles, scoring_weights, experiences,
    experience_achievements) to a single user, for a one-off single-user run. When
    None (the daily-cron default), behaviour is exactly as before this option existed.
    """
    profiles_query = client.table("profiles").select("*")
    weights_query = client.table("scoring_weights").select("*")
    if only_user_id is not None:
        profiles_query = profiles_query.eq("user_id", only_user_id)
        weights_query = weights_query.eq("user_id", only_user_id)

    profile_rows = profiles_query.execute().data or []
    weights_rows = weights_query.execute().data or []
    weights_by_user = {row["user_id"]: row for row in weights_rows}

    processed_ids = [prow["user_id"] for prow in profile_rows if prow["user_id"] in weights_by_user]
    experiences_by_user, achievements_by_experience = _fetch_experiences(client, processed_ids, only_user_id)

    users: list[UserContext] = []
    for prow in profile_rows:
        uid = prow["user_id"]
        wrow = weights_by_user.get(uid)
        if wrow is None:
            log.warning("Skipping user %s: profile row with no scoring_weights row", uid)
            continue
        users.append(UserContext(
            user_id=uid,
            profile=_build_profile(
                prow, wrow,
                experiences_by_user.get(uid, []),
                achievements_by_experience,
            ),
            rubric=_build_rubric(wrow),
        ))

    log.info("Loaded %d user(s) with profile + weights", len(users))
    return users


def _fetch_experiences(
    client,
    user_ids: list[str],
    only_user_id: str | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Bulk-fetch `experiences` + `experience_achievements` for *user_ids*, grouped in
    memory — one query per table, not one per user. When *only_user_id* is set, both
    queries are scoped to that single user (.eq) rather than the bulk (.in_) filter."""
    if not user_ids:
        return {}, {}

    experiences_query = client.table("experiences").select("*")
    achievements_query = client.table("experience_achievements").select("*")
    if only_user_id is not None:
        experiences_query = experiences_query.eq("user_id", only_user_id)
        achievements_query = achievements_query.eq("user_id", only_user_id)
    else:
        experiences_query = experiences_query.in_("user_id", user_ids)
        achievements_query = achievements_query.in_("user_id", user_ids)

    experience_rows = experiences_query.execute().data or []
    achievement_rows = achievements_query.execute().data or []

    experiences_by_user: dict[str, list[dict]] = {}
    for row in experience_rows:
        experiences_by_user.setdefault(row["user_id"], []).append(row)

    achievements_by_experience: dict[str, list[dict]] = {}
    for row in achievement_rows:
        achievements_by_experience.setdefault(row["experience_id"], []).append(row)
    for rows in achievements_by_experience.values():
        rows.sort(key=lambda r: r.get("sort_order", 0))

    return experiences_by_user, achievements_by_experience


def _build_profile(
    prow: dict,
    wrow: dict,
    experiences: list[dict] = (),
    achievements_by_experience: dict[str, list[dict]] | None = None,
) -> dict:
    """Map `profiles` + `scoring_weights` (+ `experiences`/`experience_achievements`)
    rows onto the shape gate.py / scoring.py / enrich.py expect.

    gate.py reads profile["hard_constraints"]["location"] as
    {accept_fully_remote, accept_sweden_hybrid, accept_onsite_locations} — that's the
    shape scoring_weights.location_rule holds (the gate's structured pass/fail config).
    profiles.location_constraints is separate: free-form descriptive context (e.g.
    "prefers remote or Sweden-based hybrid") folded into the LLM prompt, not used by
    the gate itself. Same split for seniority: scoring_weights.seniority_rule is the
    (currently gate-inert — see gate._check_seniority) structured rule;
    profiles.seniority_level is descriptive context.

    "experience_inventory" and "skills" prefer structured `experiences` rows when any
    exist for the relevant kind-group; otherwise they fall back to the free-text
    profiles.experience / profiles.skills columns. The fallback is per-field: a user
    with structured work/education rows but no structured skill rows gets structured
    experience text + free-text skills, and vice versa.
    """
    achievements_by_experience = achievements_by_experience or {}
    experience_text = _render_experience_text(experiences, achievements_by_experience)
    skills_text = _render_skills_text(experiences)

    return {
        "location": prow.get("location"),
        "location_constraints": prow.get("location_constraints"),
        "seniority_level": prow.get("seniority_level"),
        "comp_needs": prow.get("comp_needs"),
        "cause_priorities": prow.get("cause_priorities"),
        "values_notes": prow.get("values_notes"),
        "career_goals": prow.get("career_goals"),
        "skills": skills_text if skills_text is not None else prow.get("skills"),
        "experience_inventory": experience_text if experience_text is not None else prow.get("experience"),
        "hard_constraints": {
            "location": wrow.get("location_rule") or {},
            "seniority": wrow.get("seniority_rule"),
        },
    }


def _format_experience_entry(row: dict) -> str:
    title = row["title"]
    org = row.get("organization")
    start = row.get("start_date")
    end = row.get("end_date") or "Present"
    header = f"{title} — {org}" if org else title
    return f"{header} ({start}–{end})"


def _render_experience_text(experiences: list[dict], achievements_by_experience: dict[str, list[dict]]) -> str | None:
    """Render `work`/`education`/`extracurricular` experiences rows into readable text,
    grouped under fixed headers in a fixed order, entries ordered by sort_order.
    Returns None if the user has no rows of these kinds (triggers the free-text fallback).
    """
    by_kind: dict[str, list[dict]] = {}
    for row in experiences:
        if row["kind"] in _WORK_KINDS:
            by_kind.setdefault(row["kind"], []).append(row)
    if not by_kind:
        return None

    sections: list[str] = []
    for kind in _WORK_KINDS:
        entries = sorted(by_kind.get(kind, []), key=lambda r: r.get("sort_order", 0))
        if not entries:
            continue
        lines = [f"{_EXPERIENCE_HEADERS[kind]}:"]
        for entry in entries:
            lines.append(f"- {_format_experience_entry(entry)}")
            for achievement in achievements_by_experience.get(entry["id"], []):
                lines.append(f"  * {achievement['description']}")
            if entry.get("notes"):
                lines.append(f"  {entry['notes']}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _render_skills_text(experiences: list[dict]) -> str | None:
    """Render `skill`/`software_skill`/`language` experiences rows into readable text,
    grouped under fixed headers in a fixed order, entries ordered by sort_order.
    Returns None if the user has no rows of these kinds (triggers the free-text fallback).
    """
    by_kind: dict[str, list[dict]] = {}
    for row in experiences:
        if row["kind"] in _SKILL_KINDS:
            by_kind.setdefault(row["kind"], []).append(row)
    if not by_kind:
        return None

    sections: list[str] = []
    for kind in _SKILL_KINDS:
        entries = sorted(by_kind.get(kind, []), key=lambda r: r.get("sort_order", 0))
        if not entries:
            continue
        lines = [f"{_SKILL_HEADERS[kind]}:"]
        for entry in entries:
            lines.append(f"- {entry['title']} ({entry.get('proficiency')})")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


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
