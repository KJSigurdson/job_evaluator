"""Fetch users from Supabase and build the profile/rubric dicts gate.py, scoring.py,
and enrich.py expect (they take plain dicts — previously loaded from profile.yaml /
rubric.yaml, now built from `profiles` + `scoring_weights` rows, enriched with
structured `experiences` / `experience_achievements` rows where available).

A user is only processed if they have BOTH a `profiles` row and a `scoring_weights`
row — scoring is impossible without weights, so a profile with no weights row is
skipped (logged, not fatal). A user whose stored gate config (location_rule/
seniority_rule) is present but fails to parse is ALSO excluded — see
GateConfigError and fetch_users' docstring — since a broken gate config falling
back to {} would silently pass every posting for that user while looking healthy.
"""
from __future__ import annotations

import json
import logging

from src.schemas import ExcludedUser, UserContext

log = logging.getLogger(__name__)


class GateConfigError(Exception):
    """Raised by _parse_rule() when a user's stored location_rule/seniority_rule is
    PRESENT but malformed — not the same as an absent/empty value (None, or an
    empty/whitespace string), which is a legitimate "constraint unstated" case and
    stays permissive (an empty dict, per gate.py's bias-toward-false-positives
    design). A malformed non-empty value is different: silently falling back to {}
    there would make a broken gate config indistinguishable from an intentionally
    unrestricted one — every posting would pass the gate for that user while the
    run looked completely healthy. fetch_users() catches this, excludes the
    affected user from the run entirely, and logs an end-of-run summary."""

    def __init__(self, user_id, field, raw_value, reason):
        self.user_id = user_id
        self.field = field
        self.raw_value = raw_value
        self.reason = reason
        super().__init__(f"user {user_id}: {field} is malformed ({reason}): {raw_value!r}")

_WORK_KINDS = ("work", "education", "extracurricular", "publication")
_SKILL_KINDS = ("skill", "software_skill", "language")

_EXPERIENCE_HEADERS = {
    "work": "Work", "education": "Education", "extracurricular": "Extracurricular",
    "publication": "Publications & Writing",
}
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


def _parse_rule(value, *, user_id=None, field=None) -> dict:
    """Parse a scoring_weights rule column (location_rule / seniority_rule) into a dict.

    These are `text` columns holding JSON — PostgREST returns them as plain strings, not
    parsed objects. gate.py calls `.get()` on the parsed result, so an unparsed string
    crashes with AttributeError. An empty dict is the correct permissive default for a
    genuinely UNSTATED constraint (None, or an empty/whitespace string) — the gate
    treats unstated/empty constraints as a pass (see gate._check_location/_check_seniority).

    A value that IS present but fails to parse (invalid JSON, or valid JSON that isn't
    an object) is a different case: that's malformed data, not an unstated constraint,
    and raises GateConfigError rather than silently degrading to the same permissive
    {} — see GateConfigError's docstring for why silently swallowing this would be
    worse than raising. Logged at ERROR (not the old WARNING) with the user_id and raw
    stored value, since this is now a run-affecting condition, not just a data quirk.

    One deliberate exception: the JSON literal `null` (the four-character string
    "null", not a missing/None value) parses to Python None, not a dict — but it's a
    conventional way to explicitly encode "no rule configured" into a JSON-typed text
    column, not corrupted data, so it stays in the permissive-{} bucket alongside a
    real None and an empty string, rather than raising.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error(
                "scoring_weights for user %s: %s is not valid JSON — excluding user from "
                "this run: %r",
                user_id, field, value,
            )
            raise GateConfigError(user_id, field, value, f"invalid JSON: {exc}") from exc
        if parsed is None:
            return {}  # JSON literal `null` — conventional "unset", not malformed
        if not isinstance(parsed, dict):
            log.error(
                "scoring_weights for user %s: %s parsed to non-object JSON — excluding "
                "user from this run: %r",
                user_id, field, value,
            )
            raise GateConfigError(user_id, field, value, "parsed to non-object JSON")
        return parsed
    log.error(
        "scoring_weights for user %s: %s has unexpected type %s — excluding user from "
        "this run: %r",
        user_id, field, type(value).__name__, value,
    )
    raise GateConfigError(user_id, field, value, f"unexpected type {type(value).__name__}")


def fetch_users(
    client, only_user_id: str | None = None,
) -> tuple[list[UserContext], list[ExcludedUser]]:
    """Fetch every user with a profile row and matching scoring_weights row.

    only_user_id — scope every query (profiles, scoring_weights, experiences,
    experience_achievements) to a single user, for a one-off single-user run. When
    set, that user is fetched regardless of archived_at — an explicit request
    overrides archived status, the same way only_user_id already overrides every
    other cron default.

    On the daily-cron path (only_user_id=None), profiles rows with a non-null
    archived_at are excluded at the query level (not filtered in Python afterward):
    archived users must not spend gate/scoring cycles, and filtering at source also
    keeps _fetch_experiences from pulling data for a user who's about to be dropped
    anyway.

    Returns (users, excluded) — excluded lists every user dropped because their
    stored location_rule/seniority_rule failed to parse (see GateConfigError). This
    is a data-integrity problem, not a normal skip, so it's surfaced separately from
    the quieter "no scoring_weights row" skip (which stays a warning-only, no-entry
    case) so callers can log a run-level summary and — for --only-user-id — raise a
    clear error instead of the generic "no row found" message.
    """
    profiles_query = client.table("profiles").select("*")
    weights_query = client.table("scoring_weights").select("*")
    if only_user_id is not None:
        profiles_query = profiles_query.eq("user_id", only_user_id)
        weights_query = weights_query.eq("user_id", only_user_id)
    else:
        profiles_query = profiles_query.is_("archived_at", "null")

    profile_rows = profiles_query.execute().data or []
    weights_rows = weights_query.execute().data or []
    weights_by_user = {row["user_id"]: row for row in weights_rows}

    processed_ids = [prow["user_id"] for prow in profile_rows if prow["user_id"] in weights_by_user]
    experiences_by_user, achievements_by_experience = _fetch_experiences(client, processed_ids, only_user_id)

    users: list[UserContext] = []
    excluded: list[ExcludedUser] = []
    for prow in profile_rows:
        uid = prow["user_id"]
        wrow = weights_by_user.get(uid)
        if wrow is None:
            log.warning("Skipping user %s: profile row with no scoring_weights row", uid)
            continue
        try:
            profile = _build_profile(
                prow, wrow,
                experiences_by_user.get(uid, []),
                achievements_by_experience,
            )
        except GateConfigError as exc:
            excluded.append(ExcludedUser(user_id=exc.user_id, field=exc.field, reason=exc.reason))
            continue
        users.append(UserContext(user_id=uid, profile=profile, rubric=_build_rubric(wrow)))

    log.info("Loaded %d user(s) with profile + weights", len(users))
    if excluded:
        log.error(
            "Excluded %d user(s) from this run due to malformed gate config: %s",
            len(excluded),
            ", ".join(f"{e.user_id} ({e.field}: {e.reason})" for e in excluded),
        )
    return users, excluded


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
    This structured rule is also the only location-preference context reaching the LLM
    prompts: it rides the wholesale profile dict dump in scoring.py/enrich.py, so
    profiles.location_constraints (a free-text field the frontend no longer writes) is
    not read here — it would be redundant with hard_constraints["location"]. Same split
    for seniority: scoring_weights.seniority_rule is the structured rule
    gate._check_seniority reads — {accept_levels: [...]}, a subset of
    intern/junior/mid/senior/director — while profiles.seniority_level remains
    descriptive LLM context, not consumed by the gate.

    location_rule/seniority_rule are `text` columns holding JSON — PostgREST returns
    them as plain strings, not parsed objects — so both are run through _parse_rule()
    before being handed to gate.py, which calls .get() on them and would otherwise
    crash on a raw string.

    "experience_inventory" and "skills" prefer structured `experiences` rows when any
    exist for the relevant kind-group; otherwise they fall back to the free-text
    profiles.experience / profiles.skills columns. The fallback is per-field: a user
    with structured work/education rows but no structured skill rows gets structured
    experience text + free-text skills, and vice versa.

    "email_on_match" is the opt-in flag pipeline.py checks before sending a match-
    digest email (src/notify.py) — normalised to a strict bool so a missing/null
    column defaults to no email, not an error.
    """
    achievements_by_experience = achievements_by_experience or {}
    experience_text = _render_experience_text(experiences, achievements_by_experience)
    skills_text = _render_skills_text(experiences)

    return {
        "location": prow.get("location"),
        "seniority_level": prow.get("seniority_level"),
        "comp_needs": prow.get("comp_needs"),
        "cause_priorities": prow.get("cause_priorities"),
        "values_notes": prow.get("values_notes"),
        "career_goals": prow.get("career_goals"),
        "email_on_match": bool(prow.get("email_on_match")),
        "skills": skills_text if skills_text is not None else prow.get("skills"),
        "experience_inventory": experience_text if experience_text is not None else prow.get("experience"),
        "hard_constraints": {
            "location": _parse_rule(wrow.get("location_rule"), user_id=wrow.get("user_id"), field="location_rule"),
            "seniority": _parse_rule(wrow.get("seniority_rule"), user_id=wrow.get("user_id"), field="seniority_rule"),
        },
    }


def _format_experience_entry(row: dict) -> str:
    title = row["title"]
    org = row.get("organization")
    start = row.get("start_date")
    end = row.get("end_date") or "Present"
    header = f"{title} — {org}" if org else title
    return f"{header} ({start}–{end})"


def _format_publication_entry(row: dict) -> str:
    """A publication is a point in time, not a range — venue (organization) and a
    single date, no end_date/"Present"."""
    title = row["title"]
    venue = row.get("organization")
    date = row.get("start_date")
    header = f"{title} — {venue}" if venue else title
    return f"{header} ({date})"


def _render_experience_text(experiences: list[dict], achievements_by_experience: dict[str, list[dict]]) -> str | None:
    """Render `work`/`education`/`extracurricular`/`publication` experiences rows into
    readable text, grouped under fixed headers in a fixed order, entries ordered by
    sort_order. Returns None if the user has no rows of these kinds (triggers the
    free-text fallback).
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
            entry_line = _format_publication_entry(entry) if kind == "publication" else _format_experience_entry(entry)
            lines.append(f"- {entry_line}")
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
