"""Hard-constraint gate. Binary location + seniority checks against profile hard_constraints."""
from __future__ import annotations

import re

from src.schemas import HardGateResult, RawPosting

_REMOTE_KEYWORDS = frozenset({
    "remote", "anywhere", "distributed", "work from home", "wfh", "fully remote", "location independent",
})
_SWEDEN_KEYWORDS = frozenset({
    "sweden", "sverige", "stockholm", "göteborg", "gothenburg", "malmö", "malmoe",
    "sundsvall", "linköping", "linkoping", "uppsala", "västerås", "vasteras",
})
_SENIORITY_FAIL_KEYWORDS = frozenset({
    "entry level", "entry-level", "junior", "graduate", "intern", "internship",
    "trainee", "apprentice", "new grad", "new graduate",
})
# Matches "0–4 years" or "0-4 yrs" patterns that signal required experience below threshold.
_LOW_EXP_RE = re.compile(r"\b[0-4]\+?\s*(?:year|yr)s?\b", re.IGNORECASE)


def check(posting: RawPosting, profile: dict) -> HardGateResult:
    loc_constraints = profile["hard_constraints"]["location"]
    sen_constraints = profile["hard_constraints"]["seniority"]

    location_pass = _check_location(posting.location, loc_constraints)
    seniority_pass = _check_seniority(posting.seniority, sen_constraints)
    passed = location_pass and seniority_pass

    if not passed:
        parts = []
        if not location_pass:
            parts.append(f"location: {posting.location!r} does not match any accept condition")
        if not seniority_pass:
            parts.append(f"seniority: {posting.seniority!r} signals below threshold")
        reason = "; ".join(parts)
    else:
        reason = None

    return HardGateResult(
        passed=passed,
        location_pass=location_pass,
        seniority_pass=seniority_pass,
        reason=reason,
    )


def _check_location(location: str | None, constraints: dict) -> bool:
    if not location:
        return True  # unstated → pass (spec: bias toward false-positives)

    text = location.lower()

    if constraints.get("accept_fully_remote") and any(kw in text for kw in _REMOTE_KEYWORDS):
        return True

    if constraints.get("accept_sweden_hybrid"):
        if "hybrid" in text and any(kw in text for kw in _SWEDEN_KEYWORDS):
            return True

    for accepted in constraints.get("accept_onsite_locations", []):
        if accepted.lower() in text:
            return True

    # Any Sweden mention without an explicit outside-Sweden requirement → pass.
    # Soft scoring handles ambiguous Sweden-based roles; gate biases toward false-positives.
    if any(kw in text for kw in _SWEDEN_KEYWORDS):
        return True

    return False


def _check_seniority(seniority: str | None, constraints: dict) -> bool:
    if not seniority:
        return True  # unstated → pass

    text = seniority.lower()

    for kw in _SENIORITY_FAIL_KEYWORDS:
        if kw in text:
            return False

    if _LOW_EXP_RE.search(text):
        return False

    return True
