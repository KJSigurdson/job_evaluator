"""Hard-constraint gate. Binary location + seniority checks against profile hard_constraints."""
from __future__ import annotations

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


def check(posting: RawPosting, profile: dict) -> HardGateResult:
    loc_constraints = profile["hard_constraints"]["location"]

    location_pass = _check_location(posting.location, loc_constraints)
    seniority_pass = _check_seniority(posting.seniority)
    passed = location_pass and seniority_pass

    if not passed:
        parts = []
        if not location_pass:
            parts.append(f"location: {posting.location!r} does not match any accept condition")
        if not seniority_pass:
            parts.append(f"seniority: {posting.seniority!r} signals below threshold")
        reason = "; ".join(parts)
        # Diagnostic-only categorisation for pipeline.py's per-user rejection-reason
        # summary — does not feed back into passed/location_pass/seniority_pass.
        if not location_pass and not seniority_pass:
            rejection_reason = "hard_constraints"
        elif not location_pass:
            rejection_reason = "location"
        else:
            rejection_reason = "seniority"
    else:
        reason = None
        rejection_reason = None

    return HardGateResult(
        passed=passed,
        location_pass=location_pass,
        seniority_pass=seniority_pass,
        reason=reason,
        rejection_reason=rejection_reason,
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


def _check_seniority(seniority: str | None) -> bool:
    if not seniority:
        return True  # unstated → pass

    text = seniority.lower()

    for kw in _SENIORITY_FAIL_KEYWORDS:
        if kw in text:
            return False

    return True
