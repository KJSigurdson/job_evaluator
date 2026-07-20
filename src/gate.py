"""Hard-constraint gate. Binary location + seniority checks against profile hard_constraints.

Seniority is per-user, mirroring location: a posting's seniority text is classified
into level buckets (intern/junior/mid/senior/director), and the gate passes unless
the posting's classified level(s) are entirely outside the user's accept_levels.
"""
from __future__ import annotations

import re

from src.schemas import HardGateResult, RawPosting

_REMOTE_KEYWORDS = frozenset({
    "remote", "anywhere", "distributed", "work from home", "wfh", "fully remote", "location independent",
})

# Level -> keywords that signal a posting is at that seniority level. "lead" -> senior
# and "principal" -> director are deliberate design choices, not oversights.
_LEVEL_KEYWORDS = {
    "intern":   ["intern", "internship", "trainee", "apprentice", "new grad", "new graduate", "graduate"],
    "junior":   ["junior", "jr", "entry level", "entry-level"],
    "mid":      ["mid", "mid-level", "mid level", "intermediate"],
    "senior":   ["senior", "sr", "lead", "staff"],
    "director": ["director", "head of", "principal", "vp", "vice president", "chief"],
}


def check(posting: RawPosting, profile: dict) -> HardGateResult:
    loc_constraints = profile["hard_constraints"]["location"]
    seniority_constraints = profile["hard_constraints"]["seniority"]

    location_pass = _check_location(posting.location, loc_constraints)
    seniority_pass = _check_seniority(posting.seniority, seniority_constraints)
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

    # fully remote — substring match (not word-boundary): "remotely"/"remote-first" etc.
    if constraints.get("accept_fully_remote") and any(kw in text for kw in _REMOTE_KEYWORDS):
        return True

    # hybrid — requires the word "hybrid" AND a region-token match
    if "hybrid" in text and _matches_region(text, constraints.get("accept_hybrid_in", [])):
        return True

    # onsite — region-token match, no keyword required
    if _matches_region(text, constraints.get("accept_onsite_in", [])):
        return True

    return False


def _matches_region(text: str, tokens: list[str]) -> bool:
    """Word-boundary match against a list of user-selected region tokens (country,
    city, etc.). Raw substring matching would false-match short tokens like "US"
    inside "Russia" or "UK" inside "Fukuoka" — same lesson as _levels_in_text.
    """
    return any(re.search(rf"\b{re.escape(t.lower())}\b", text) for t in tokens if t)


def _levels_in_text(text: str) -> set[str]:
    """Return the set of seniority-level buckets whose keywords appear in *text*.

    Word-boundary matching (not plain substring `in`) is required: short tokens like
    "sr", "vp", "jr", "lead", "mid" would otherwise false-match inside unrelated words
    — e.g. "mid" inside "amid" or "Admin", "lead" inside "leadership".
    """
    lowered = text.lower()
    matched: set[str] = set()
    for level, keywords in _LEVEL_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                matched.add(level)
                break
    return matched


def _check_seniority(seniority: str | None, constraints: dict) -> bool:
    accept = constraints.get("accept_levels", [])
    if not accept:
        return True  # empty selection = no filter (bias toward false-positives)

    if not seniority:
        return True  # unstated posting → pass

    matched = _levels_in_text(seniority)
    if not matched:
        return True  # posting text doesn't map to any known level → pass

    return bool(matched & set(accept))
