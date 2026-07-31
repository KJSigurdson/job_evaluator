"""True North v2 hard-constraint gate — the same location + seniority pass/fail
semantics as src/gate.py (bias toward false-positives; unstated -> pass), but
evaluated against a v2 `jobs` row's structured enrichment instead of v1's raw
posting text. v1's gate.py is untouched; this is a separate implementation for
shadow-mode comparison (see src/shadow_compare.py), not a replacement.

Two things v2 does differently, both intentional, not bugs to reconcile:
  - Seniority is read directly from enrichment.seniority (already classified into
    the five buckets by the enrichment LLM call) instead of re-deriving buckets
    from raw text via regex — v1's _LEVEL_KEYWORDS classification step is now
    upstream, in the enrichment prompt, not in the gate itself.
  - "Fully remote" is decided from enrichment.work_arrangement's structured mode/
    remote_scope/eligible_regions instead of a keyword substring match on raw
    location text — v1 cannot distinguish "Remote" from "Remote, US-authorized-
    only" from text alone and would pass both identically; v2 can. This is
    exactly the kind of behavior difference the shadow comparison exists to
    surface, not something to suppress for parity.
Hybrid/onsite matching still falls back to the job's raw location_field text
(region-token matching), same as v1, since JobEnrichment doesn't carry a
structured office-location field.
"""
from __future__ import annotations

import re

from src.gate import _matches_region  # reuse verbatim, don't refork
from src.schemas import HardGateResult

# Coarse ISO-code/region-group -> common name aliases, used only to bridge v2's
# eligible_regions vocabulary ("US", "EU") against a user's free-text region
# tokens (accept_hybrid_in/accept_onsite_in: "Sweden", "Sundsvall") and
# profile["location"]. Deliberately small and imprecise — good enough for a
# shadow-mode comparison tool, not a substitute for real geo resolution.
_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "US": ("us", "usa", "united states", "america"),
    "UK": ("uk", "united kingdom", "britain", "england"),
    "EU": ("eu", "europe", "european union"),
    "CA": ("ca", "canada"),
    "SE": ("se", "sweden"),
    "DE": ("de", "germany"),
    "EMEA": ("emea", "europe", "middle east", "africa"),
}


def check(job: dict, profile: dict) -> HardGateResult:
    """job: a `jobs` row dict with at least location_field and enrichment (the
    JobEnrichment jsonb, as a plain dict — either from Supabase directly or via
    JobEnrichment.model_dump())."""
    loc_constraints = profile["hard_constraints"]["location"]
    seniority_constraints = profile["hard_constraints"]["seniority"]
    enrichment = job.get("enrichment") or {}

    location_pass = _check_location_v2(
        job.get("location_field"), enrichment.get("work_arrangement") or {}, loc_constraints, profile,
    )
    seniority_pass = _check_seniority_v2(enrichment.get("seniority") or [], seniority_constraints)
    passed = location_pass and seniority_pass

    if not passed:
        parts = []
        if not location_pass:
            parts.append(f"location: {job.get('location_field')!r} does not match any accept condition")
        if not seniority_pass:
            parts.append(f"seniority: {enrichment.get('seniority')!r} signals below threshold")
        reason = "; ".join(parts)
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


def _check_location_v2(location_field: str | None, work_arrangement: dict, constraints: dict, profile: dict) -> bool:
    mode = work_arrangement.get("mode", "unspecified")

    if not location_field and mode == "unspecified":
        return True  # genuinely unstated -> pass (bias toward false-positives, same as v1)

    if constraints.get("accept_fully_remote") and mode == "remote":
        remote_scope = work_arrangement.get("remote_scope", "unspecified")
        if remote_scope in ("global", "unspecified"):
            return True
        if remote_scope == "region_restricted":
            if _region_overlap(work_arrangement.get("eligible_regions") or [], constraints, profile):
                return True

    text = (location_field or "").lower()
    if "hybrid" in text and _matches_region(text, constraints.get("accept_hybrid_in", [])):
        return True
    if _matches_region(text, constraints.get("accept_onsite_in", [])):
        return True

    return False


def _region_overlap(eligible_regions: list[str], constraints: dict, profile: dict) -> bool:
    """Best-effort match between v2's ISO/region-group codes and the user's
    free-text region tokens (accept_hybrid_in/accept_onsite_in) plus
    profile["location"]. See this module's docstring for the known limits."""
    user_tokens = [
        t.lower() for t in (constraints.get("accept_hybrid_in", []) + constraints.get("accept_onsite_in", []))
        if t
    ]
    if profile.get("location"):
        user_tokens.append(profile["location"].lower())

    for region in eligible_regions:
        code = region.upper()
        aliases = _REGION_ALIASES.get(code, (region.lower(),))
        for alias in aliases:
            if any(re.search(rf"\b{re.escape(alias)}\b", token) for token in user_tokens):
                return True
    return False


def _check_seniority_v2(seniority_buckets: list[str], constraints: dict) -> bool:
    accept = constraints.get("accept_levels", [])
    if not accept:
        return True  # empty selection = no filter (bias toward false-positives)
    if not seniority_buckets:
        return True  # posting doesn't map to any known level -> pass
    return bool(set(seniority_buckets) & set(accept))
