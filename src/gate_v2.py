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
from dataclasses import dataclass

from src.gate import _matches_region  # reuse verbatim, don't refork

# GLOBAL FALLBACK posting_type rule (named _DEFAULT_EXCLUDED_POSTING_TYPES,
# not _EXCLUDED_POSTING_TYPES, since 2026-11-04 — see _check_posting_type)
# — applied only for a user who hasn't answered profiles.opportunity_types
# (null = unanswered). {volunteer, funding, speculative} excluded by
# default — boards mix real jobs with these opportunity types, and none
# belongs in a "job match" list by default. speculative added after Animal
# Advocacy Careers' recon: several of its listings are literal "general
# application"/"expression of interest"/"talent pool" posts with no
# specific role (2 had zero description text after stripping AAC's own "No
# description found" placeholder) — exactly what the enrichment prompt's
# own speculative definition describes (see enrich_jobs.py's posting_type
# prompt section), so they'd otherwise reach Tier 1 scoring despite not
# being individual job openings. program is DELIBERATELY NOT excluded by
# default — see enrich_jobs.py's prompt: it's a structured course/
# accelerator/incubator, a materially different category from
# speculative's "no specific role", and no confirmed AAC (or other source)
# posting has been found that's actually mis-filed there.
# None (missing, or a pre-posting_type-field enrichment row) is treated as
# "job" — a null value here is never itself a reason to filter a posting,
# under EITHER the global fallback or a user's own explicit allow-list.
_DEFAULT_EXCLUDED_POSTING_TYPES = frozenset({"volunteer", "funding", "speculative"})


@dataclass
class GateV2Result:
    """v2-specific gate result — not src.schemas.HardGateResult (v1's), which has
    no posting_type_pass field and shouldn't be modified for a v2-only concern.
    Field names deliberately match HardGateResult's (passed/reason/
    rejection_reason) so existing callers reading those don't need to change."""
    passed: bool
    location_pass: bool
    seniority_pass: bool
    posting_type_pass: bool
    reason: str | None = None
    rejection_reason: str | None = None

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


def check(job: dict, profile: dict) -> GateV2Result:
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
    opportunity_types = profile.get("opportunity_types")
    posting_type_pass = _check_posting_type(enrichment.get("posting_type"), opportunity_types)
    passed = location_pass and seniority_pass and posting_type_pass

    if not passed:
        parts = []
        failed_dims = []
        if not location_pass:
            parts.append(f"location: {job.get('location_field')!r} does not match any accept condition")
            failed_dims.append("location")
        if not seniority_pass:
            parts.append(f"seniority: {enrichment.get('seniority')!r} signals below threshold")
            failed_dims.append("seniority")
        if not posting_type_pass:
            if opportunity_types is not None:
                parts.append(f"posting_type: {enrichment.get('posting_type')!r} not in this user's opportunity_types {opportunity_types!r}")
            else:
                parts.append(f"posting_type: {enrichment.get('posting_type')!r} is excluded by default")
            failed_dims.append("posting_type")
        reason = "; ".join(parts)
        # Same "hard_constraints" bucket convention as v1's gate.py for a
        # multi-dimension failure — now covering any 2-or-3-way combination
        # across location/seniority/posting_type, not just the original two.
        rejection_reason = failed_dims[0] if len(failed_dims) == 1 else "hard_constraints"
    else:
        reason = None
        rejection_reason = None

    return GateV2Result(
        passed=passed,
        location_pass=location_pass,
        seniority_pass=seniority_pass,
        posting_type_pass=posting_type_pass,
        reason=reason,
        rejection_reason=rejection_reason,
    )


def _check_posting_type(posting_type: str | None, opportunity_types: list[str] | None) -> bool:
    """opportunity_types is profile["opportunity_types"] (profiles.
    opportunity_types, text[], null = unanswered) — None -> the global
    fallback rule unchanged (_DEFAULT_EXCLUDED_POSTING_TYPES); a real list
    -> pass only if this job's posting_type is in it. Either way, a null/
    missing job posting_type is treated as 'job' first (same substitution
    both paths need, so it's done once here rather than twice)."""
    effective_type = posting_type or "job"
    if opportunity_types is None:
        return effective_type not in _DEFAULT_EXCLUDED_POSTING_TYPES
    return effective_type in opportunity_types


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
