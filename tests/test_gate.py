"""Tests for hard-gate logic (gate.py)."""
from __future__ import annotations

import pytest

from src.gate import check
from tests.conftest import make_posting


# ---------------------------------------------------------------------------
# Location — pass cases
# ---------------------------------------------------------------------------

def test_unstated_location_passes(profile):
    result = check(make_posting(location=None), profile)
    assert result.location_pass

def test_remote_passes(profile):
    result = check(make_posting(location="Remote"), profile)
    assert result.location_pass

def test_fully_remote_passes(profile):
    result = check(make_posting(location="Fully Remote (global)"), profile)
    assert result.location_pass

def test_anywhere_passes(profile):
    result = check(make_posting(location="Anywhere"), profile)
    assert result.location_pass

def test_sweden_hybrid_passes(profile):
    result = check(make_posting(location="Stockholm, Sweden (hybrid)"), profile)
    assert result.location_pass

def test_sundsvall_onsite_passes(profile):
    result = check(make_posting(location="Sundsvall, on-site"), profile)
    assert result.location_pass

def test_sweden_mention_passes(profile):
    # Generic "Sweden" mention → pass (bias toward false-positives)
    result = check(make_posting(location="Sweden"), profile)
    assert result.location_pass


# ---------------------------------------------------------------------------
# Location — fail cases
# ---------------------------------------------------------------------------

def test_london_onsite_fails(profile):
    result = check(make_posting(location="London, UK (on-site)"), profile)
    assert not result.location_pass
    assert not result.passed

def test_us_only_fails(profile):
    result = check(make_posting(location="New York, USA"), profile)
    assert not result.location_pass

def test_nairobi_fails(profile):
    result = check(make_posting(location="Nairobi, Kenya (on-site required)"), profile)
    assert not result.location_pass


# ---------------------------------------------------------------------------
# Seniority — pass cases
# ---------------------------------------------------------------------------

def test_unstated_seniority_passes(profile):
    result = check(make_posting(seniority=None), profile)
    assert result.seniority_pass

def test_senior_passes(profile):
    result = check(make_posting(seniority="Senior Data Analyst, 7+ years"), profile)
    assert result.seniority_pass

def test_director_passes(profile):
    result = check(make_posting(seniority="Director level, 10+ years experience"), profile)
    assert result.seniority_pass


# ---------------------------------------------------------------------------
# Seniority — fail cases
# ---------------------------------------------------------------------------

def test_junior_fails(profile):
    result = check(make_posting(seniority="Junior Data Analyst"), profile)
    assert not result.seniority_pass
    assert not result.passed

def test_entry_level_fails(profile):
    result = check(make_posting(seniority="Entry-level position"), profile)
    assert not result.seniority_pass

def test_graduate_fails(profile):
    result = check(make_posting(seniority="Graduate Analyst Programme"), profile)
    assert not result.seniority_pass

def test_intern_fails(profile):
    result = check(make_posting(seniority="Internship — 6 months"), profile)
    assert not result.seniority_pass


# ---------------------------------------------------------------------------
# Combined gate result
# ---------------------------------------------------------------------------

def test_reason_populated_on_failure(profile):
    result = check(make_posting(location="London, UK", seniority="Junior"), profile)
    assert not result.passed
    assert result.reason is not None
    assert len(result.reason) > 0

def test_reason_none_on_pass(profile):
    result = check(make_posting(location="Remote", seniority="Senior, 8+ years"), profile)
    assert result.passed
    assert result.reason is None
