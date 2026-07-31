"""Tests for src/models/job_enrichment.py: the v2 jobs.enrichment schema. Strict
validation — every model rejects unknown fields."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.job_enrichment import JobEnrichment, SalaryRange, WorkArrangement


def _work_arrangement(**overrides) -> dict:
    defaults = dict(mode="remote", remote_scope="global")
    defaults.update(overrides)
    return defaults


def _enrichment(**overrides) -> dict:
    defaults = dict(summary="A data leadership role.", work_arrangement=_work_arrangement())
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# JobEnrichment — top level
# ---------------------------------------------------------------------------

def test_job_enrichment_accepts_minimal_valid_shape():
    e = JobEnrichment(**_enrichment())
    assert e.summary == "A data leadership role."
    assert e.seniority == []
    assert e.cause_area == []
    assert e.skill_tags == []
    assert e.salary is None


def test_job_enrichment_accepts_full_shape():
    e = JobEnrichment(**_enrichment(
        seniority=["senior", "director"],
        cause_area=["global health", "animal welfare"],
        skill_tags=["python", "sql"],
        salary={"min": 80000, "max": 110000, "currency": "USD"},
        work_arrangement=_work_arrangement(
            mode="hybrid", remote_scope="region_restricted",
            eligible_regions=["SE", "EU"], timezone_constraint="CET +/- 2h",
            restriction_type="timezone",
        ),
    ))
    assert e.seniority == ["senior", "director"]
    assert e.salary.min == 80000
    assert e.work_arrangement.eligible_regions == ["SE", "EU"]


def test_job_enrichment_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        JobEnrichment(**_enrichment(unexpected_field="x"))


def test_job_enrichment_requires_summary():
    with pytest.raises(ValidationError):
        JobEnrichment(work_arrangement=_work_arrangement())


def test_job_enrichment_requires_work_arrangement():
    with pytest.raises(ValidationError):
        JobEnrichment(summary="x")


def test_job_enrichment_rejects_unknown_seniority_bucket():
    with pytest.raises(ValidationError):
        JobEnrichment(**_enrichment(seniority=["expert"]))


def test_job_enrichment_accepts_all_five_seniority_buckets():
    e = JobEnrichment(**_enrichment(seniority=["intern", "junior", "mid", "senior", "director"]))
    assert len(e.seniority) == 5


# ---------------------------------------------------------------------------
# SalaryRange
# ---------------------------------------------------------------------------

def test_salary_range_all_fields_optional():
    s = SalaryRange()
    assert s.min is None
    assert s.max is None
    assert s.currency is None


def test_salary_range_partial_fields():
    s = SalaryRange(min=50000)
    assert s.min == 50000
    assert s.max is None


def test_salary_range_rejects_unknown_field():
    with pytest.raises(ValidationError):
        SalaryRange(min=1, bonus=2)


# ---------------------------------------------------------------------------
# WorkArrangement
# ---------------------------------------------------------------------------

def test_work_arrangement_defaults():
    w = WorkArrangement()
    assert w.mode == "unspecified"
    assert w.remote_scope == "unspecified"
    assert w.eligible_regions == []
    assert w.timezone_constraint is None
    assert w.restriction_type is None


def test_work_arrangement_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        WorkArrangement(mode="flexible")


def test_work_arrangement_rejects_unknown_remote_scope():
    with pytest.raises(ValidationError):
        WorkArrangement(remote_scope="worldwide")


def test_work_arrangement_rejects_unknown_restriction_type():
    with pytest.raises(ValidationError):
        WorkArrangement(restriction_type="visa")


def test_work_arrangement_accepts_both_restriction_types():
    assert WorkArrangement(restriction_type="authorization").restriction_type == "authorization"
    assert WorkArrangement(restriction_type="timezone").restriction_type == "timezone"


def test_work_arrangement_rejects_unknown_field():
    with pytest.raises(ValidationError):
        WorkArrangement(mode="remote", extra_field="x")
