"""Unit tests for 80k Hours source parse_hits() and helpers — no network calls."""
from __future__ import annotations

from datetime import date

import pytest

from src.sources.eightyk import _extract_comp, _extract_deadline, _strip_html, parse_hits


# ---------------------------------------------------------------------------
# Fixture helper — real field names from the Algolia index
# ---------------------------------------------------------------------------

def hit(**overrides) -> dict:
    base = {
        "title": "Head of Data",
        "company_name": "GiveDirectly",
        "url_external": "https://givedirectly.org/jobs/head-of-data",
        "tags_location_80k": ["Remote"],
        "tags_exp_required": ["Senior"],
        "tags_area": ["EA-aligned organisations"],
        "tags_skill": ["Data analysis"],
        "salary": "$120,000",
        "closes_at": 1893456000,  # 2030-01-01 — safely in the future
        "description_short": "<p>Lead our data team.</p>",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# parse_hits — basic field mapping
# ---------------------------------------------------------------------------

def test_title_mapped():
    [p] = parse_hits([hit()])
    assert p.title == "Head of Data"


def test_org_from_company_name():
    [p] = parse_hits([hit()])
    assert p.org == "GiveDirectly"


def test_source_is_80k():
    [p] = parse_hits([hit()])
    assert p.source == "80k"


def test_url_from_url_external():
    [p] = parse_hits([hit()])
    assert p.url == "https://givedirectly.org/jobs/head-of-data"


def test_location_from_tags_location_80k():
    [p] = parse_hits([hit(tags_location_80k=["Remote"])])
    assert p.location == "Remote"


def test_location_multiple_values_joined():
    [p] = parse_hits([hit(tags_location_80k=["London", "Remote"])])
    assert p.location == "London, Remote"


def test_location_none_when_absent():
    [p] = parse_hits([hit(tags_location_80k=[])])
    assert p.location is None


def test_seniority_from_tags_exp_required():
    [p] = parse_hits([hit(tags_exp_required=["Senior", "Mid-level"])])
    assert p.seniority == "Senior, Mid-level"


def test_seniority_none_when_absent():
    [p] = parse_hits([hit(tags_exp_required=[])])
    assert p.seniority is None


def test_cause_area_from_tags_area():
    [p] = parse_hits([hit(tags_area=["Global health", "Animal welfare"])])
    assert p.cause_area == "Global health, Animal welfare"


def test_cause_area_none_when_absent():
    [p] = parse_hits([hit(tags_area=[])])
    assert p.cause_area is None


def test_comp_from_salary():
    [p] = parse_hits([hit(salary="$120,000")])
    assert p.comp == "$120,000"


def test_comp_none_when_salary_absent():
    h = hit()
    del h["salary"]
    [p] = parse_hits([h])
    assert p.comp is None


def test_comp_none_when_salary_empty():
    [p] = parse_hits([hit(salary="")])
    assert p.comp is None


def test_deadline_from_closes_at():
    [p] = parse_hits([hit(closes_at=1893456000)])
    assert p.deadline == date(2030, 1, 1)


def test_deadline_none_when_closes_at_zero():
    [p] = parse_hits([hit(closes_at=0)])
    assert p.deadline is None


def test_deadline_none_when_closes_at_absent():
    h = hit()
    del h["closes_at"]
    [p] = parse_hits([h])
    assert p.deadline is None


# ---------------------------------------------------------------------------
# parse_hits — raw_text content
# ---------------------------------------------------------------------------

def test_raw_text_contains_title():
    [p] = parse_hits([hit()])
    assert "Head of Data" in p.raw_text


def test_raw_text_contains_org():
    [p] = parse_hits([hit()])
    assert "GiveDirectly" in p.raw_text


def test_raw_text_contains_tags_area():
    [p] = parse_hits([hit(tags_area=["Global health"])])
    assert "Global health" in p.raw_text


def test_raw_text_contains_tags_skill():
    [p] = parse_hits([hit(tags_skill=["Python"])])
    assert "Python" in p.raw_text


def test_raw_text_contains_stripped_description():
    [p] = parse_hits([hit(description_short="<p>Lead <b>our</b> data team.</p>")])
    assert "Lead" in p.raw_text
    assert "<p>" not in p.raw_text


# ---------------------------------------------------------------------------
# parse_hits — skipping rules
# ---------------------------------------------------------------------------

def test_skips_hit_with_empty_title():
    assert parse_hits([hit(title="")]) == []


def test_skips_hit_with_missing_title():
    h = hit()
    del h["title"]
    assert parse_hits([h]) == []


def test_skips_hit_with_empty_url_external():
    assert parse_hits([hit(url_external="")]) == []


def test_skips_hit_with_missing_url_external():
    h = hit()
    del h["url_external"]
    assert parse_hits([h]) == []


def test_multiple_hits_all_returned():
    h2 = hit(title="M&E Lead", url_external="https://example.org/mne")
    assert len(parse_hits([hit(), h2])) == 2


# ---------------------------------------------------------------------------
# _extract_comp
# ---------------------------------------------------------------------------

def test_extract_comp_returns_salary_string():
    assert _extract_comp({"salary": "$90,000–$110,000"}) == "$90,000–$110,000"


def test_extract_comp_none_when_absent():
    assert _extract_comp({}) is None


def test_extract_comp_none_when_empty():
    assert _extract_comp({"salary": "  "}) is None


def test_extract_comp_none_when_not_string():
    assert _extract_comp({"salary": 90000}) is None


# ---------------------------------------------------------------------------
# _extract_deadline
# ---------------------------------------------------------------------------

def test_extract_deadline_from_valid_timestamp():
    assert _extract_deadline({"closes_at": 1893456000}) == date(2030, 1, 1)


def test_extract_deadline_none_when_zero():
    assert _extract_deadline({"closes_at": 0}) is None


def test_extract_deadline_none_when_negative():
    assert _extract_deadline({"closes_at": -1}) is None


def test_extract_deadline_none_when_absent():
    assert _extract_deadline({}) is None


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello  world"


def test_strip_html_empty_string():
    assert _strip_html("") == ""


def test_strip_html_plain_text_unchanged():
    assert _strip_html("No tags here") == "No tags here"
