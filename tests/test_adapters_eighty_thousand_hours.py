"""Tests for src/adapters/eighty_thousand_hours.py's pure parse step (no network)."""
from __future__ import annotations

from src.adapters.base import JobPosting
from src.adapters.eighty_thousand_hours import SOURCE_BOARD, parse_hits


def _hit(**overrides) -> dict:
    defaults = dict(
        title="Head of Data",
        url_external="https://jobs.80000hours.org/head-of-data",
        company_name="GiveDirectly",
        tags_location_80k=["Remote, Global"],
        description_short="<p>We are looking for a <b>Head of Data</b>.</p>",
    )
    defaults.update(overrides)
    return defaults


def test_parse_hits_builds_job_posting():
    [posting] = parse_hits([_hit()])
    assert isinstance(posting, JobPosting)
    assert posting.title == "Head of Data"
    assert posting.url == "https://jobs.80000hours.org/head-of-data"
    assert posting.org == "GiveDirectly"
    assert posting.source_board == SOURCE_BOARD


def test_parse_hits_strips_html_from_description():
    [posting] = parse_hits([_hit()])
    assert "<p>" not in posting.description
    assert "<b>" not in posting.description
    assert "Head of Data" in posting.description


def test_parse_hits_joins_location_tags():
    [posting] = parse_hits([_hit(tags_location_80k=["Remote", "Sweden"])])
    assert posting.location_field == "Remote, Sweden"


def test_parse_hits_location_none_when_no_tags():
    [posting] = parse_hits([_hit(tags_location_80k=[])])
    assert posting.location_field is None


def test_parse_hits_skips_hit_with_no_title():
    assert parse_hits([_hit(title="")]) == []


def test_parse_hits_skips_hit_with_no_url():
    assert parse_hits([_hit(url_external="")]) == []


def test_parse_hits_skips_hit_with_missing_fields_entirely():
    assert parse_hits([{}]) == []


def test_parse_hits_handles_missing_optional_fields_gracefully():
    hit = {"title": "Analyst", "url_external": "https://example.org/analyst"}
    [posting] = parse_hits([hit])
    assert posting.org == ""
    assert posting.location_field is None
    assert posting.description == ""


def test_parse_hits_preserves_order():
    hits = [_hit(title=f"Role {i}", url_external=f"https://example.org/{i}") for i in range(5)]
    postings = parse_hits(hits)
    assert [p.title for p in postings] == [f"Role {i}" for i in range(5)]


def test_parse_hits_multiple_hits():
    postings = parse_hits([_hit(), _hit(title="Chief of Staff", url_external="https://example.org/cos")])
    assert len(postings) == 2
