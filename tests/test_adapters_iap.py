"""Tests for src/adapters/iap.py's pure parse step (no network)."""
from __future__ import annotations

from src.adapters import iap
from src.adapters.base import FetchResult, JobPosting
from src.adapters.iap import IAPAdapter, SOURCE_BOARD, parse_rows


def cell(text: str = "", url: str | None = None) -> dict:
    return {"text": text, "url": url}


def row(*cells: dict) -> list[dict]:
    return list(cells)


_HEADER = row(
    cell("Organization"), cell("Role"), cell("Job description"),
    cell("Location"), cell("Salary"), cell("Cause Area"),
    cell("Application deadline"), cell(""), cell("Date added"),
)

_GOOD_ROW = row(
    cell("GiveDirectly"),
    cell("Director of Data"),
    cell("Apply here", url="https://givedirectly.org/jobs/director-data"),
    cell("Remote"),
    cell("$120,000"),
    cell("Global health and development"),
    cell("2026-12-31"),
    cell(""),
    cell("2026-01-10"),
)


def sheet(*data_rows) -> list:
    return [_HEADER, *data_rows]


def test_parse_rows_builds_job_posting():
    [posting] = parse_rows(sheet(_GOOD_ROW))
    assert isinstance(posting, JobPosting)
    assert posting.title == "Director of Data"
    assert posting.org == "GiveDirectly"
    assert posting.url == "https://givedirectly.org/jobs/director-data"
    assert posting.location_field == "Remote"
    assert posting.source_board == SOURCE_BOARD


def test_parse_rows_description_joins_available_text():
    [posting] = parse_rows(sheet(_GOOD_ROW))
    assert "GiveDirectly" in posting.description
    assert "Director of Data" in posting.description
    assert "Global health and development" in posting.description


def test_parse_rows_header_after_preamble():
    preamble = [row(cell("IAP Referral Overview")), row()]
    postings = parse_rows([*preamble, _HEADER, _GOOD_ROW])
    assert len(postings) == 1


def test_parse_rows_raises_when_no_header_found():
    import pytest
    with pytest.raises(ValueError):
        parse_rows([row(cell("not a header"))])


def test_parse_rows_skips_empty_rows():
    postings = parse_rows(sheet(row(), _GOOD_ROW))
    assert len(postings) == 1


def test_parse_rows_skips_various_titles():
    various_row = row(
        cell("Org"), cell("Various roles"), cell("x", url="https://example.org/x"),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )
    assert parse_rows(sheet(various_row)) == []


def test_parse_rows_skips_missing_org_title_or_url():
    missing_url_row = row(
        cell("Org"), cell("Title"), cell(""),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )
    assert parse_rows(sheet(missing_url_row)) == []


def test_parse_rows_url_falls_back_to_cell_text_if_it_looks_like_a_url():
    text_url_row = row(
        cell("Org"), cell("Title"), cell("https://example.org/x"),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )
    [posting] = parse_rows(sheet(text_url_row))
    assert posting.url == "https://example.org/x"


def test_parse_rows_location_none_when_blank():
    blank_loc_row = row(
        cell("Org"), cell("Title"), cell("x", url="https://example.org/x"),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )
    [posting] = parse_rows(sheet(blank_loc_row))
    assert posting.location_field is None


# ---------------------------------------------------------------------------
# fetch() — wraps fetch_raw_rows() into a FetchResult
# ---------------------------------------------------------------------------

def test_fetch_wraps_rows_into_fetch_result(monkeypatch):
    monkeypatch.setattr(iap, "fetch_raw_rows", lambda: sheet(row(
        cell("Org"), cell("Title"), cell("x", url="https://example.org/x"),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )))

    result = IAPAdapter().fetch()

    assert isinstance(result, FetchResult)
    assert len(result.postings) == 1
    assert result.fetched == len(sheet(row(
        cell("Org"), cell("Title"), cell("x", url="https://example.org/x"),
        cell(""), cell(""), cell(""), cell(""), cell(""), cell(""),
    )))
    assert result.nb_hits is None
    assert result.complete is True
