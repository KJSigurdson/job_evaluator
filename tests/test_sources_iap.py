"""Unit tests for IAP source parse_rows() and helpers — no network calls."""
from __future__ import annotations

from datetime import date

import pytest

from src.sources.iap import Cell, RawRow, _parse_date, parse_rows


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def cell(text: str = "", url: str | None = None) -> Cell:
    return {"text": text, "url": url}


def row(*cells: Cell) -> RawRow:
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


def sheet(*data_rows: RawRow) -> list[RawRow]:
    return [_HEADER, *data_rows]


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def test_header_at_row_zero():
    assert len(parse_rows(sheet(_GOOD_ROW))) == 1


def test_header_after_preamble_rows():
    preamble = [row(cell("IAP Referral Overview")), row()]
    postings = parse_rows([*preamble, _HEADER, _GOOD_ROW])
    assert len(postings) == 1


def test_raises_when_no_header_found():
    with pytest.raises(ValueError, match="Header row"):
        parse_rows([_GOOD_ROW])


def test_rows_before_header_are_ignored():
    before = row(cell("GiveDirectly"), cell("Ignored"), cell("https://ignore.example.com"))
    postings = parse_rows([before, _HEADER, _GOOD_ROW])
    assert len(postings) == 1
    assert postings[0].org == "GiveDirectly"
    assert postings[0].url == "https://givedirectly.org/jobs/director-data"


# ---------------------------------------------------------------------------
# Basic field mapping
# ---------------------------------------------------------------------------

def test_org_mapped():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.org == "GiveDirectly"


def test_title_mapped():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.title == "Director of Data"


def test_seniority_equals_title():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.seniority == p.title


def test_source_is_iap():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.source == "iap"


def test_location_mapped():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.location == "Remote"


def test_comp_mapped():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.comp == "$120,000"


def test_deadline_mapped():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert p.deadline == date(2026, 12, 31)


def test_raw_text_contains_org():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert "GiveDirectly" in p.raw_text


def test_raw_text_contains_title():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert "Director of Data" in p.raw_text


def test_raw_text_contains_cause_area():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert "Global health" in p.raw_text


def test_raw_text_contains_location():
    [p] = parse_rows(sheet(_GOOD_ROW))
    assert "Remote" in p.raw_text


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

def test_hyperlink_preferred_over_cell_text():
    r = row(
        cell("Org"), cell("Role"),
        cell("Click here", url="https://real-url.org/job"),
        cell("Remote"), cell(), cell(), cell(), cell(), cell(),
    )
    [p] = parse_rows(sheet(r))
    assert p.url == "https://real-url.org/job"


def test_cell_text_used_when_already_a_url():
    r = row(
        cell("Org"), cell("Role"),
        cell("https://fallback.org/job"),  # no hyperlink, but text is a URL
        cell("Remote"), cell(), cell(), cell(), cell(), cell(),
    )
    [p] = parse_rows(sheet(r))
    assert p.url == "https://fallback.org/job"


def test_display_text_without_hyperlink_skips_row():
    r = row(
        cell("Org"), cell("Role"),
        cell("See attached PDF"),  # not a URL, no hyperlink
        cell("Remote"), cell(), cell(), cell(), cell(), cell(),
    )
    assert parse_rows(sheet(r)) == []


# ---------------------------------------------------------------------------
# Stopping and skipping rules
# ---------------------------------------------------------------------------

def test_blank_row_in_middle_is_skipped():
    # The IAP sheet uses blank/separator rows inside the table; they must not stop parsing.
    empty = row()
    later = row(
        cell("ACE"), cell("Analyst"), cell("https://ace.org/job"),
        cell("Remote"), cell(), cell("Animal welfare"), cell(), cell(), cell(),
    )
    postings = parse_rows(sheet(_GOOD_ROW, empty, later))
    assert len(postings) == 2  # both rows returned despite the blank between them


def test_all_blank_cells_skipped_not_fatal():
    all_blank = row(cell(""), cell(""), cell(""), cell(""), cell(""))
    later = row(
        cell("Org2"), cell("Role2"), cell("https://example.com/job2"),
        cell(), cell(), cell(), cell(), cell(), cell(),
    )
    postings = parse_rows(sheet(_GOOD_ROW, all_blank, later))
    assert len(postings) == 2


def test_skips_row_with_no_org():
    r = row(
        cell(""), cell("Role"), cell("https://example.com/job"),
        cell(), cell(), cell(), cell(), cell(), cell(),
    )
    assert parse_rows(sheet(r)) == []


def test_skips_row_with_no_title():
    r = row(
        cell("Org"), cell(""), cell("https://example.com/job"),
        cell(), cell(), cell(), cell(), cell(), cell(),
    )
    assert parse_rows(sheet(r)) == []


def test_skips_various_roles_rows():
    for title in ("Various roles", "VARIOUS ROLES", "Various Roles (see careers page)"):
        r = row(
            cell("Org"), cell(title), cell("https://example.com/careers"),
            cell(), cell(), cell(), cell(), cell(), cell(),
        )
        assert parse_rows(sheet(r)) == [], f"Expected {title!r} to be filtered out"


def test_skips_row_with_no_url():
    r = row(
        cell("Org"), cell("Role"), cell(""),
        cell(), cell(), cell(), cell(), cell(), cell(),
    )
    assert parse_rows(sheet(r)) == []


def test_multiple_valid_rows_all_returned():
    r2 = row(
        cell("ACE"), cell("Analyst"),
        cell("https://ace.org/job"),
        cell("Remote"), cell(), cell("Animal welfare"), cell(), cell(), cell(),
    )
    assert len(parse_rows(sheet(_GOOD_ROW, r2))) == 2


def test_short_row_does_not_crash():
    r = row(cell("Org"), cell("Role"), cell("https://example.com/job"))
    [p] = parse_rows(sheet(r))
    assert p.location is None
    assert p.comp is None
    assert p.deadline is None


# ---------------------------------------------------------------------------
# Optional / blank fields
# ---------------------------------------------------------------------------

def test_location_none_when_blank():
    r = row(
        cell("Org"), cell("Role"), cell("https://example.com/job"),
        cell(""),  # blank location
        cell(), cell(), cell(), cell(), cell(),
    )
    [p] = parse_rows(sheet(r))
    assert p.location is None


def test_comp_none_when_blank():
    r = row(
        cell("Org"), cell("Role"), cell("https://example.com/job"),
        cell("Remote"), cell(""),  # blank comp
        cell(), cell(), cell(), cell(),
    )
    [p] = parse_rows(sheet(r))
    assert p.comp is None


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

def test_date_iso_format():
    assert _parse_date("2026-12-31") == date(2026, 12, 31)


def test_date_long_format():
    assert _parse_date("December 31, 2026") == date(2026, 12, 31)


def test_date_us_slash_format():
    assert _parse_date("12/31/2026") == date(2026, 12, 31)


def test_date_none_on_blank():
    assert _parse_date("") is None


def test_date_none_on_rolling():
    assert _parse_date("Rolling") is None


def test_date_none_on_asap():
    assert _parse_date("ASAP") is None


def test_date_none_on_open_ended():
    assert _parse_date("Open until filled") is None
