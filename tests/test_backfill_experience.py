"""Tests for scripts/backfill_experience.py's pure logic: markdown-fence
stripping, schema validation, and current-title derivation. run_dry_run()/
run_commit() are Supabase + Anthropic I/O and print()-only — not unit-tested
here, same convention as scripts/validate_gates.py's test file."""
from __future__ import annotations

import pytest

from scripts.backfill_experience import (
    _derive_current_title,
    _strip_markdown_fences,
    _validate_extraction,
)


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------

def test_strip_markdown_fences_plain_json_untouched():
    assert _strip_markdown_fences('{"a": 1}') == '{"a": 1}'


def test_strip_markdown_fences_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert _strip_markdown_fences(text) == '{"a": 1}'


def test_strip_markdown_fences_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert _strip_markdown_fences(text) == '{"a": 1}'


def test_strip_markdown_fences_strips_surrounding_whitespace():
    assert _strip_markdown_fences('  \n{"a": 1}\n  ') == '{"a": 1}'


# ---------------------------------------------------------------------------
# _validate_extraction
# ---------------------------------------------------------------------------

def _valid_payload(**overrides) -> dict:
    defaults = dict(
        experience=[{
            "title": "Engineer", "organization": "Acme", "start_date": "2020-01",
            "end_date": None, "description": None,
        }],
        education=[{
            "institution": "State University", "degree": "BSc", "field": "CS",
            "start_year": "2016", "end_year": "2020",
        }],
        skills=["python"],
    )
    defaults.update(overrides)
    return defaults


def test_validate_extraction_accepts_well_formed_payload():
    result = _validate_extraction(_valid_payload())
    assert result["experience"][0]["title"] == "Engineer"
    assert result["education"][0]["institution"] == "State University"
    assert result["skills"] == ["python"]


def test_validate_extraction_rejects_non_dict_top_level():
    with pytest.raises(ValueError, match="top-level JSON must be an object"):
        _validate_extraction(["not", "a", "dict"])


def test_validate_extraction_rejects_non_list_experience():
    with pytest.raises(ValueError, match="'experience' must be a list"):
        _validate_extraction(_valid_payload(experience={"not": "a list"}))


def test_validate_extraction_rejects_non_list_education():
    with pytest.raises(ValueError, match="'education' must be a list"):
        _validate_extraction(_valid_payload(education="nope"))


def test_validate_extraction_rejects_non_list_skills():
    with pytest.raises(ValueError, match="'skills' must be a list"):
        _validate_extraction(_valid_payload(skills="python"))


def test_validate_extraction_rejects_experience_entry_missing_key():
    payload = _valid_payload(experience=[{"title": "Engineer", "organization": "Acme"}])
    with pytest.raises(ValueError, match=r"experience\[0\] missing required key"):
        _validate_extraction(payload)


def test_validate_extraction_rejects_education_entry_missing_key():
    payload = _valid_payload(education=[{"institution": "State University"}])
    with pytest.raises(ValueError, match=r"education\[0\] missing required key"):
        _validate_extraction(payload)


def test_validate_extraction_rejects_non_dict_experience_entry():
    with pytest.raises(ValueError, match=r"experience\[0\] is not an object"):
        _validate_extraction(_valid_payload(experience=["not a dict"]))


def test_validate_extraction_accepts_empty_lists():
    result = _validate_extraction(_valid_payload(experience=[], education=[], skills=[]))
    assert result == {"experience": [], "education": [], "skills": []}


# ---------------------------------------------------------------------------
# _derive_current_title
# ---------------------------------------------------------------------------

def _entry(**overrides) -> dict:
    defaults = dict(title="Engineer", organization="Acme", start_date="2020-01", end_date=None, description=None)
    defaults.update(overrides)
    return defaults


def test_derive_current_title_no_current_role_returns_none():
    experience = [_entry(end_date="2019-01")]
    assert _derive_current_title(experience) is None


def test_derive_current_title_empty_experience_returns_none():
    assert _derive_current_title([]) is None


def test_derive_current_title_single_current_role():
    experience = [_entry(title="Data Lead", organization="Acme", end_date=None)]
    assert _derive_current_title(experience) == "Data Lead at Acme"


def test_derive_current_title_picks_latest_start_date_among_current_roles():
    experience = [
        _entry(title="Junior Engineer", organization="Old Co", start_date="2015-01", end_date=None),
        _entry(title="Senior Engineer", organization="New Co", start_date="2022-06", end_date=None),
    ]
    assert _derive_current_title(experience) == "Senior Engineer at New Co"


def test_derive_current_title_ignores_past_roles_when_picking_current():
    experience = [
        _entry(title="Past Role", organization="Old Co", start_date="2099-01", end_date="2099-12"),
        _entry(title="Current Role", organization="New Co", start_date="2020-01", end_date=None),
    ]
    assert _derive_current_title(experience) == "Current Role at New Co"


def test_derive_current_title_missing_title_or_organization_returns_none():
    experience = [_entry(title=None, organization="Acme", end_date=None)]
    assert _derive_current_title(experience) is None


def test_derive_current_title_missing_start_date_still_considered():
    experience = [_entry(title="Freelancer", organization="Self", start_date=None, end_date=None)]
    assert _derive_current_title(experience) == "Freelancer at Self"
