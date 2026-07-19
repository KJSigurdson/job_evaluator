"""Tests for enrich.py's response-parsing helpers — pure, no Anthropic API calls.

Covers the emphasize_in_cv/deemphasize list[str] guarantee: these must survive
parsing as real lists, including the defensive case where a model returns a
JSON-array-in-a-string instead of a native array.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.enrich import _coerce_list_field, _parse_enrichment_output


def _tool_input(**overrides) -> dict:
    defaults = dict(
        org_summary="A charity evaluator.",
        why_fits="Strong mission alignment.",
        why_not_fits="No fundraising experience.",
        emphasize_in_cv=["BI leadership"],
        deemphasize=["IT portfolio work"],
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# _coerce_list_field
# ---------------------------------------------------------------------------

def test_coerce_list_field_passes_through_real_list():
    assert _coerce_list_field(["a", "b"]) == ["a", "b"]


def test_coerce_list_field_parses_json_array_string():
    assert _coerce_list_field('["a", "b"]') == ["a", "b"]


def test_coerce_list_field_empty_json_array_string():
    assert _coerce_list_field("[]") == []


def test_coerce_list_field_leaves_malformed_string_unchanged():
    assert _coerce_list_field("not json at all") == "not json at all"


def test_coerce_list_field_leaves_non_array_json_string_unchanged():
    # A valid JSON string that parses to a non-list (e.g. an object or scalar) is not
    # our case to fix — leave it for normal list[str] validation to reject clearly.
    assert _coerce_list_field('{"a": 1}') == '{"a": 1}'
    assert _coerce_list_field("42") == "42"


def test_coerce_list_field_passes_through_non_string_non_list():
    assert _coerce_list_field(None) is None


# ---------------------------------------------------------------------------
# _parse_enrichment_output — end-to-end through Tier2EnrichmentOutput validation
# ---------------------------------------------------------------------------

def test_parse_enrichment_output_native_arrays_stay_lists():
    result = _parse_enrichment_output(_tool_input())
    assert result.emphasize_in_cv == ["BI leadership"]
    assert result.deemphasize == ["IT portfolio work"]
    assert isinstance(result.emphasize_in_cv, list)
    assert isinstance(result.deemphasize, list)


def test_parse_enrichment_output_coerces_json_array_string_to_list():
    tool_input = _tool_input(
        emphasize_in_cv='["BI leadership", "CRM ownership"]',
        deemphasize='["IT portfolio work"]',
    )
    result = _parse_enrichment_output(tool_input)
    assert result.emphasize_in_cv == ["BI leadership", "CRM ownership"]
    assert isinstance(result.emphasize_in_cv, list)
    assert result.deemphasize == ["IT portfolio work"]
    assert isinstance(result.deemphasize, list)


def test_parse_enrichment_output_empty_json_array_string_becomes_empty_list():
    tool_input = _tool_input(emphasize_in_cv="[]")
    result = _parse_enrichment_output(tool_input)
    assert result.emphasize_in_cv == []


def test_parse_enrichment_output_malformed_string_still_raises():
    tool_input = _tool_input(emphasize_in_cv="not a json array")
    with pytest.raises(ValidationError):
        _parse_enrichment_output(tool_input)


def test_parse_enrichment_output_other_fields_untouched():
    result = _parse_enrichment_output(_tool_input(org_summary="X"))
    assert result.org_summary == "X"
