"""Tests for scripts/validate_gates.py's pure validation logic. main() itself is
Supabase I/O and print()-only — not unit-tested here; _validate_field is the whole
audit logic and is pure."""
from __future__ import annotations

from scripts.validate_gates import _validate_field
from src.schemas import LocationRule, SeniorityRule


def test_validate_field_none_is_valid_permissive():
    ok, detail = _validate_field(None, LocationRule)
    assert ok is True
    assert "permissive" in detail


def test_validate_field_empty_string_is_valid_permissive():
    ok, detail = _validate_field("   ", LocationRule)
    assert ok is True
    assert "permissive" in detail


def test_validate_field_valid_json_string_passes():
    ok, detail = _validate_field(
        '{"accept_fully_remote": true, "accept_hybrid_in": [], "accept_onsite_in": []}', LocationRule,
    )
    assert ok is True
    assert detail == "ok"


def test_validate_field_dict_passthrough_passes():
    ok, detail = _validate_field({"accept_levels": ["senior"]}, SeniorityRule)
    assert ok is True


def test_validate_field_invalid_json_fails():
    ok, detail = _validate_field("{not valid json", LocationRule)
    assert ok is False
    assert "invalid JSON" in detail


def test_validate_field_unknown_key_fails():
    ok, detail = _validate_field('{"accept_fully_remote": true, "extra_field": 1}', LocationRule)
    assert ok is False


def test_validate_field_bad_seniority_level_fails():
    ok, detail = _validate_field('{"accept_levels": ["expert"]}', SeniorityRule)
    assert ok is False


def test_validate_field_unexpected_type_fails():
    ok, detail = _validate_field(42, LocationRule)
    assert ok is False
    assert "unexpected type" in detail
