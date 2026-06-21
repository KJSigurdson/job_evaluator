"""Tests for URL canonicalization and dedup logic (dedup.py)."""
from __future__ import annotations

import pytest

from src.dedup import canonical_url, is_duplicate, org_title_key


# ---------------------------------------------------------------------------
# canonical_url — tracking param stripping
# ---------------------------------------------------------------------------

def test_strips_utm_source():
    url = "https://jobs.example.com/role?utm_source=newsletter&id=42"
    assert canonical_url(url) == "https://jobs.example.com/role?id=42"

def test_strips_all_utm_params():
    url = "https://jobs.example.com/role?utm_source=a&utm_medium=b&utm_campaign=c&utm_term=d&utm_content=e&id=42"
    assert canonical_url(url) == "https://jobs.example.com/role?id=42"

def test_strips_ref_param():
    url = "https://jobs.example.com/role?ref=twitter&id=42"
    assert canonical_url(url) == "https://jobs.example.com/role?id=42"

def test_strips_fbclid():
    url = "https://jobs.example.com/role?fbclid=abc123&id=42"
    assert canonical_url(url) == "https://jobs.example.com/role?id=42"

def test_drops_fragment():
    url = "https://jobs.example.com/role/42#apply-section"
    assert canonical_url(url) == "https://jobs.example.com/role/42"

def test_preserves_non_tracking_params():
    url = "https://jobs.example.com/role?id=42&page=2"
    result = canonical_url(url)
    assert "id=42" in result
    assert "page=2" in result

def test_lowercases_scheme_and_host():
    url = "HTTPS://Jobs.Example.COM/role?id=42"
    result = canonical_url(url)
    assert result.startswith("https://jobs.example.com/")

def test_idempotent():
    url = "https://jobs.example.com/role?utm_source=x&id=42#section"
    assert canonical_url(url) == canonical_url(canonical_url(url))

def test_deterministic_param_order():
    url_a = "https://jobs.example.com/role?b=2&a=1"
    url_b = "https://jobs.example.com/role?a=1&b=2"
    assert canonical_url(url_a) == canonical_url(url_b)

def test_clean_url_unchanged():
    url = "https://jobs.example.com/role/42"
    assert canonical_url(url) == url


# ---------------------------------------------------------------------------
# org_title_key — normalisation
# ---------------------------------------------------------------------------

def test_org_title_key_lowercases():
    assert org_title_key("GiveDirectly", "Head of Data") == org_title_key("givedirectly", "head of data")

def test_org_title_key_strips_whitespace():
    assert org_title_key("  GiveDirectly  ", "  Head of Data  ") == org_title_key("GiveDirectly", "Head of Data")

def test_org_title_key_distinct_orgs():
    assert org_title_key("OrgA", "Same Title") != org_title_key("OrgB", "Same Title")

def test_org_title_key_distinct_titles():
    assert org_title_key("Same Org", "Title A") != org_title_key("Same Org", "Title B")


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

def test_is_duplicate_exact_match():
    existing = {"https://jobs.example.com/role/42"}
    assert is_duplicate("https://jobs.example.com/role/42", existing)

def test_is_duplicate_after_stripping_utm():
    existing = {"https://jobs.example.com/role/42"}
    assert is_duplicate("https://jobs.example.com/role/42?utm_source=newsletter", existing)

def test_is_not_duplicate_different_id():
    existing = {"https://jobs.example.com/role/42"}
    assert not is_duplicate("https://jobs.example.com/role/99", existing)

def test_is_not_duplicate_empty_set():
    assert not is_duplicate("https://jobs.example.com/role/42", set())
