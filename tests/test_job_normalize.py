"""Tests for src/job_normalize.py: v2 jobs-table URL normalization and content
hashing."""
from __future__ import annotations

from src.job_normalize import content_hash, normalize_job_url


# ---------------------------------------------------------------------------
# normalize_job_url
# ---------------------------------------------------------------------------

def test_normalize_job_url_strips_utm_params():
    a = normalize_job_url("https://example.org/job/123?utm_source=80k&utm_medium=email")
    b = normalize_job_url("https://example.org/job/123?utm_source=iap&utm_campaign=spring")
    assert a == b == "https://example.org/job/123"


def test_normalize_job_url_different_utm_params_same_canonical_form():
    """Explicit requirement: same URL with different UTM params -> same canonical form."""
    a = normalize_job_url("https://example.org/job/123?utm_source=80k")
    b = normalize_job_url("https://example.org/job/123?utm_source=probablygood&ref=newsletter")
    assert a == b


def test_normalize_job_url_strips_trailing_slash():
    assert normalize_job_url("https://example.org/job/123/") == normalize_job_url("https://example.org/job/123")


def test_normalize_job_url_strips_fragment():
    assert normalize_job_url("https://example.org/job/123#apply-now") == "https://example.org/job/123"


def test_normalize_job_url_preserves_root_slash():
    """The bare root path must not be stripped down to an empty path."""
    assert normalize_job_url("https://example.org/") == "https://example.org/"


def test_normalize_job_url_lowercases_scheme_and_host():
    assert normalize_job_url("HTTPS://Example.ORG/job/123") == "https://example.org/job/123"


def test_normalize_job_url_trailing_slash_with_query_string():
    a = normalize_job_url("https://example.org/job/123/?utm_source=x")
    b = normalize_job_url("https://example.org/job/123")
    assert a == b


def test_normalize_job_url_preserves_non_tracking_query_params():
    assert normalize_job_url("https://example.org/job/123?ref_id=5") == "https://example.org/job/123?ref_id=5"


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

def test_content_hash_same_for_whitespace_variation():
    """Explicit requirement: whitespace variations of a description -> same hash."""
    a = content_hash("Head of Data   at    GiveDirectly")
    b = content_hash("Head of Data at GiveDirectly")
    assert a == b


def test_content_hash_same_for_leading_trailing_whitespace():
    assert content_hash("  Head of Data  ") == content_hash("Head of Data")


def test_content_hash_same_for_case_variation():
    """Explicit requirement: case variations of a description -> same hash."""
    assert content_hash("HEAD OF DATA at GiveDirectly") == content_hash("head of data at givedirectly")


def test_content_hash_different_for_different_content():
    assert content_hash("Head of Data") != content_hash("Chief of Staff")


def test_content_hash_is_sha256_hex_digest():
    h = content_hash("some job description text")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_newline_and_tab_variation_same_hash():
    a = content_hash("Line one\nLine two\tindented")
    b = content_hash("Line one Line two indented")
    assert a == b
