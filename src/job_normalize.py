"""URL normalization and content hashing for the v2 `jobs` table (True North v2,
shared job enrichment architecture — see
supabase/migrations/20260731075824_true_north_v2_schema.sql).

normalize_job_url() reuses src/dedup.py's canonical_url() for the UTM/tracking-param
stripping and fragment-dropping it already does for the v1 pipeline, adding a
trailing-slash strip on top. dedup.py itself is NOT modified — this is purely
additive, so v1's existing dedup behavior (and every test/assumption built on it
this session) is unaffected.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse, urlunparse

from src.dedup import canonical_url as _v1_canonical_url


def normalize_job_url(url: str) -> str:
    """Canonical form of *url* for the v2 jobs table: dedup.canonical_url()'s
    UTM/tracking-param stripping, fragment-dropping, and scheme/host lowercasing,
    plus a trailing slash stripped from the path (so ".../123/" and ".../123"
    canonicalize identically). The bare root path "/" is left alone."""
    canonical = _v1_canonical_url(url)
    parsed = urlparse(canonical)
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))


def content_hash(description: str) -> str:
    """sha256 hex digest of *description*, normalized so semantically-identical
    text differing only in whitespace or case hashes identically: lowercased,
    leading/trailing whitespace stripped, internal whitespace runs collapsed to a
    single space."""
    normalized = re.sub(r"\s+", " ", description.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
