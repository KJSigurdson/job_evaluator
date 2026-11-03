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

import difflib
import hashlib
import re
from urllib.parse import urlparse, urlunparse

from src.dedup import canonical_url as _v1_canonical_url

# Above this normalized-text similarity ratio, a content_hash change is treated
# as cosmetic (e.g. a single appended sentence) rather than a real posting
# update — see src/ingest.py's changed-content branch. Chosen from real
# production diffs: genuine content edits (e.g. an appended contact line) score
# well below this; only near-byte-identical text scores at or above it.
RE_ENRICH_SIMILARITY_THRESHOLD = 0.97


def normalize_job_url(url: str) -> str:
    """Canonical form of *url* for the v2 jobs table: dedup.canonical_url()'s
    UTM/tracking-param stripping and scheme/host lowercasing, plus a trailing
    slash stripped from the path (so ".../123/" and ".../123" canonicalize
    identically). The bare root path "/" is left alone.

    Fragment handling is NOT simply "drop it" (dedup.canonical_url()'s
    behavior, unconditional, left unmodified there since v1 still relies on
    that): a hash-ROUTED fragment (one starting with "/", e.g.
    "#/job/details/261") encodes the actual resource identity for a
    client-side-routed SPA apply page, unlike an ordinary same-page anchor
    (e.g. "#contact", "#apply-form") which really is just page-scroll
    position and is correctly dropped. Confirmed for real: Animal Advocacy
    Careers' listings link to Compassion in World Farming's careers site
    (ciwf.livevacancies.co.uk), which posts multiple DISTINCT jobs whose
    URLs differ ONLY by this kind of fragment
    (".../#/job/details/261" vs "...248" vs "...258" vs "...249") —
    dropping it unconditionally collapsed all of them to one canonical_url.
    """
    original_fragment = urlparse(url).fragment
    canonical = _v1_canonical_url(url)
    parsed = urlparse(canonical)
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    fragment = original_fragment if original_fragment.startswith("/") else ""
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, fragment))


def _normalize_text(text: str) -> str:
    """Lowercased, leading/trailing whitespace stripped, internal whitespace runs
    collapsed to a single space — shared by content_hash() and text_similarity()
    so two strings that hash identically also score a perfect 1.0 similarity."""
    return re.sub(r"\s+", " ", text.strip().lower())


def content_hash(description: str) -> str:
    """sha256 hex digest of *description*, normalized so semantically-identical
    text differing only in whitespace or case hashes identically."""
    return hashlib.sha256(_normalize_text(description).encode("utf-8")).hexdigest()


def text_similarity(a: str, b: str) -> float:
    """difflib.SequenceMatcher ratio (0..1) between *a* and *b*, both normalized
    the same way content_hash() normalizes — used to distinguish a cosmetic
    content_hash change from a real one (see RE_ENRICH_SIMILARITY_THRESHOLD)."""
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()
