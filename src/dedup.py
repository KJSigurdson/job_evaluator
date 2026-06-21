"""URL canonicalization and org+title dedup key. Notion is the single source of truth for seen URLs."""
from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Known tracking/referral params that must be stripped before comparing URLs.
_STRIP_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "referrer", "source", "campaign", "fbclid", "gclid", "msclkid",
    "mc_eid", "mc_cid",
})


def canonical_url(url: str) -> str:
    """Return a stable, comparable form of *url*.

    Strips tracking query params, drops the fragment, and lowercases scheme + host.
    Remaining params are sorted for determinism.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    clean = {k: v for k, v in params.items() if k not in _STRIP_PARAMS}
    clean_query = urlencode(sorted(clean.items()), doseq=True)
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        clean_query,
        "",  # drop fragment
    ))


def org_title_key(org: str, title: str) -> str:
    """Fallback dedup key for re-posts where the URL changes between runs."""
    return f"{org.strip().lower()}::{title.strip().lower()}"


def is_duplicate(url: str, existing_canonical_urls: set[str]) -> bool:
    return canonical_url(url) in existing_canonical_urls
