"""80,000 Hours job board source via Algolia search index.

Network boundary: fetch_raw_hits() (HTTP + pagination)
Pure logic:       parse_hits(hits) (raw dicts → List[RawPosting])

Run as a script for a live smoke-test:
    python -m src.sources.eightyk
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import httpx

from src.schemas import RawPosting

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch() -> list[RawPosting]:
    return parse_hits(fetch_raw_hits())


def fetch_raw_hits() -> list[dict]:
    """POST to the Algolia /query endpoint, paginate through all pages, return all raw hits.

    NOTE: Both the 80k and ProbablyGood public API keys lack the 'browse' ACL,
    so the cursor-based /browse endpoint returns 403. We use /query with page
    pagination; the effective ceiling is Algolia's paginationLimitedTo (default
    1000). At 866 hits the 80k index is under that limit today, so all results
    are returned. If the index grows past 1000, a server-side key upgrade would
    be needed.
    """
    app_id = os.environ["ALGOLIA_APP_ID"]
    api_key = os.environ["ALGOLIA_API_KEY"]
    index   = os.environ["ALGOLIA_INDEX_80K"]

    url = f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
        "Content-Type": "application/json",
    }

    hits: list[dict] = []
    with httpx.Client(timeout=30) as client:
        for page in range(1000):  # safety ceiling; broken by nbPages check
            resp = client.post(url, headers=headers, json={"hitsPerPage": 1000, "page": page})
            resp.raise_for_status()
            data = resp.json()
            hits.extend(data.get("hits", []))
            if page >= data.get("nbPages", 1) - 1:
                break

    return hits


# ---------------------------------------------------------------------------
# Facet-bucketed fetch — True North v2's adapter uses this instead of
# fetch_raw_hits() above (which v1's own fetch() still calls, UNCHANGED).
#
# fetch_raw_hits()'s single unfiltered query hits Algolia's paginationLimitedTo
# cliff once the index's true nbHits exceeds 1000: Algolia reports the real
# nbHits (e.g. 1015) but reports nbPages=1 regardless and serves zero hits for
# page>=1 — confirmed directly against the live index (page=1 returns
# nbHits=0/nbPages=0/0 hits even though page=0's own response claims
# nbHits=1015). fetch_raw_hits()'s "if page >= nbPages - 1: break" check never
# even attempts page 1, so ~15 postings are silently dropped every run with no
# log line.
#
# The fix: split the SAME query into several smaller queries, one per value of
# a low-cardinality facet attribute, each comfortably under 1000 hits on its
# own, then union the per-bucket hits (deduped by objectID — a hit can carry
# more than one value for most candidate facets, so the SAME hit legitimately
# appears in multiple buckets). tags_exp_required was chosen over the other
# candidate facets available on this index (tags_country, tags_area,
# tags_role_type, tags_skill, tags_degree_required) by querying each live and
# comparing: tags_exp_required has only 5 values, its largest bucket is ~40%
# of the total (headroom as the pool grows — tags_degree_required's largest
# bucket was already ~86%), and a direct union-of-per-bucket-objectIDs check
# against the live index matched nbHits exactly (full coverage — no hit lacks
# an experience-required tag). Discovered dynamically every call (via the
# facets query below), not hardcoded, so a brand-new tag value Algolia starts
# returning tomorrow is picked up automatically rather than silently excluded
# from every bucket the way the ORIGINAL bug worked.
# ---------------------------------------------------------------------------

_BUCKET_FACET = "tags_exp_required"
_BUCKET_PAGE_SIZE = 1000


@dataclass
class BucketedFetchResult:
    hits: list[dict]
    nb_hits: int
    fetched: int
    complete: bool
    bucket_sizes: dict[str, int] = field(default_factory=dict)


def _algolia_query_url_and_headers() -> tuple[str, dict]:
    app_id = os.environ["ALGOLIA_APP_ID"]
    api_key = os.environ["ALGOLIA_API_KEY"]
    index = os.environ["ALGOLIA_INDEX_80K"]
    url = f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
        "Content-Type": "application/json",
    }
    return url, headers


def fetch_raw_hits_bucketed() -> BucketedFetchResult:
    """Fetch every hit via per-facet-value queries (see module comment above),
    union-deduped by objectID. complete=True iff the deduped union's size
    equals the index's own reported nbHits AND no individual bucket itself
    returned a full 1000-hit page (which would mean paginationLimitedTo is
    capping that bucket too, same cliff one level down) — either condition
    failing is logged as a WARNING, not raised; callers treat complete=False
    as "this run's fetch may be missing postings", not an error.
    """
    url, headers = _algolia_query_url_and_headers()

    resp = httpx.post(url, headers=headers, json={"hitsPerPage": 0, "page": 0, "facets": [_BUCKET_FACET]}, timeout=30)
    resp.raise_for_status()
    discovery = resp.json()
    nb_hits = discovery.get("nbHits", 0)
    facet_values = list((discovery.get("facets") or {}).get(_BUCKET_FACET, {}).keys())

    if not facet_values:
        # Facet came back empty (index misconfigured, or genuinely no hits) —
        # fall back to a single plain query rather than returning nothing;
        # still subject to the same pagination cap fetch_raw_hits() has, so
        # complete reflects that honestly. A single httpx.post() call (not
        # fetch_raw_hits()'s own httpx.Client pagination loop) deliberately —
        # keeps this whole function's network surface on one call shape.
        log.warning(
            "fetch_raw_hits_bucketed: facet %r returned no values — falling back to "
            "a single unbucketed page (same pagination cap as before)", _BUCKET_FACET,
        )
        resp = httpx.post(url, headers=headers, json={"hitsPerPage": _BUCKET_PAGE_SIZE, "page": 0}, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        fetched = len(hits)
        complete = fetched >= nb_hits
        if not complete:
            log.warning(
                "fetch_raw_hits_bucketed: fallback fetch got %d hit(s) but nbHits=%d — incomplete",
                fetched, nb_hits,
            )
        return BucketedFetchResult(hits=hits, nb_hits=nb_hits, fetched=fetched, complete=complete)

    by_object_id: dict[str, dict] = {}
    bucket_sizes: dict[str, int] = {}
    any_bucket_capped = False

    for value in facet_values:
        bucket_count = 0
        for page in range(1000):  # safety ceiling; broken by nbPages check, same as fetch_raw_hits()
            resp = httpx.post(url, headers=headers, json={
                "hitsPerPage": _BUCKET_PAGE_SIZE, "page": page,
                "facetFilters": [[f"{_BUCKET_FACET}:{value}"]],
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            page_hits = data.get("hits", [])
            for h in page_hits:
                by_object_id[h["objectID"]] = h
            bucket_count += len(page_hits)
            if len(page_hits) >= _BUCKET_PAGE_SIZE:
                any_bucket_capped = True
                log.warning(
                    "fetch_raw_hits_bucketed: bucket %s=%r itself returned a full %d-hit "
                    "page — may be hitting the same pagination cap one level down",
                    _BUCKET_FACET, value, _BUCKET_PAGE_SIZE,
                )
            if page >= data.get("nbPages", 1) - 1:
                break
        bucket_sizes[value] = bucket_count

    hits = list(by_object_id.values())
    fetched = len(hits)
    complete = (fetched == nb_hits) and not any_bucket_capped
    if not complete:
        log.warning(
            "fetch_raw_hits_bucketed: deduped union has %d hit(s), index reports nbHits=%d "
            "(any_bucket_capped=%s) — some postings may be missing this run. bucket_sizes=%s",
            fetched, nb_hits, any_bucket_capped, bucket_sizes,
        )
    return BucketedFetchResult(hits=hits, nb_hits=nb_hits, fetched=fetched, complete=complete, bucket_sizes=bucket_sizes)


# ---------------------------------------------------------------------------
# Pure parse — no network, fully testable
# ---------------------------------------------------------------------------

def parse_hits(hits: list[dict]) -> list[RawPosting]:
    """Convert Algolia hit dicts to RawPosting objects."""
    postings: list[RawPosting] = []

    for hit in hits:
        title = (hit.get("title") or "").strip()
        url   = (hit.get("url_external") or "").strip()

        if not title or not url:
            continue

        org      = (hit.get("company_name") or "").strip()
        location = ", ".join(hit.get("tags_location_80k") or [])
        seniority = ", ".join(hit.get("tags_exp_required") or []) or None
        cause_area = ", ".join(hit.get("tags_area") or []) or None
        comp      = _extract_comp(hit)
        deadline  = _extract_deadline(hit)
        # The Algolia index's actual field is "posted_at" — "posted_at_unix"
        # (the previous key here) doesn't exist on any hit, confirmed directly
        # against the live index, so this always returned None. v1's
        # _STALE_CUTOFF_DAYS recency filter (src/pipeline.py) was therefore
        # silently exempting every 80k posting lacking a `deadline` from that
        # filter instead of actually enforcing it — this fixes that for real
        # if v1 is ever re-enabled; v1 is currently disabled in production
        # (daily.yml's `if: false`), so this has no live effect today.
        posted_at = _extract_unix_date(hit, "posted_at")

        raw_text = " | ".join(filter(None, [
            title,
            org,
            location,
            ", ".join(hit.get("tags_area") or []),
            ", ".join(hit.get("tags_skill") or []),
            _strip_html(hit.get("description_short") or ""),
        ]))

        postings.append(RawPosting(
            url=url,
            title=title,
            org=org,
            source="80k",
            location=location or None,
            seniority=seniority,
            comp=comp,
            cause_area=cause_area,
            deadline=deadline,
            posted_at=posted_at,
            raw_text=raw_text,
        ))

    return postings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_unix_date(hit: dict, key: str) -> date | None:
    val = hit.get(key)
    if not val or not isinstance(val, (int, float)) or val <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(val)).date()
    except (OSError, OverflowError, ValueError):
        return None


def _extract_comp(hit: dict) -> str | None:
    """Return salary string if present and non-empty; None otherwise.

    salary_limit (9999999) and salary_type ("Not Found") are sentinels on other
    fields — we only check the salary string field itself.
    """
    salary = hit.get("salary")
    if isinstance(salary, str) and salary.strip():
        return salary.strip()
    return None


def _extract_deadline(hit: dict) -> date | None:
    return _extract_unix_date(hit, "closes_at")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# ---------------------------------------------------------------------------
# Smoke-test entry point  (python -m src.sources.eightyk)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    from dotenv import load_dotenv

    load_dotenv()

    raw_hits = fetch_raw_hits()
    print(f"Total raw hits: {len(raw_hits)}")

    if raw_hits:
        print("\n--- First raw hit (full) ---")
        print(_json.dumps(raw_hits[0], indent=2, default=str))

    postings = parse_hits(raw_hits)
    print(f"\nParsed {len(postings)} postings")
    for p in postings[:3]:
        print(_json.dumps(p.model_dump(mode="json"), indent=2, default=str))
