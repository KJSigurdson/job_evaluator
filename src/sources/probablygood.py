"""Probably Good job board source.

Key fetch:        fetch_search_key() → (app_id, api_key, index)
Network boundary: fetch_raw_hits() (key fetch + Algolia pagination)
Pure logic:       parse_hits(hits) (raw dicts → List[RawPosting])

Run as a script for a live smoke-test:
    python -m src.sources.probablygood
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import httpx

from src.schemas import RawPosting

log = logging.getLogger(__name__)

_GRAPHQL_URL = "https://backend.jobs.probablygood.org/api/graphql/AlgoliaSearchKey"
# query includes jobs_object_id_field — confirmed live (2026-11) that its ABSENCE
# is exactly what broke this integration starting mid-September: the backend
# started returning {"data": null, "errors": [{"message": "An unknown error
# occurred."}]} for the old query (no other difference — same operationName,
# same variables={}, no persistedQuery extensions either side, minimal headers
# with no cookies/sec-ch-ua needed) and 200s with real data the moment this one
# field is added back. Whatever changed on their platform (a persisted-document/
# allow-list update is the likely shape, per their vendor's error message) means
# the query sent here must match their currently-registered document exactly —
# if they add/remove fields again, this will need updating again the same way.
_GRAPHQL_BODY = {
    "operationName": "AlgoliaSearchKey",
    "variables": {},
    "query": (
        "query AlgoliaSearchKey {\n"
        "  algolia_search_key {\n"
        "    api_key\n"
        "    app_id\n"
        "    index_name\n"
        "    index_name_sorted_by_votes\n"
        "    index_name_profiles\n"
        "    index_name_jobs\n"
        "    index_name_jobs_sorted_by_closes_at\n"
        "    jobs_object_id_field\n"
        "    __typename\n"
        "  }\n"
        "}"
    ),
}
_GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://jobs.probablygood.org",
    "Referer": "https://jobs.probablygood.org/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch() -> list[RawPosting]:
    return parse_hits(fetch_raw_hits())


def fetch_search_key() -> tuple[str, str, str]:
    """Fetch a short-lived secured Algolia key from the ProbablyGood backend.

    Returns (app_id, api_key, index_name_jobs). The key has visibility filters
    baked in — no additional Algolia filter params are needed.
    """
    with httpx.Client(timeout=15) as client:
        resp = client.post(_GRAPHQL_URL, headers=_GRAPHQL_HEADERS, json=_GRAPHQL_BODY)
        resp.raise_for_status()

    body = resp.json()
    if body.get("data") is None:
        raise RuntimeError(f"AlgoliaSearchKey returned data:null — body: {resp.text}")

    key_data = body["data"]["algolia_search_key"]
    return key_data["app_id"], key_data["api_key"], key_data["index_name_jobs"]


def fetch_raw_hits() -> list[dict]:
    """Fetch a secured Algolia key, then retrieve jobs from the main index via /query.

    Index choice: AlgoliaSearchKey exposes two jobs replicas:
      - jobs_prod                      sorts by posted_at_unix DESC (newest-first)
      - jobs_by_closes_at              sorts by closes_at (deadline) — not useful here

    jobs_prod is correct — newest-first, so the postings closest to the
    pagination cap are the oldest ones, not the newest.

    NOTE (corrected 2026-11, was previously misattributed to a backend ACL
    restriction): the 1000-hit cap below is Algolia's own paginationLimitedTo,
    the SAME mechanism src/sources/eightyk.py's fetch_raw_hits() hits — nbHits
    is reported correctly (confirmed live: 1812) but nbPages is reported as 1
    regardless, and page>=1 returns 0 hits even though more exist. True North
    v2's adapter uses fetch_raw_hits_bucketed() below instead, for exactly the
    same reason the 80k adapter moved off this function's equivalent. This
    function is kept as-is for v1 (src/pipeline.py), which still has the
    original 1000-hit-cap limitation — unaffected by this change.
    """
    app_id, api_key, index = fetch_search_key()

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
# fetch_raw_hits() above (v1's own fetch() still calls that, unchanged). Same
# fix, same rationale, as src/sources/eightyk.py's fetch_raw_hits_bucketed() —
# see that function's module comment for the full explanation of WHY a single
# query can't retrieve more than 1000 hits here regardless of nbPages.
#
# tags_area.name was chosen over the other candidate facets on this index
# (tags_experience.name, tags_workload.name, has_salary, org.organization_type.name,
# tags_skill.name) by querying each live: tags_area.name's largest bucket is
# ~35% of the total (headroom as the pool grows), and several of the other
# candidates have a SINGLE bucket already over 1000 hits today (tags_workload.name
# 's "Full-Time" bucket is 1396; org.organization_type.name's "Nonprofit" is
# 1103) — which would immediately hit the SAME pagination cliff one level
# down. A direct union-of-per-bucket-objectIDs check against the live index
# matched nbHits exactly (full coverage — no hit lacks a tags_area entry).
# ---------------------------------------------------------------------------

_BUCKET_FACET = "tags_area.name"
_BUCKET_PAGE_SIZE = 1000


@dataclass
class BucketedFetchResult:
    hits: list[dict]
    nb_hits: int
    fetched: int
    complete: bool
    bucket_sizes: dict[str, int] = field(default_factory=dict)


def fetch_raw_hits_bucketed() -> BucketedFetchResult:
    """Fetch every hit via per-facet-value queries (see module comment above),
    union-deduped by objectID. complete=True iff the deduped union's size
    equals the index's own reported nbHits AND no individual bucket itself
    returned a full 1000-hit page (same two-part check as eightyk.py's
    equivalent) — either failing is logged as a WARNING, not raised; callers
    treat complete=False as "this run's fetch may be missing postings", not
    an error.
    """
    app_id, api_key, index = fetch_search_key()
    url = f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
        "Content-Type": "application/json",
    }

    resp = httpx.post(url, headers=headers, json={"hitsPerPage": 0, "page": 0, "facets": [_BUCKET_FACET]}, timeout=30)
    resp.raise_for_status()
    discovery = resp.json()
    nb_hits = discovery.get("nbHits", 0)
    facet_values = list((discovery.get("facets") or {}).get(_BUCKET_FACET, {}).keys())

    if not facet_values:
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
    """Convert ProbablyGood Algolia hit dicts to RawPosting objects."""
    postings: list[RawPosting] = []

    for hit in hits:
        title = (hit.get("title") or "").strip()
        url   = (hit.get("url_external") or "").strip()

        if not title or not url:
            continue

        org      = ((hit.get("org") or {}).get("name") or "").strip()
        location = ", ".join(_tag_names(hit.get("locations")))
        seniority = ", ".join(_tag_names(hit.get("tags_experience"))) or None
        cause_area = ", ".join(_tag_names(hit.get("tags_area"))) or None
        comp      = _extract_comp(hit)
        deadline  = _extract_unix_date(hit, "closes_at_unix")
        posted_at = _extract_unix_date(hit, "posted_at_unix")

        raw_text = " | ".join(filter(None, [
            title,
            org,
            location,
            ", ".join(_tag_names(hit.get("tags_area"))),
            ", ".join(_tag_names(hit.get("tags_skill"))),
            ", ".join(_tag_names(hit.get("tags_workload"))),
            _strip_html(hit.get("description") or ""),
        ]))

        postings.append(RawPosting(
            url=url,
            title=title,
            org=org,
            source="probably_good",
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

def _tag_names(tags: list[dict] | None) -> list[str]:
    """Extract non-empty name strings from a list of tag objects."""
    if not tags:
        return []
    return [n for t in tags if (n := (t.get("name") or "").strip())]


def _extract_comp(hit: dict) -> str | None:
    salary = hit.get("salary_text")
    if isinstance(salary, str) and salary.strip():
        return salary.strip()
    return None


def _extract_unix_date(hit: dict, key: str) -> date | None:
    val = hit.get(key)
    if not val or not isinstance(val, (int, float)) or val <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(val)).date()
    except (OSError, OverflowError, ValueError):
        return None


def _extract_deadline(hit: dict) -> date | None:
    return _extract_unix_date(hit, "closes_at_unix")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# ---------------------------------------------------------------------------
# Smoke-test entry point  (python -m src.sources.probablygood)
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
