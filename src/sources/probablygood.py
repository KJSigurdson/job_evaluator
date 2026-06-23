"""Probably Good job board source.

Key fetch:        fetch_search_key() → (app_id, api_key, index)
Network boundary: fetch_raw_hits() (key fetch + Algolia pagination)
Pure logic:       parse_hits(hits) (raw dicts → List[RawPosting])

Run as a script for a live smoke-test:
    python -m src.sources.probablygood
"""
from __future__ import annotations

import re
from datetime import date, datetime

import httpx

from src.schemas import RawPosting

_GRAPHQL_URL = "https://backend.jobs.probablygood.org/api/graphql/AlgoliaSearchKey"
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

    jobs_prod is correct: with hitsPerPage=1000 the response covers ~5 weeks of
    recent postings (verified: first hit posted 2026-06-19, 1000th posted 2026-05-14).
    The 1477-hit total means ~477 older postings are beyond the cap, but since the
    pipeline runs daily and dedup skips seen URLs, nothing recent is lost.

    The secured key lacks the 'browse' ACL so the cursor-based /browse endpoint
    returns 403 — the 1000-result cap is a backend permission constraint.
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
