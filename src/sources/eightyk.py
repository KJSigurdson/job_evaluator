"""80,000 Hours job board source via Algolia search index.

Network boundary: fetch_raw_hits() (HTTP + pagination)
Pure logic:       parse_hits(hits) (raw dicts → List[RawPosting])

Run as a script for a live smoke-test:
    python -m src.sources.eightyk
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime

import httpx

from src.schemas import RawPosting


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
        comp      = _extract_comp(hit)
        deadline  = _extract_deadline(hit)
        posted_at = _extract_unix_date(hit, "posted_at_unix")

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
