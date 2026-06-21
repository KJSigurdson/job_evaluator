"""Notion integration. Insert-only: never reads back or modifies existing rows.

Both public functions accept an injectable *client* so tests can pass a fake without
hitting the real Notion API.
"""
from __future__ import annotations

from datetime import date

from src.dedup import canonical_url
from src.schemas import NotionInsertRow


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_existing_urls(client, db_id: str) -> set[str]:
    """Return canonical URLs of every role already in the Notion database.

    Paginates automatically. Used at pipeline start to build the dedup set.
    """
    canonical_urls: set[str] = set()
    cursor = None

    while True:
        kwargs: dict = {"database_id": db_id}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = client.databases.query(**kwargs)

        for page in response["results"]:
            url_prop = page["properties"].get("URL", {})
            raw = url_prop.get("url") or ""
            if raw:
                canonical_urls.add(canonical_url(raw))

        if not response.get("has_more"):
            break
        cursor = response["next_cursor"]

    return canonical_urls


def insert_row(client, db_id: str, row: NotionInsertRow) -> None:
    """Insert *row* as a new page. Raises RuntimeError if the URL is already present.

    The caller should have already checked dedup before calling this function;
    this is a second-line idempotency guard.
    """
    existing = fetch_existing_urls(client, db_id)
    if canonical_url(row.url) in existing:
        raise RuntimeError(f"insert_row: URL already in Notion, skipping: {row.url}")

    client.pages.create(
        parent={"database_id": db_id},
        properties=_build_properties(row),
    )


# ---------------------------------------------------------------------------
# Property mapping
# ---------------------------------------------------------------------------

def _build_properties(row: NotionInsertRow) -> dict:
    def text(val: str | None) -> dict:
        return {"rich_text": [{"text": {"content": val or ""}}]}

    def number(val: float | None) -> dict:
        return {"number": val}

    def date_prop(val: date | None) -> dict:
        return {"date": {"start": val.isoformat()} if val else None}

    return {
        "Role":                  {"title": [{"text": {"content": row.role}}]},
        "Org":                   text(row.org),
        "Org summary":           text(row.org_summary),
        "Source":                {"select": {"name": row.source}},
        "URL":                   {"url": row.url},
        "Date found":            date_prop(row.date_found),
        "Deadline":              date_prop(row.deadline),
        "Comp":                  text(row.comp),
        "Fit score":             number(row.fit_score),
        "Cause mission fit":     number(row.cause_mission_fit),
        "Role function fit":     number(row.role_function_fit),
        "Location compatibility": number(row.location_compatibility),
        "Seniority match":       number(row.seniority_match),
        "Comp adequacy":         number(row.comp_adequacy),
        "Values alignment":      number(row.values_alignment),
        "Skill growth":          number(row.skill_growth),
        "Why it fits":           text(row.why_fits),
        "Why it might not fit":  text(row.why_not_fits),
        "Emphasize in CV":       text(row.emphasize_in_cv),
        "De-emphasize":          text(row.deemphasize),
        "Status":                {"select": {"name": row.status}},
    }
