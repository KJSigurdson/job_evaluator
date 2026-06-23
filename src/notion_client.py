"""Notion integration. Insert-only: never reads back or modifies existing rows.

Both public functions accept an injectable *client* so tests can pass a fake without
hitting the real Notion API.

notion-client v3 API notes
--------------------------
* databases.query was removed in v3.0.0.  Querying pages in a database now uses
  data_sources.query(data_source_id=...) where the data_source_id is NOT the same
  as the database_id — it must be resolved via databases.retrieve first.
* pages.create now requires parent={"type": "data_source_id", "data_source_id": ...}
  instead of the old parent={"database_id": ...}.
"""
from __future__ import annotations

from datetime import date

from src.dedup import canonical_url
from src.schemas import NotionInsertRow

# Process-level cache: database_id → data_source_id.
# databases.retrieve is called at most once per db_id per process.
_ds_id_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_ds_id(client, db_id: str) -> str:
    """Return the data_source_id that corresponds to *db_id*.

    Caches the result so databases.retrieve is only called once per db_id
    per process (i.e. once per pipeline run).
    """
    if db_id not in _ds_id_cache:
        db = client.databases.retrieve(database_id=db_id)
        _ds_id_cache[db_id] = db["data_sources"][0]["id"]
    return _ds_id_cache[db_id]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_existing_urls(client, db_id: str) -> set[str]:
    """Return canonical URLs of every role already in the Notion database.

    Resolves the data_source_id from db_id (cached), then paginates via
    data_sources.query.  iterate_paginated_api handles cursor tracking.
    """
    from notion_client.helpers import iterate_paginated_api

    ds_id = _resolve_ds_id(client, db_id)
    canonical_urls: set[str] = set()
    for page in iterate_paginated_api(client.data_sources.query, data_source_id=ds_id):
        url_prop = (page.get("properties") or {}).get("URL", {})
        raw = url_prop.get("url") or ""
        if raw:
            canonical_urls.add(canonical_url(raw))

    return canonical_urls


def insert_row(client, db_id: str, row: NotionInsertRow) -> None:
    """Insert *row* as a new page. Raises RuntimeError if the URL is already present.

    The caller should have already checked dedup before calling this function;
    this is a second-line idempotency guard.
    """
    existing = fetch_existing_urls(client, db_id)
    if canonical_url(row.url) in existing:
        raise RuntimeError(f"insert_row: URL already in Notion, skipping: {row.url}")

    ds_id = _resolve_ds_id(client, db_id)
    client.pages.create(
        parent={"type": "data_source_id", "data_source_id": ds_id},
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
        "Cause fit":             number(row.cause_mission_fit),
        "Role fit":              number(row.role_function_fit),
        "Location fit":          number(row.location_compatibility),
        "Seniority fit":         number(row.seniority_match),
        "Comp fit":              number(row.comp_adequacy),
        "Values fit":            number(row.values_alignment),
        "Skill growth":          number(row.skill_growth),
        "Why it fits":           text(row.why_fits),
        "Why it might not fit":  text(row.why_not_fits),
        "Emphasize in CV":       text(row.emphasize_in_cv),
        "De-emphasize":          text(row.deemphasize),
        "Status":                {"select": {"name": row.status}},
    }
