"""IAP referral Google Sheet source.

Network boundary: fetch_raw_rows() (auth + API call)
Pure logic:       parse_rows() (raw rows → List[RawPosting])

Run as a script for a live smoke-test:
    python -m src.sources.iap
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import TypedDict

from src.schemas import RawPosting

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


# ---------------------------------------------------------------------------
# Types shared with tests
# ---------------------------------------------------------------------------

class Cell(TypedDict):
    text: str        # formattedValue from the Sheets API
    url: str | None  # hyperlink target if cell contains one, else None


RawRow = list[Cell]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch() -> list[RawPosting]:
    return parse_rows(fetch_raw_rows())


def fetch_raw_rows() -> list[RawRow]:
    """Authenticate with the service account and pull all rows from the configured tab."""
    from googleapiclient.discovery import build  # deferred: not needed in pure-parse tests

    sheet_id = os.environ["IAP_SHEET_ID"]
    tab_name = os.environ["IAP_SHEET_TAB"]

    service = build("sheets", "v4", credentials=_load_credentials())
    result = (
        service.spreadsheets()
        .get(spreadsheetId=sheet_id, ranges=[tab_name], includeGridData=True)
        .execute()
    )

    sheets = result.get("sheets", [])
    sheet = next((s for s in sheets if s["properties"]["title"] == tab_name), None)
    if sheet is None:
        raise ValueError(f"Tab {tab_name!r} not found in spreadsheet {sheet_id!r}")

    row_data = sheet["data"][0].get("rowData", [])
    rows: list[RawRow] = []
    for row in row_data:
        cells: RawRow = [
            {"text": c.get("formattedValue") or "", "url": c.get("hyperlink")}
            for c in row.get("values", [])
        ]
        rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# Pure parse — no network, fully testable
# ---------------------------------------------------------------------------

def parse_rows(rows: list[RawRow]) -> list[RawPosting]:
    """Convert raw Sheets rows to RawPosting objects. Locates the header row dynamically."""
    header_idx = _find_header(rows)
    if header_idx is None:
        raise ValueError("Header row with 'Organization' in column A not found")

    postings: list[RawPosting] = []
    for row in rows[header_idx + 1:]:
        if _is_empty_row(row):
            break

        c = _pad(row, 9)

        org   = c[0]["text"].strip()
        title = c[1]["text"].strip()
        url   = _extract_url(c[2])
        location  = c[3]["text"].strip() or None
        comp      = c[4]["text"].strip() or None
        cause_area = c[5]["text"].strip()
        deadline  = _parse_date(c[6]["text"].strip())
        # Column H (index 7): not used
        # Column I (index 8): date added — included in raw_text for scorer context

        if not org or not title or not url:
            continue

        raw_text = " | ".join(filter(None, [org, title, location, comp, cause_area]))

        postings.append(RawPosting(
            url=url,
            title=title,
            org=org,
            source="iap",
            location=location,
            seniority=title,  # lets gate's junior/graduate/intern keyword checks scan the role name
            comp=comp,
            deadline=deadline,
            raw_text=raw_text,
        ))

    return postings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_header(rows: list[RawRow]) -> int | None:
    for i, row in enumerate(rows):
        if row and row[0]["text"].strip().lower() == "organization":
            return i
    return None


def _is_empty_row(row: RawRow) -> bool:
    return not row or all(not c["text"].strip() for c in row)


def _pad(row: RawRow, length: int) -> RawRow:
    return row + [{"text": "", "url": None}] * max(0, length - len(row))


def _extract_url(cell: Cell) -> str | None:
    """Prefer the hyperlink target; fall back to cell text if it already looks like a URL."""
    if cell["url"]:
        return cell["url"]
    text = cell["text"].strip()
    if text.startswith(("http://", "https://")):
        return text
    return None


def _parse_date(text: str) -> date | None:
    """Best-effort date parse; returns None on blank or unrecognised format."""
    if not text:
        return None
    from dateutil import parser as dp
    try:
        return dp.parse(text, fuzzy=False).date()
    except (ValueError, OverflowError, TypeError):
        return None


def _load_credentials():
    from google.oauth2 import service_account

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

    # File path takes precedence over an inline JSON blob
    if os.path.exists(raw):
        return service_account.Credentials.from_service_account_file(raw, scopes=_SCOPES)

    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must be a file path or a JSON blob"
        ) from exc

    return service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)


# ---------------------------------------------------------------------------
# Smoke-test entry point  (python -m src.sources.iap)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    from dotenv import load_dotenv

    load_dotenv()
    postings = fetch()
    print(f"Fetched {len(postings)} postings from IAP sheet")
    for p in postings[:3]:
        print(_json.dumps(p.model_dump(mode="json"), indent=2, default=str))
