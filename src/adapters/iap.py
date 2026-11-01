"""IAP referral Google Sheet adapter for the True North v2 ingest pipeline.

Reuses src/sources/iap.py's fetch_raw_rows() — the exact same Sheets API fetch
src/pipeline.py's v1 run already calls — completely unmodified. Only the PARSE
step is new: this module's parse_rows() builds JobPosting (the v2 adapter shape)
from the same raw Sheets rows iap.parse_rows() builds RawPosting (the v1 shape)
from. v1's own parse_rows()/RawPosting path is untouched; this is a second,
independent parse of the same raw data, not a replacement.

description: IAP has no dedicated description column — same as v1's own
raw_text, this is the only substantive text the sheet provides, so it's reused
as JobPosting.description rather than left empty.
"""
from __future__ import annotations

from src.adapters.base import FetchResult, JobPosting, SourceAdapter
from src.sources.iap import RawRow, fetch_raw_rows

SOURCE_BOARD = "iap"


class IAPAdapter(SourceAdapter):
    source_board = SOURCE_BOARD

    def fetch(self) -> FetchResult:
        # A Sheets read has no "total available" concept to compare against —
        # nb_hits=None, complete=True (nothing known to be missing).
        rows = fetch_raw_rows()
        return FetchResult(postings=parse_rows(rows), fetched=len(rows), nb_hits=None, complete=True)


def parse_rows(rows: list[RawRow]) -> list[JobPosting]:
    """Convert raw Sheets rows to JobPosting objects. Mirrors iap.parse_rows()'s
    header-row location, required-field check, and "various" title skip,
    independently — not imported from there, since that function returns
    RawPosting, a different shape."""
    header_idx = _find_header(rows)
    if header_idx is None:
        raise ValueError("Header row with 'Organization' in column A not found")

    postings: list[JobPosting] = []
    for row in rows[header_idx + 1:]:
        if _is_empty_row(row):
            continue

        c = _pad(row, 9)

        org = c[0]["text"].strip()
        title = c[1]["text"].strip()
        url = _extract_url(c[2])
        location = c[3]["text"].strip() or None
        comp = c[4]["text"].strip() or None
        cause_area = c[5]["text"].strip()

        if not org or not title or not url:
            continue
        if "various" in title.lower():
            continue

        description = " | ".join(filter(None, [org, title, location, comp, cause_area]))

        postings.append(JobPosting(
            url=url,
            org=org,
            title=title,
            description=description,
            location_field=location,
            source_board=SOURCE_BOARD,
        ))

    return postings


def _find_header(rows: list[RawRow]) -> int | None:
    for i, row in enumerate(rows):
        if row and row[0]["text"].strip().lower() == "organization":
            return i
    return None


def _is_empty_row(row: RawRow) -> bool:
    return not row or all(not c["text"].strip() for c in row)


def _pad(row: RawRow, length: int) -> RawRow:
    return row + [{"text": "", "url": None}] * max(0, length - len(row))


def _extract_url(cell) -> str | None:
    if cell["url"]:
        return cell["url"]
    text = cell["text"].strip()
    if text.startswith(("http://", "https://")):
        return text
    return None
