"""80,000 Hours adapter for the True North v2 ingest pipeline.

Reuses src/sources/eightyk.py's fetch_raw_hits() — the exact same network fetch
src/pipeline.py's v1 run already calls — completely unmodified. Only the PARSE step
is new: this module's parse_hits() builds JobPosting (the v2 adapter shape) from
the same raw Algolia hits eightyk.parse_hits() builds RawPosting (the v1 shape)
from. v1's own parse_hits()/RawPosting path is untouched; this is a second,
independent parse of the same raw data, not a replacement.

description uses description_short, matching v1's choice — the Algolia index's
plain `description` field is empty across every current hit (verified directly
against the live index), so description_short is the only description text this
source actually provides.
"""
from __future__ import annotations

import re

from src.adapters.base import FetchResult, JobPosting, SourceAdapter
from src.sources.eightyk import fetch_raw_hits_bucketed

SOURCE_BOARD = "80000_hours"


class EightyThousandHoursAdapter(SourceAdapter):
    source_board = SOURCE_BOARD

    def fetch(self) -> FetchResult:
        # fetch_raw_hits_bucketed(), not fetch_raw_hits() — the plain query
        # silently drops postings past Algolia's paginationLimitedTo cliff
        # once the index exceeds 1000 hits (confirmed against production:
        # nbHits=1015, nbPages=1, page=1 empty). v1's own fetch() still calls
        # fetch_raw_hits() directly, unaffected by this.
        result = fetch_raw_hits_bucketed()
        return FetchResult(
            postings=parse_hits(result.hits), fetched=result.fetched,
            nb_hits=result.nb_hits, complete=result.complete,
        )


def parse_hits(hits: list[dict]) -> list[JobPosting]:
    """Convert Algolia hit dicts to JobPosting objects. Mirrors eightyk.parse_hits()'s
    title/url required-field check and HTML-stripping, independently — not imported
    from there, since that function returns RawPosting, a different shape."""
    postings: list[JobPosting] = []

    for hit in hits:
        title = (hit.get("title") or "").strip()
        url = (hit.get("url_external") or "").strip()
        if not title or not url:
            continue

        org = (hit.get("company_name") or "").strip()
        location = ", ".join(hit.get("tags_location_80k") or []) or None
        description = _strip_html(hit.get("description_short") or "")

        postings.append(JobPosting(
            url=url,
            org=org,
            title=title,
            description=description,
            location_field=location,
            source_board=SOURCE_BOARD,
        ))

    return postings


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()
