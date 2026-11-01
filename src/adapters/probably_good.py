"""Probably Good adapter for the True North v2 ingest pipeline.

Reuses src/sources/probablygood.py's fetch_raw_hits_bucketed() — same
AlgoliaSearchKey GraphQL call + secured-key Algolia query src/pipeline.py's
v1 run already makes (via that module's fetch_raw_hits(), which stays
unmodified), but split across facet-value queries to get past Algolia's
paginationLimitedTo cliff the same way the 80k adapter already does — see
that source module's fetch_raw_hits_bucketed() for the full rationale. Only
the PARSE step is new: this module's parse_hits() builds JobPosting (the v2
adapter shape) from the same raw Algolia hits probablygood.parse_hits()
builds RawPosting (the v1 shape) from. v1's own parse_hits()/RawPosting path
is untouched; this is a second, independent parse of the same raw data, not
a replacement.

description uses the `description` field directly (unlike 80k's adapter, this
board's Algolia index actually populates it — matching v1's own choice).
"""
from __future__ import annotations

import re

from src.adapters.base import FetchResult, JobPosting, SourceAdapter
from src.sources.probablygood import fetch_raw_hits_bucketed

SOURCE_BOARD = "probably_good"


class ProbablyGoodAdapter(SourceAdapter):
    source_board = SOURCE_BOARD

    def fetch(self) -> FetchResult:
        result = fetch_raw_hits_bucketed()
        return FetchResult(
            postings=parse_hits(result.hits), fetched=result.fetched,
            nb_hits=result.nb_hits, complete=result.complete,
        )


def parse_hits(hits: list[dict]) -> list[JobPosting]:
    """Convert Algolia hit dicts to JobPosting objects. Mirrors
    probablygood.parse_hits()'s title/url required-field check and HTML-stripping,
    independently — not imported from there, since that function returns
    RawPosting, a different shape."""
    postings: list[JobPosting] = []

    for hit in hits:
        title = (hit.get("title") or "").strip()
        url = (hit.get("url_external") or "").strip()
        if not title or not url:
            continue

        org = ((hit.get("org") or {}).get("name") or "").strip()
        location = ", ".join(_tag_names(hit.get("locations"))) or None
        description = _strip_html(hit.get("description") or "")

        postings.append(JobPosting(
            url=url,
            org=org,
            title=title,
            description=description,
            location_field=location,
            source_board=SOURCE_BOARD,
        ))

    return postings


def _tag_names(tags: list[dict] | None) -> list[str]:
    if not tags:
        return []
    return [n for t in tags if (n := (t.get("name") or "").strip())]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()
