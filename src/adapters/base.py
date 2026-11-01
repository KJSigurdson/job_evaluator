"""Base interface for True North v2 source adapters — the shared job-enrichment
ingest pipeline's contract (src/ingest.py consumes this, not src/schemas.RawPosting,
which stays the v1 pipeline's own shape). Each adapter wraps an EXISTING v1 source
module's fetch code rather than re-implementing scraping; see
src/adapters/eighty_thousand_hours.py.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class JobPosting:
    url: str
    org: str
    title: str
    description: str
    location_field: str | None
    source_board: str


@dataclass
class FetchResult:
    """What one adapter.fetch() call returns — postings plus enough fetch-
    completeness metadata for pipeline_v2.py to write a source_fetch_log row.
    fetched: raw hit/row count the source actually returned this call (before
    JobPosting parsing drops anything missing a required field) — NOT
    len(postings), so a parse-time skip doesn't masquerade as an incomplete
    fetch. nb_hits: the source's own reported total available, if the source
    exposes one (Algolia's nbHits); None for sources with no such concept
    (IAP's sheet, ProbablyGood currently unaudited for a cap of its own).
    complete: True iff this call is confident it got everything available —
    False means downstream code (closure, most notably) must not treat a
    missing posting as evidence the job is gone, only as "this source's fetch
    may have been partial this run."
    """
    postings: list[JobPosting]
    fetched: int
    nb_hits: int | None
    complete: bool


class SourceAdapter(abc.ABC):
    # Each concrete adapter sets this (reusing its module's existing
    # SOURCE_BOARD constant) so pipeline_v2.py can log a source_fetch_log row
    # even when fetch() raises before returning anything.
    source_board: str

    @abc.abstractmethod
    def fetch(self) -> FetchResult:
        """Fetch every current posting from this source board."""
