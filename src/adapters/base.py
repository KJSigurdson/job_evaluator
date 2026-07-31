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


class SourceAdapter(abc.ABC):
    @abc.abstractmethod
    def fetch(self) -> list[JobPosting]:
        """Fetch every current posting from this source board."""
