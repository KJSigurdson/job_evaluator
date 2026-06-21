"""IAP referral Google Doc scraper (step 3). Reads via Google Docs/Drive API with service account."""
from __future__ import annotations

from src.schemas import RawPosting


def fetch() -> list[RawPosting]:
    raise NotImplementedError
