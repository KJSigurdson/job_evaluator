"""80,000 Hours job board scraper (step 4). JSON endpoint preferred; HTML+LLM fallback."""
from __future__ import annotations

from src.schemas import RawPosting


def fetch() -> list[RawPosting]:
    raise NotImplementedError
