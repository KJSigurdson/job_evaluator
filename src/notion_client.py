"""Notion integration (step 2). Query existing URLs; insert-only — never modifies existing rows."""
from __future__ import annotations

from src.schemas import NotionInsertRow


def fetch_existing_urls() -> set[str]:
    raise NotImplementedError


def insert_row(row: NotionInsertRow) -> None:
    raise NotImplementedError
