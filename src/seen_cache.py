"""Persisted seen-URL cache. Prevents re-scoring sub-threshold and gated-out roles.

Storage: state/seen.json  (JSON, version=1)
Shape:
  {
    "version": 1,
    "entries": {
      "<canonical_url>": {
        "first_seen": "<iso8601>",
        "verdict": "gated_out" | "below_threshold" | "inserted",
        "fit_score": <float> | null
      }
    }
  }

load_seen / save_seen are the only IO.  is_seen / record / prune are pure.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.dedup import canonical_url

log = logging.getLogger(__name__)

_EMPTY: dict = {"version": 1, "entries": {}}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_seen(path: Path | str) -> dict:
    """Load the cache from *path*. Returns an empty cache if the file is missing
    or unreadable — never raises."""
    path = Path(path)
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("unexpected shape")
        return data
    except Exception as exc:
        log.warning("seen_cache at %s is corrupt, starting fresh: %s", path, exc)
        return {"version": 1, "entries": {}}


def save_seen(path: Path | str, data: dict) -> None:
    """Atomically write *data* to *path* (write temp, then os.replace).

    Creates parent directories if they don't exist. A crashed run cannot
    leave the cache file truncated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Pure operations
# ---------------------------------------------------------------------------

def is_seen(url: str, data: dict) -> bool:
    """Return True if the canonical form of *url* is already in the cache."""
    return canonical_url(url) in data["entries"]


def record(
    data: dict,
    url: str,
    verdict: str,
    fit_score: float | None,
) -> dict:
    """Record (or update) the terminal verdict for *url*.

    Mutates *data* in place and returns it.  The first_seen timestamp is
    preserved if the URL is already present (only verdict/fit_score update).
    """
    key = canonical_url(url)
    if key in data["entries"]:
        data["entries"][key]["verdict"]   = verdict
        data["entries"][key]["fit_score"] = fit_score
    else:
        data["entries"][key] = {
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "verdict":    verdict,
            "fit_score":  fit_score,
        }
    return data


def prune(data: dict, max_age_days: int = 180) -> dict:
    """Return a new cache dict with entries older than *max_age_days* removed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    kept = {}
    for key, entry in data["entries"].items():
        try:
            first_seen = datetime.fromisoformat(entry["first_seen"])
            # fromisoformat returns an aware datetime if the string has a tz offset
            if first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)
            if first_seen >= cutoff:
                kept[key] = entry
        except (KeyError, ValueError):
            pass  # malformed entry — drop it
    return {"version": data.get("version", 1), "entries": kept}
