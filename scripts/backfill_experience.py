"""One-off backfill: migrate two legacy users' free-text profiles.experience
blobs into structured user_experience/user_education rows, via one Claude
extraction call per user. Scope is hardcoded to exactly two users (_USER_IDS
below) — this is a one-off migration, not a general tool; discovering users
dynamically here would risk silently processing someone never reviewed.

Two modes:
  Default (no flags): DRY RUN. Fetches each user's profiles.experience, runs
    extraction, writes scripts/backfill_preview.json. No DB writes at all.
  --commit: reads scripts/backfill_preview.json back (does NOT re-extract —
    commits exactly what was reviewed) and inserts into user_experience/
    user_education. Also sets profiles.current_title, but only if it's
    currently null (see _derive_current_title).

Usage:
    python -m scripts.backfill_experience              # dry run
    python -m scripts.backfill_experience --commit      # commit reviewed preview
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anthropic

from src.supabase_client import get_client

log = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 4096

# Hardcoded on purpose — see module docstring.
_USER_IDS = {
    "fd5eb2df-a9cf-4a62-9907-6bd98c8c9600": "Kalle Sigurdson",
    "61a4ce03-1a7b-4394-ac51-283a7866264c": "Adam Holt",
}

_PREVIEW_PATH = Path(__file__).parent / "backfill_preview.json"

_SYSTEM_PROMPT = """You extract structured career history from a free-text profile blob.

Return ONLY valid JSON — no markdown code fences, no commentary before or after
the JSON — matching EXACTLY this schema:

{
  "experience": [
    {"title": string, "organization": string, "start_date": string|null,
     "end_date": string|null, "description": string|null}
  ],
  "education": [
    {"institution": string, "degree": string, "field": string|null,
     "start_year": string|null, "end_year": string|null}
  ],
  "skills": [string]
}

Rules:
- Dates in "experience" must be "YYYY-MM" strings ONLY where the source text
  actually supports that precision. If the source doesn't state a month, or
  doesn't state a date at all, use null. NEVER guess or fabricate a date.
- end_date must be null for a role the source describes as current/ongoing
  (e.g. "Present", "current").
- Only populate "description" if the source contains actual descriptive text
  about the role beyond its title/organization/dates (responsibilities,
  achievements, scope, etc.). NEVER write a description that merely restates
  the title or employer name. If there is no such text, set description to
  null.
- start_year/end_year in "education" follow the same never-guess rule as
  dates above — null if not stated.
- Do NOT invent roles, employers, or institutions that are not present in the
  source text. If the source lists no education, return an empty education
  list. If it lists no skills, return an empty skills list.
"""


def _strip_markdown_fences(text: str) -> str:
    """Defensive cleanup in case the model wraps its JSON in a ```/```json
    fence despite the system prompt telling it not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _validate_extraction(data: object) -> dict:
    """Raise ValueError on anything that doesn't match the schema documented in
    _SYSTEM_PROMPT. Never returns a partial/best-effort result — callers treat
    any exception here as a full extraction failure for that user."""
    if not isinstance(data, dict):
        raise ValueError(f"top-level JSON must be an object, got {type(data).__name__}")

    experience = data.get("experience")
    education = data.get("education")
    skills = data.get("skills")
    if not isinstance(experience, list):
        raise ValueError("'experience' must be a list")
    if not isinstance(education, list):
        raise ValueError("'education' must be a list")
    if not isinstance(skills, list):
        raise ValueError("'skills' must be a list")

    for i, entry in enumerate(experience):
        if not isinstance(entry, dict):
            raise ValueError(f"experience[{i}] is not an object")
        for key in ("title", "organization", "start_date", "end_date", "description"):
            if key not in entry:
                raise ValueError(f"experience[{i}] missing required key {key!r}")

    for i, entry in enumerate(education):
        if not isinstance(entry, dict):
            raise ValueError(f"education[{i}] is not an object")
        for key in ("institution", "degree", "field", "start_year", "end_year"):
            if key not in entry:
                raise ValueError(f"education[{i}] missing required key {key!r}")

    return {"experience": experience, "education": education, "skills": skills}


def extract_experience(anthropic_client: anthropic.Anthropic, experience_text: str) -> dict:
    """One synchronous Messages API call. Returns the validated
    {"experience": [...], "education": [...], "skills": [...]} dict. Raises
    (json.JSONDecodeError / ValueError) on any parse/schema failure — callers
    catch this and record an error entry instead of a partial extraction."""
    response = anthropic_client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": experience_text}],
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    cleaned = _strip_markdown_fences(raw_text)
    data = json.loads(cleaned)
    return _validate_extraction(data)


def _derive_current_title(experience: list[dict]) -> str | None:
    """title + organization of the current role (end_date null), formatted as
    "<title> at <organization>". If several are current, picks the one with
    the latest start_date ("YYYY-MM" strings sort correctly as plain strings;
    a missing start_date sorts lowest). None if there's no current role, or
    the winning entry is missing a title/organization to build the string from.
    """
    current = [e for e in experience if not e.get("end_date")]
    if not current:
        return None
    current.sort(key=lambda e: e.get("start_date") or "", reverse=True)
    best = current[0]
    title, organization = best.get("title"), best.get("organization")
    if not title or not organization:
        return None
    return f"{title} at {organization}"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def run_dry_run() -> None:
    client = get_client()
    anthropic_client = anthropic.Anthropic()

    profiles_by_id = {
        row["user_id"]: row
        for row in client.table("profiles").select("user_id,display_name,experience")
        .in_("user_id", list(_USER_IDS)).execute().data
    }

    preview: list[dict] = []
    for user_id, fallback_name in _USER_IDS.items():
        profile = profiles_by_id.get(user_id)
        display_name = (profile or {}).get("display_name") or fallback_name

        if profile is None:
            print(f"{display_name} ({user_id}): no profiles row found — skipping")
            preview.append({"user_id": user_id, "error": "no profiles row found"})
            continue

        source_blob = profile.get("experience") or ""
        print(f"Extracting experience for {display_name} ({user_id})...")
        try:
            extracted = extract_experience(anthropic_client, source_blob)
        except Exception as exc:
            log.error("Extraction failed for %s (%s): %s", display_name, user_id, exc)
            preview.append({"user_id": user_id, "error": str(exc)})
            continue

        preview.append({
            "user_id": user_id,
            "display_name": display_name,
            "source_blob": source_blob,
            "extracted": extracted,
            "derived_current_title": _derive_current_title(extracted["experience"]),
        })
        print(
            f"  -> {len(extracted['experience'])} experience entr(ies), "
            f"{len(extracted['education'])} education entr(ies)"
        )

    _PREVIEW_PATH.write_text(json.dumps(preview, indent=2))
    print(f"\nWrote {_PREVIEW_PATH} — review it, then re-run with --commit.")


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

def run_commit() -> None:
    if not _PREVIEW_PATH.exists():
        print(f"No preview file at {_PREVIEW_PATH} — run without --commit first to generate one.")
        return

    preview = json.loads(_PREVIEW_PATH.read_text())
    client = get_client()

    for entry in preview:
        user_id = entry["user_id"]
        display_name = entry.get("display_name", user_id)

        if "error" in entry:
            print(f"Skipping {display_name} ({user_id}): preview recorded an extraction error: {entry['error']}")
            continue

        existing = client.table("user_experience").select("id").eq("user_id", user_id).execute().data
        if existing:
            print(f"Skipping {display_name} ({user_id}): {len(existing)} existing user_experience row(s) already present.")
            continue

        extracted = entry["extracted"]
        experience_rows = [
            {
                "user_id": user_id,
                "sort_order": i,
                "title": item.get("title"),
                "organization": item.get("organization"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
                "description": item.get("description"),
            }
            for i, item in enumerate(extracted.get("experience", []))
        ]
        education_rows = [
            {
                "user_id": user_id,
                "sort_order": i,
                "institution": item.get("institution"),
                "degree": item.get("degree"),
                "field": item.get("field"),
                "start_year": item.get("start_year"),
                "end_year": item.get("end_year"),
            }
            for i, item in enumerate(extracted.get("education", []))
        ]

        if experience_rows:
            client.table("user_experience").insert(experience_rows).execute()
        if education_rows:
            client.table("user_education").insert(education_rows).execute()

        derived_title = entry.get("derived_current_title")
        if derived_title:
            profile_rows = client.table("profiles").select("current_title").eq("user_id", user_id).execute().data
            current_title = profile_rows[0].get("current_title") if profile_rows else None
            if current_title is None:
                client.table("profiles").update({"current_title": derived_title}).eq("user_id", user_id).execute()
                print(f"{display_name}: set profiles.current_title -> {derived_title!r}")
            else:
                print(f"{display_name}: profiles.current_title already set ({current_title!r}) — leaving untouched")

        print(f"{display_name} ({user_id}): inserted {len(experience_rows)} experience row(s), {len(education_rows)} education row(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", action="store_true",
        help="Commit the reviewed scripts/backfill_preview.json to the DB (default: dry run)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    if args.commit:
        run_commit()
    else:
        run_dry_run()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
