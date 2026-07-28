"""One-off migration: reformat existing `matches` enrichment text into the bulleted
style enrich.py's prompt now produces (see enrich.py's FORMAT rules). Rows written
before that prompt change have why_fits/why_not_fits as prose paragraphs and
emphasize_in_cv/deemphasize items as full sentences — this rewrites ONLY those four
columns via a cheap, formatting-only LLM call. It does NOT re-score, re-enrich against
the original posting, or touch fit_score/dimension_scores/org_summary/status/
user_notes/discarded/anything else — it has no access to the original posting text and
doesn't need it, since it only reformats text already sitting in the matches table.

Usage:
    python -m src.reformat_match_enrichment --limit 50 [--dry-run] [--user-id ID]
    python -m src.reformat_match_enrichment --confirm-all [--dry-run] [--user-id ID]

--limit N       process at most N rows (required unless --confirm-all is given).
--confirm-all   process every matching row, unbounded — required in place of --limit
                so a bare run can't accidentally reformat every historical match
                unattended on first use.
--user-id ID    scope to a single user's rows instead of all users.
--dry-run       run the LLM reformatting pass but never call matches_store.update_
                enrichment — logs before/after per row instead of writing.

Writes go through matches_store.update_enrichment(), a targeted .update() scoped to
(user_id, canonical_url) that can only ever touch these four columns (see that
function's docstring) — never a full-row upsert.

Concurrency, backoff, and multi-block tool-use parsing mirror src/scoring.py's
score_many/_acall_llm and src/enrich.py's enrich_many/_acall_llm. Per the codebase's
existing convention (see those modules' _merge_tool_use_blocks docstrings), the
helpers are duplicated here rather than imported — each module keeps its own private
helpers self-contained.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random

from anthropic import AsyncAnthropic, RateLimitError
from pydantic import BaseModel, ValidationError

from src.matches_store import update_enrichment
from src.supabase_client import get_client

log = logging.getLogger(__name__)

_TIER1_MODEL_DEFAULT = "claude-haiku-4-5-20251001"  # same cheap-model default as scoring.py
_REFORMAT_CONCURRENCY_DEFAULT = 10
_RATE_LIMIT_MAX_ATTEMPTS = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 1.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 30.0
_LIST_FIELDS = ("emphasize_in_cv", "deemphasize")


class ReformatError(Exception):
    """Raised when the reformatting LLM call fails after one retry."""


class _ReformatOutput(BaseModel):
    """Subset of Tier2EnrichmentOutput's fields — just the four columns this migration
    touches. Kept local to this script (not added to schemas.py) since it's a one-off
    migration, not a shape the rest of the app needs to know about."""
    why_fits: str
    why_not_fits: str
    emphasize_in_cv: list[str]
    deemphasize: list[str]


# ---------------------------------------------------------------------------
# Prompt construction — pure, unit-testable without hitting the Anthropic API
# ---------------------------------------------------------------------------

def _build_prompt(row: dict) -> str:
    """Build the reformatting prompt for one matches row. Pure function so tests can
    assert its content (formatting-only, no re-evaluation) without stubbing the LLM."""
    return (
        "You are reformatting EXISTING text for a job match — you are NOT evaluating "
        "the job posting, re-scoring it, or forming a new judgment. Restructure the "
        "text below into the required format WITHOUT changing its substantive "
        "content, claims, or assessment of fit. Do not add any new claim that isn't "
        "already present, and do not remove or soften any existing claim. This is a "
        "formatting pass only.\n\n"
        f"CURRENT why_fits:\n{row['why_fits']}\n\n"
        f"CURRENT why_not_fits:\n{row['why_not_fits']}\n\n"
        f"CURRENT emphasize_in_cv:\n{json.dumps(row['emphasize_in_cv'])}\n\n"
        f"CURRENT deemphasize:\n{json.dumps(row['deemphasize'])}\n\n"
        "FORMAT — why_fits and why_not_fits: each is a single string containing a "
        "short bulleted list, not a paragraph. Every line starts with \"- \" and "
        "lines are separated by newlines within the string. Plain bullets only — "
        "no bold text, no markdown headers, no ** formatting. 3-6 bullets per "
        "field, one distinct claim per bullet (reuse the claims already present in "
        "the current text — do not invent new ones), no multi-sentence bullets.\n\n"
        "FORMAT — emphasize_in_cv and deemphasize: each list item is a short "
        "phrase (roughly 3-8 words) naming a specific experience or skill, not a "
        "sentence explaining why it matters. Example: \"IVC Evidensia "
        "volume-growth modelling\", not \"Emphasize the volume growth program you "
        "built at IVC Evidensia because it demonstrates quantitative modelling "
        "skills relevant to the role.\" Reword each existing item down to that short "
        "phrase — do not add or drop items."
    )


def _coerce_list_field(value):
    """Same defensive coercion as enrich.py's _coerce_list_field: if the model
    returns a JSON-array-in-a-string for emphasize_in_cv/deemphasize instead of a
    native array, parse it back into a real list rather than fail validation."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    return parsed if isinstance(parsed, list) else value


def _parse_reformat_output(tool_input: dict) -> _ReformatOutput:
    normalized = dict(tool_input)
    for field in _LIST_FIELDS:
        if field in normalized:
            normalized[field] = _coerce_list_field(normalized[field])
    return _ReformatOutput.model_validate(normalized)


def _merge_tool_use_blocks(response, tool_name: str) -> dict:
    """Merge every tool_use content block named *tool_name* into one dict — a model
    can occasionally split structured output across multiple tool_use blocks instead
    of one block with every key. See scoring.py's/enrich.py's identical helper for the
    full rationale; duplicated here per this codebase's no-shared-utils convention."""
    merged: dict = {}
    found = False
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            merged.update(block.input)
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in response.content")
    return merged


def _backoff_delay(attempt: int) -> float:
    """Duplicated from scoring.py's/enrich.py's identical helper."""
    base = min(_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt), _RATE_LIMIT_MAX_DELAY_SECONDS)
    jitter = base * random.uniform(-0.2, 0.2)
    return max(0.0, base + jitter)


# ---------------------------------------------------------------------------
# Real LLM call
# ---------------------------------------------------------------------------

async def _acall_llm(row: dict) -> _ReformatOutput:
    """Call Claude with tool-use to reformat one row's enrichment text. Own
    RateLimitError-specific backoff loop, same shape as scoring.py's/enrich.py's
    _acall_llm."""
    model = os.environ.get("TIER1_MODEL") or _TIER1_MODEL_DEFAULT
    prompt = _build_prompt(row)

    client = AsyncAnthropic()
    response = None
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                tools=[{
                    "name": "reformat_enrichment",
                    "description": "Reformat existing job-match enrichment text into the bulleted style.",
                    "input_schema": _ReformatOutput.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "reformat_enrichment"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except RateLimitError:
            if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            delay = _backoff_delay(attempt)
            log.info(
                "Reformat rate-limited for %s (attempt %d/%d) — backing off %.1fs",
                row.get("canonical_url"), attempt + 1, _RATE_LIMIT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    try:
        tool_input = _merge_tool_use_blocks(response, "reformat_enrichment")
        return _parse_reformat_output(tool_input)
    except (IndexError, AttributeError, ValidationError) as exc:
        raise ValueError(f"Reformat response parse error: {exc}") from exc


# ---------------------------------------------------------------------------
# Retry wrapper + bounded concurrency — mirrors ascore/aenrich + score_many/enrich_many
# ---------------------------------------------------------------------------

async def _areformat_with_retry(row: dict, _allm_fn=None) -> _ReformatOutput:
    """_allm_fn injectable (async callable taking a row dict) — stub in tests, omit
    in production. Retries once on parse/validation failure, then raises
    ReformatError — same one-shot-retry semantics as scoring.py/enrich.py."""
    llm_fn = _allm_fn if _allm_fn is not None else _acall_llm
    for attempt in range(2):
        try:
            return await llm_fn(row)
        except Exception as exc:
            if attempt == 0:
                log.warning("Reformat attempt 1 failed for %s: %s", row.get("canonical_url"), exc)
                continue
            raise ReformatError(
                f"Reformat failed after retry for {row.get('canonical_url')}: {exc}"
            ) from exc
    raise ReformatError("unreachable")  # satisfies type checker


async def reformat_many(
    rows: list[dict],
    max_concurrency: int = 10,
    _allm_fn=None,
) -> list[_ReformatOutput | ReformatError]:
    """Reformat every row in *rows* concurrently, bounded by max_concurrency via an
    asyncio.Semaphore. Preserves *rows*' input order (one output per input, same
    index). Per-row ReformatError is caught inside that row's task and returned as a
    value (not raised) — mirrors score_many's/enrich_many's isolation contract: one
    bad row is logged and skipped, not fatal to the batch. Any OTHER exception is a
    real bug and propagates out of gather, aborting the whole batch.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(row: dict) -> _ReformatOutput | ReformatError:
        async with semaphore:
            try:
                return await _areformat_with_retry(row, _allm_fn)
            except ReformatError as exc:
                return exc

    return await asyncio.gather(*(_one(row) for row in rows))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    user_id: str | None = None,
    confirm_all: bool = False,
    _client=None,
    _allm_fn=None,
) -> dict:
    """Fetch matches rows, reformat their enrichment text, and write the result back
    via matches_store.update_enrichment() — unless dry_run, in which case
    update_enrichment is never called; before/after is logged per row instead.

    Returns {"processed": int, "updated": int, "failed": int} for callers/tests.
    """
    if limit is None and not confirm_all:
        raise ValueError("--limit N or --confirm-all is required")

    client = _client if _client is not None else get_client()

    query = client.table("matches").select(
        "user_id,canonical_url,why_fits,why_not_fits,emphasize_in_cv,deemphasize"
    )
    if user_id is not None:
        query = query.eq("user_id", user_id)
    resp = query.execute()
    rows = resp.data or []

    if limit is not None:
        rows = rows[:limit]

    concurrency = int(os.environ.get("REFORMAT_CONCURRENCY") or _REFORMAT_CONCURRENCY_DEFAULT)
    outcomes = asyncio.run(reformat_many(rows, max_concurrency=concurrency, _allm_fn=_allm_fn))

    processed = 0
    updated = 0
    failed = 0

    for row, outcome in zip(rows, outcomes):
        processed += 1
        if isinstance(outcome, ReformatError):
            failed += 1
            log.error("Reformat FAILED for user=%s url=%s: %s", row["user_id"], row["canonical_url"], outcome)
            continue

        reformatted = outcome
        if dry_run:
            log.info(
                "DRY RUN — user=%s url=%s\n"
                "  BEFORE why_fits: %r\n  AFTER  why_fits: %r\n"
                "  BEFORE why_not_fits: %r\n  AFTER  why_not_fits: %r\n"
                "  BEFORE emphasize_in_cv: %r\n  AFTER  emphasize_in_cv: %r\n"
                "  BEFORE deemphasize: %r\n  AFTER  deemphasize: %r",
                row["user_id"], row["canonical_url"],
                row["why_fits"], reformatted.why_fits,
                row["why_not_fits"], reformatted.why_not_fits,
                row["emphasize_in_cv"], reformatted.emphasize_in_cv,
                row["deemphasize"], reformatted.deemphasize,
            )
            continue

        update_enrichment(
            client, row["user_id"], row["canonical_url"],
            why_fits=reformatted.why_fits,
            why_not_fits=reformatted.why_not_fits,
            emphasize_in_cv=reformatted.emphasize_in_cv,
            deemphasize=reformatted.deemphasize,
        )
        updated += 1

    log.info(
        "Reformat migration complete | processed=%d updated=%d failed=%d%s",
        processed, updated, failed, "  (DRY RUN — nothing written)" if dry_run else "",
    )
    return {"processed": processed, "updated": updated, "failed": failed}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="One-off migration: reformat existing matches enrichment text to bulleted style"
    )
    parser.add_argument("--dry-run", action="store_true", help="Log before/after per row; never write")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Cap rows processed")
    parser.add_argument("--user-id", type=str, default=None, metavar="ID", help="Scope to one user")
    parser.add_argument(
        "--confirm-all", action="store_true",
        help="Process every matching row, unbounded (required in place of --limit)",
    )
    args = parser.parse_args()

    if args.limit is None and not args.confirm_all:
        parser.error("--limit N or --confirm-all is required")

    from dotenv import load_dotenv
    load_dotenv()

    summary = run(
        dry_run=args.dry_run, limit=args.limit, user_id=args.user_id, confirm_all=args.confirm_all,
    )
    print(json.dumps(summary, indent=2))
