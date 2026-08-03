"""True North v2 enrichment worker: jobs with status='pending_enrich' -> structured
JobEnrichment (jsonb) + title/summary embedding, via the Anthropic Message Batches
API (cron work, not real-time — batch pricing, ~50% of the synchronous rate).

Additive only — reads/writes exclusively the v2 `jobs`/`run_metrics`/
`batch_submissions`/`batch_results` tables. Nothing here touches any v1 table or
v1 code path.

Flow per run:
  1. Select every job with status='pending_enrich' (optionally capped via `limit`,
     for sampling before committing to the full backlog).
  2. Submit ONE Anthropic Batches API request per job (custom_id = job id) — or,
     if a prior run left an outstanding (submitted but not yet retrieved) batch
     for this module, resume polling THAT one instead of submitting new work
     (see src.supabase_client.find_outstanding_batch — protects against a runner
     dying mid-poll and the next run duplicating an already-paid-for batch).
  3. Poll until the batch finishes, fetch results, and immediately persist every
     raw result to `batch_results` (src.supabase_client.persist_batch_results) —
     BEFORE any parsing/validation/DB-write risk, so a crash anywhere after this
     point never loses paid-for Anthropic output; see reprocess_staged() for the
     recovery path.
  4. Parse each result via _merge_tool_use_blocks (same pattern as scoring.py's/
     enrich.py's identical helper — duplicated here per this codebase's established
     no-shared-utils convention) and validate against
     src.models.job_enrichment.JobEnrichment (strict).
  5. Any job that failed to parse/validate, or whose batch result wasn't
     "succeeded", gets ONE retry: a second, smaller batch containing only those
     jobs, with the validation/parse error appended to the prompt. A job that
     fails AGAIN is marked status='stale' and logged — NEVER written with partial
     enrichment.
  6. run_metrics is written right after usage is tallied (round 1 + round 2, if
     any) — BEFORE the final `jobs` write — so a crash in that final write no
     longer loses the cost record, only requires the write to be retried (see
     reprocess_staged()).
  7. Successfully enriched jobs get a title+summary embedding via Voyage AI
     (voyage-3.5, 1024 dims) if VOYAGE_API_KEY is set; if not, the embedding step is
     skipped (embedding stays NULL) and this is reported clearly — enrichment
     itself is never blocked on the embedding key being absent.
  8. Batched writes to `jobs` (uniform-payload update, same NULL-fill-avoidance
     rule as src/ingest.py).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from pydantic import ValidationError

from src.models.job_enrichment import JobEnrichment
from src.supabase_client import (
    chunked_upsert,
    fetch_all,
    find_outstanding_batch,
    persist_batch_results,
    record_batch_retrieved,
    record_batch_submitted,
)

log = logging.getLogger(__name__)

_MODULE = "enrich_jobs"
_MODEL = "claude-haiku-4-5"
_TOOL_NAME = "enrich_job"
_MAX_TOKENS = 1024
_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 3600  # batches can legitimately take up to 24h; this run waits up to 1h before giving up

# A job whose content_hash keeps changing (see src/ingest.py's cross-board/minor-
# update guardrails — this is the backstop behind those) is re-enriched at most
# once per this many days, never more often, regardless of how often status
# flips back to pending_enrich. A job with enriched_at still inside the window
# stays pending_enrich (so it's picked up the moment the window elapses) but is
# deferred out of THIS run's selection, never charged for twice in one window.
_RE_ENRICH_COOLDOWN_DAYS = 7

# Batches API pricing is documented at 50% of the synchronous per-token rate.
# *** VERIFY THESE AGAINST https://www.anthropic.com/pricing BEFORE TRUSTING
# *** estimated_cost_usd — hardcoded here because there is no API to read current
# *** pricing from, and pricing changes over time. Flagged explicitly in this run's
# *** report, not silently trusted.
_HAIKU_SYNC_INPUT_USD_PER_MTOK = 1.00
_HAIKU_SYNC_OUTPUT_USD_PER_MTOK = 5.00
_BATCH_DISCOUNT = 0.5

_VOYAGE_MODEL = "voyage-3.5"
_VOYAGE_DIMENSIONS = 1024
_VOYAGE_BATCH_SIZE = 128  # Voyage's per-request cap

_JOB_SELECT_COLUMNS = "id,title,org,location_field,raw_description,canonical_url,source_board,content_hash,enriched_at"


# ---------------------------------------------------------------------------
# Prompt — see this module's docstring; DO NOT invoke the Batches API with this
# until it's been reviewed (see enrich_jobs.py's caller / the task that added this).
# ---------------------------------------------------------------------------

_SENIORITY_KEYWORD_MAP = """\
    intern:   intern, internship, trainee, apprentice, new grad, graduate
    junior:   junior, jr, entry level, entry-level
    mid:      mid, mid-level, intermediate
    senior:   senior, sr, lead, staff
    director: director, head of, principal, vp, vice president, chief"""


def build_prompt(*, title: str, org: str, location_field: str | None, description: str) -> str:
    """The v2 enrichment prompt. Pure/testable — no API call — so its exact text
    can be inspected and reviewed without spending anything."""
    return (
        "Analyze the following job posting and extract structured enrichment data.\n\n"
        "JOB POSTING:\n"
        f"Title: {title}\n"
        f"Organization: {org}\n"
        f"Location field (as listed by the source board): {location_field or '(not stated)'}\n\n"
        f"Description:\n{description}\n\n"
        "Return the following fields:\n\n"
        "1. summary: A neutral, ~100-word summary of the role — what the org does, "
        "what the role involves, and any standout requirements. Do not editorialize "
        "or assess fit; describe only what the posting states.\n\n"
        "2. seniority: Classify this role into one or more of exactly five buckets — "
        "intern, junior, mid, senior, director — based on the title and description. "
        "Use this keyword mapping as your primary signal (the same mapping this "
        "organization's existing gate logic uses, for consistency):\n"
        f"{_SENIORITY_KEYWORD_MAP}\n"
        "(\"Lead\" maps to senior and \"Principal\" maps to director — deliberate, not "
        "a typo.) If the title/description doesn't clearly map to any bucket, use "
        "your judgment based on described years of experience or scope of "
        "responsibility rather than leaving this empty.\n\n"
        "3. cause_area: List of cause areas this role relates to (e.g. \"global "
        "health\", \"AI safety\", \"animal welfare\", \"biosecurity\"), if stated or "
        "clearly implied. Empty list if none is evident.\n\n"
        "4. skill_tags: List of specific skills/technologies/competencies this role "
        "requires or emphasizes (e.g. \"Python\", \"stakeholder management\", "
        "\"SQL\").\n\n"
        "5. salary: {min, max, currency}, each independently optional/null if not "
        "stated. Do not guess a figure that isn't in the text.\n\n"
        "6. work_arrangement — read this section carefully, it is the most "
        "error-prone field:\n"
        "   - mode: one of remote, hybrid, onsite, unspecified — based on how the "
        "role is actually described, not assumed from \"remote\" alone. If the "
        "location field names one or more specific physical places (a city, "
        "\"San Francisco Bay Area\", an office address) and neither the location "
        "field nor the description body mentions remote/hybrid work anywhere, "
        "classify this as onsite — do NOT fall back to unspecified just because "
        "the word \"onsite\" itself isn't used. This applies the same way whether "
        "one city is named or several (e.g. \"New York, San Francisco\" with no "
        "remote/hybrid language is onsite, not unspecified). Reserve unspecified "
        "for postings that give no location/work-mode signal at all.\n"
        "   - remote_scope: one of global, region_restricted, unspecified.\n"
        "     CRITICAL: \"Remote\" stated without further qualification is NOT "
        "automatically global. Scan the ENTIRE description body — not just the "
        "location field — for work-authorization or eligibility language (e.g. "
        "\"must be authorized to work in the United States\", \"must be based in "
        "the EU\", \"candidates must have a valid visa for...\"). If such language "
        "restricts eligibility to specific countries/regions, set remote_scope to "
        "region_restricted and populate eligible_regions accordingly, even if the "
        "location field alone just says \"Remote\".\n"
        "     If you find genuinely NO restriction stated ANYWHERE in the posting — "
        "neither the location field nor the description body — set remote_scope to "
        "unspecified. Do NOT guess \"global\" in the absence of a stated "
        "restriction; \"unspecified\" and \"global\" are different things and must "
        "not be conflated.\n"
        "   - eligible_regions: list of ISO country codes (e.g. \"US\", \"SE\", "
        "\"IN\") or recognized region groups (\"EU\", \"EMEA\") this role is open "
        "to, derived from stated restrictions. Interpret shorthand consistently:\n"
        "       \"US-remote\", \"remote (EST hours)\", \"must be authorized to work "
        "in the US\" -> [\"US\"]\n"
        "       \"Remote, EU\" -> [\"EU\"]\n"
        "     Empty list if remote_scope is global or unspecified.\n"
        "   - timezone_constraint: a short free-text description of any stated "
        "timezone requirement (e.g. \"within 3 hours of US Eastern\"), or null if "
        "none is stated.\n"
        "   - restriction_type: authorization if the restriction is about legal "
        "work eligibility/visa status, timezone if it's purely about working-hours "
        "overlap, or null if there is no restriction (remote_scope is global or "
        "unspecified). Use authorization when the posting states BOTH an "
        "authorization requirement and a timezone preference, since the "
        "authorization requirement is the binding constraint.\n"
        "     IMPORTANT: only set restriction_type to authorization if the "
        "DESCRIPTION BODY itself contains explicit eligibility/visa/work-"
        "authorization language (e.g. \"must be authorized to work in...\", "
        "\"must be a resident of...\", \"visa sponsorship not available\"). A bare "
        "region name in the location field alone (e.g. \"Remote, USA\" with "
        "nothing further in the body) supports populating eligible_regions via "
        "the shorthand rule above, but is NOT sufficient by itself to justify "
        "restriction_type=authorization — leave restriction_type null in that "
        "case rather than inferring a legal-eligibility reason that isn't stated "
        "anywhere in the text.\n\n"
        "7. posting_type: classify what KIND of listing this is — job boards mix "
        "real employment with other opportunity types, and downstream matching "
        "needs to tell them apart. Choose exactly one:\n"
        "   - job: a standard paid employment role with a defined position, "
        "reporting line, and ongoing responsibilities.\n"
        "   - internship: a fixed-term, typically entry-level training position "
        "explicitly labeled as an internship.\n"
        "   - fellowship: a funded, time-limited individual placement, usually "
        "research- or policy-oriented (e.g. a named fellowship or scholar "
        "program like Fulbright).\n"
        "   - volunteer: explicitly unpaid volunteer work.\n"
        "   - funding: a grant, prize, or funding opportunity for a project or "
        "organization — the applicant isn't joining a team, they're requesting "
        "money (e.g. \"grants provided for...\", \"apply for funding\").\n"
        "   - program: a structured course, accelerator, incubator, or training "
        "program that isn't employment and isn't individually-funded research "
        "(e.g. a multi-week course, an incubator cohort).\n"
        "   - speculative: an open-ended \"expression of interest\", \"general "
        "application\", \"talent pool\", or speculative application not tied to "
        "one specific open role.\n"
        "     If the posting is a standard paid role, use job — do not default "
        "to speculative just because the description is short.\n\n"
        "Call the tool with your findings for all fields above."
    )


def build_retry_prompt(*, title: str, org: str, location_field: str | None,
                        description: str, validation_error: str) -> str:
    """Same prompt, with the prior attempt's validation error appended — used for
    the one allowed retry after a parse/validation failure."""
    base = build_prompt(title=title, org=org, location_field=location_field, description=description)
    return (
        f"{base}\n\n"
        "NOTE: A previous attempt at this task failed validation with the "
        f"following error — correct it in this attempt:\n{validation_error}"
    )


# ---------------------------------------------------------------------------
# Response parsing — pure, unit-testable without hitting the Anthropic API.
# Operates on plain dicts (result.model_dump(mode="json") — see _run_batch()),
# not Anthropic SDK objects, so the live path and reprocess_staged()'s replay
# path (fed from batch_results.raw jsonb) share identical parsing code with no
# dual-shape handling.
# ---------------------------------------------------------------------------

def _merge_tool_use_blocks(message: dict, tool_name: str) -> dict:
    """Merge every tool_use content block named *tool_name* into one dict — same
    fix as scoring.py's/enrich.py's identical helper (a model can split structured
    output across multiple tool_use blocks instead of one). Duplicated, not
    imported, per this codebase's established no-shared-utils convention."""
    merged: dict = {}
    found = False
    for block in message["content"]:
        if block["type"] == "tool_use" and block["name"] == tool_name:
            merged.update(block["input"])
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in message content")
    return merged


def _parse_enrichment(message: dict) -> JobEnrichment:
    tool_input = _merge_tool_use_blocks(message, _TOOL_NAME)
    return JobEnrichment.model_validate(tool_input)


def _parse_one_result(result: dict) -> tuple[JobEnrichment | None, dict | None, str | None]:
    """Given one raw batch result dict, return (enrichment, usage, error) —
    exactly one of (enrichment, error) is non-None. usage is None only when the
    batch item itself didn't succeed (nothing to tally). Shared by the live
    round-1/round-2 loops and reprocess_staged(), so recovery uses the exact
    same per-item logic as a normal run."""
    if result["type"] != "succeeded":
        return None, None, f"batch result type={result['type']!r}"
    usage = result["message"]["usage"]
    try:
        return _parse_enrichment(result["message"]), usage, None
    except (IndexError, AttributeError, ValidationError) as exc:
        return None, usage, str(exc)


_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Output structured enrichment data extracted from a job posting.",
    "input_schema": JobEnrichment.model_json_schema(),
}


# ---------------------------------------------------------------------------
# Batches API plumbing
# ---------------------------------------------------------------------------

def _batch_request(job_id: str, prompt: str) -> dict:
    return {
        "custom_id": job_id,
        "params": {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _run_batch(anthropic_client, requests: list[dict], *, client, module: str = _MODULE) -> dict:
    """Submit *requests* (custom_id -> params) as one Batches API call — or, if a
    prior run left an outstanding (submitted but not yet retrieved) batch for
    *module*, resume polling THAT one instead, ignoring *requests* entirely
    (whatever's already in flight is trusted as-is; any newly-pending work not
    covered by it is simply picked up on the next normal run). Poll until
    finished, fetch results, persist every raw result to `batch_results`
    immediately (before any parsing/write risk), and return {custom_id: raw_dict}.

    Blocking (time.sleep polling) — this module is invoked as a cron/manual job,
    not from within the real-time v1 pipeline.
    """
    batch_id = find_outstanding_batch(client, module)
    if batch_id is not None:
        log.warning(
            "Resuming outstanding batch %s for module %r instead of submitting new work "
            "(a prior run likely didn't finish polling it)", batch_id, module,
        )
    else:
        batch = anthropic_client.messages.batches.create(requests=requests)
        batch_id = batch.id
        record_batch_submitted(client, batch_id, module)
        log.info("Submitted batch %s with %d request(s)", batch_id, len(requests))

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while True:
        batch = anthropic_client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch {batch_id} did not finish within {_POLL_TIMEOUT_SECONDS}s")
        log.info("Batch %s still %s — polling again in %ds", batch_id, batch.processing_status, _POLL_INTERVAL_SECONDS)
        time.sleep(_POLL_INTERVAL_SECONDS)

    results = {}
    for item in anthropic_client.messages.batches.results(batch_id):
        results[item.custom_id] = item.result.model_dump(mode="json")
    persist_batch_results(client, batch_id, results)
    record_batch_retrieved(client, batch_id)
    return results


# ---------------------------------------------------------------------------
# Embeddings — Voyage AI, stubbed behind VOYAGE_API_KEY's presence
# ---------------------------------------------------------------------------

def embeddings_available() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed *texts* via Voyage AI's REST API directly (no voyageai package
    installed — this stays a plain httpx call so nothing new is added to
    requirements.txt). Raises if called without VOYAGE_API_KEY set — callers must
    check embeddings_available() first and skip this step entirely otherwise.

    Chunks *texts* into groups of at most _VOYAGE_BATCH_SIZE (Voyage's per-request
    cap) — one HTTP call per chunk, results concatenated back in the original
    order. A caller embedding the full backlog in one call would otherwise exceed
    the API's limit.
    """
    api_key = os.environ["VOYAGE_API_KEY"]
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _VOYAGE_BATCH_SIZE):
        chunk = texts[i:i + _VOYAGE_BATCH_SIZE]
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": chunk, "model": _VOYAGE_MODEL, "output_dimension": _VOYAGE_DIMENSIONS},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings.extend(item["embedding"] for item in data["data"])
    return embeddings


def _embed_enriched(enriched: dict[str, JobEnrichment], by_id: dict[str, dict]) -> dict[str, list[float]]:
    """Best-effort title+summary embedding for every successfully-enriched job.
    Never raises — a Voyage outage or absent key must not lose enrichment that
    already succeeded (and was already paid for)."""
    if not enriched:
        return {}
    if not embeddings_available():
        log.warning(
            "VOYAGE_API_KEY not set — skipping embeddings for %d enriched job(s). "
            "Enrichment itself is NOT blocked; embedding stays NULL until the key exists.",
            len(enriched),
        )
        return {}
    job_ids = list(enriched.keys())
    texts = [f"{by_id[jid]['title']} {enriched[jid].summary}" for jid in job_ids]
    try:
        vectors = embed_texts(texts)
        return dict(zip(job_ids, vectors))
    except httpx.HTTPError as exc:
        log.warning(
            "Voyage embedding call failed — skipping embeddings for %d enriched job(s) this "
            "run (enrichment itself is unaffected): %s", len(enriched), exc,
        )
        return {}


def _write_enrichment(client, *, enriched: dict[str, JobEnrichment], failures: dict[str, str],
                       by_id: dict[str, dict], embeddings: dict[str, list[float]]) -> None:
    """Batched writes to `jobs` — uniform payload per group, same
    NULL-fill-avoidance rule as src/ingest.py: enriched-job updates and
    stale-job updates are separate calls. canonical_url/source_board/
    content_hash are NOT NULL with no default — Postgres validates NOT NULL on
    an upsert's INSERT branch even though every one of these rows already
    exists and only the ON CONFLICT DO UPDATE branch actually fires (same
    quirk already hit and fixed in src/ingest.py), so every payload here must
    carry them through even though this step conceptually only touches
    enrichment fields (confirmed the hard way against production).
    """
    now = datetime.now(timezone.utc).isoformat()
    enriched_updates = [
        {
            "id": job_id,
            "canonical_url": by_id[job_id]["canonical_url"],
            "source_board": by_id[job_id]["source_board"],
            "content_hash": by_id[job_id]["content_hash"],
            "enrichment": enrichment.model_dump(mode="json"),
            "embedding": embeddings.get(job_id),
            "enriched_at": now,
            "status": "enriched",
        }
        for job_id, enrichment in enriched.items()
    ]
    stale_updates = [
        {
            "id": job_id,
            "canonical_url": by_id[job_id]["canonical_url"],
            "source_board": by_id[job_id]["source_board"],
            "content_hash": by_id[job_id]["content_hash"],
            "status": "stale",
        }
        for job_id in failures
    ]

    # chunked_upsert, not one .upsert().execute() for the whole list: confirmed
    # the hard way against production — an unchunked ~1136-row upsert here (each
    # row carrying a jsonb enrichment blob) hit Postgres's statement timeout and
    # lost the entire batch's enrichment (Anthropic spend included).
    if enriched_updates:
        chunked_upsert(client.table("jobs"), enriched_updates, on_conflict="id")
    if stale_updates:
        chunked_upsert(client.table("jobs"), stale_updates, on_conflict="id")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _split_cooldown(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Split *rows* into (eligible, deferred_job_ids). A row is deferred when its
    enriched_at is more recent than _RE_ENRICH_COOLDOWN_DAYS ago — its
    content_hash changed again since then (status flipped back to
    pending_enrich by src/ingest.py) but the cooldown caps actual re-enrich
    spend to at most once per window. status stays pending_enrich for a
    deferred row, so the run right after the cutoff elapses picks it up
    normally — nothing here is lost, only delayed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=_RE_ENRICH_COOLDOWN_DAYS)
    eligible, deferred_ids = [], []
    for row in rows:
        enriched_at = row.get("enriched_at")
        if enriched_at and datetime.fromisoformat(enriched_at) > cutoff:
            deferred_ids.append(row["id"])
        else:
            eligible.append(row)
    return eligible, deferred_ids


def enrich_jobs(
    *, client, anthropic_client, limit: int | None = None, dry_run: bool = False,
    job_ids: list[str] | None = None, run_id: str | None = None, trigger: str | None = None,
) -> dict:
    """Enrich every job with status='pending_enrich' (capped at *limit* if given —
    used for the sample run before committing to the full backlog), except jobs
    re-enriched within the last _RE_ENRICH_COOLDOWN_DAYS (see _split_cooldown) —
    those are deferred, not selected, this run. dry_run=True selects and reports
    counts without submitting any batch or writing anything. job_ids, if given,
    restricts selection to that specific set (still gated on
    status='pending_enrich') — used to hand-pick a representative sample (e.g. one
    that deliberately includes "Remote" postings) rather than taking arbitrary DB
    order.

    Returns a summary dict: {selected, enriched, stale, embeddings_skipped, deferred}.
    """
    query = client.table("jobs").select(_JOB_SELECT_COLUMNS).eq("status", "pending_enrich")
    if job_ids is not None:
        query = query.in_("id", job_ids)
    # fetch_all, not a plain .execute(): PostgREST silently truncates a response
    # at db.max_rows (1000) — the pending_enrich backlog can exceed that.
    rows = fetch_all(query)
    rows, deferred_ids = _split_cooldown(rows)
    if limit is not None:
        rows = rows[:limit]

    if dry_run:
        log.info("DRY RUN — would enrich %d job(s) (%d deferred by re-enrich cooldown)", len(rows), len(deferred_ids))
        return {
            "selected": len(rows), "enriched": 0, "stale": 0,
            "embeddings_skipped": not embeddings_available(), "deferred": len(deferred_ids),
        }

    if not rows:
        log.info("No pending_enrich jobs found — nothing to do (%d deferred by cooldown)", len(deferred_ids))
        return {
            "selected": 0, "enriched": 0, "stale": 0,
            "embeddings_skipped": not embeddings_available(), "deferred": len(deferred_ids),
        }

    by_id = {row["id"]: row for row in rows}

    # Round 1
    requests = [
        _batch_request(row["id"], build_prompt(
            title=row["title"], org=row["org"],
            location_field=row.get("location_field"), description=row["raw_description"] or "",
        ))
        for row in rows
    ]
    results = _run_batch(anthropic_client, requests, client=client)
    total_api_calls = len(requests)

    enriched: dict[str, JobEnrichment] = {}
    failures: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for job_id, result in results.items():
        enrichment, usage, error = _parse_one_result(result)
        if usage is not None:
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
        if enrichment is not None:
            enriched[job_id] = enrichment
        else:
            failures[job_id] = error

    # Round 2 — one retry for whatever failed, with the error appended
    if failures:
        log.warning("%d job(s) failed round 1, retrying once: %s", len(failures), list(failures.keys()))
        retry_requests = [
            _batch_request(job_id, build_retry_prompt(
                title=by_id[job_id]["title"], org=by_id[job_id]["org"],
                location_field=by_id[job_id].get("location_field"),
                description=by_id[job_id]["raw_description"] or "",
                validation_error=error,
            ))
            for job_id, error in failures.items()
        ]
        retry_results = _run_batch(anthropic_client, retry_requests, client=client, module=f"{_MODULE}_retry")
        total_api_calls += len(retry_requests)

        still_failing: dict[str, str] = {}
        for job_id, result in retry_results.items():
            enrichment, usage, error = _parse_one_result(result)
            if usage is not None:
                total_input_tokens += usage["input_tokens"]
                total_output_tokens += usage["output_tokens"]
            if enrichment is not None:
                enriched[job_id] = enrichment
            else:
                still_failing[job_id] = f"{error} (retry)"

        failures = still_failing

    for job_id, reason in failures.items():
        log.error("Job %s permanently failed enrichment, marking stale: %s", job_id, reason)

    # run_metrics written NOW — right after usage is tallied, BEFORE the final
    # `jobs` write below — so a crash in that write no longer loses the cost
    # record (confirmed the hard way: this used to be written last, and a
    # crashed write meant ~$2.65 of paid Anthropic output went unrecorded).
    now = datetime.now(timezone.utc).isoformat()
    client.table("run_metrics").insert({
        "run_at": now,
        "run_id": run_id,
        "stage": "enrich",
        "trigger": trigger,
        "jobs_enriched": len(enriched),
        # NOT tier1_calls — this is enrich_jobs.py's own Haiku call count, a
        # different stage from tier1_rerank.py's Tier 1 rerank calls. Writing it
        # into tier1_calls conflated the two (confirmed: every "Enrich" stage row
        # was double-booked into the Tier 1 column, making per-stage call volume
        # unreadable from run_metrics alone).
        "enrichment_calls": total_api_calls,
        "estimated_cost_usd": round(
            total_input_tokens / 1_000_000 * _HAIKU_SYNC_INPUT_USD_PER_MTOK * _BATCH_DISCOUNT
            + total_output_tokens / 1_000_000 * _HAIKU_SYNC_OUTPUT_USD_PER_MTOK * _BATCH_DISCOUNT,
            4,
        ),
    }).execute()

    embeddings = _embed_enriched(enriched, by_id)
    _write_enrichment(client, enriched=enriched, failures=failures, by_id=by_id, embeddings=embeddings)

    log.info(
        "Enrichment complete: %d selected, %d enriched, %d stale, %d deferred by cooldown, embeddings_skipped=%s",
        len(rows), len(enriched), len(failures), len(deferred_ids), not embeddings_available(),
    )
    return {
        "selected": len(rows), "enriched": len(enriched), "stale": len(failures),
        "embeddings_skipped": not embeddings_available(), "deferred": len(deferred_ids),
    }


def reprocess_staged(client, batch_id: str) -> dict:
    """Recovery path: reprocess unprocessed `batch_results` rows for *batch_id*
    WITHOUT calling the Anthropic API again — the raw results are already
    staged from a prior run's `_run_batch()` call that didn't finish (crashed
    somewhere after retrieval, e.g. during the final `jobs` write). Re-derives
    each job's original metadata (title/org/canonical_url/source_board/
    content_hash) from `jobs` by id, re-runs the SAME `_parse_one_result()`
    logic the live path uses, recomputes embeddings for anything that parses
    successfully (the original run's embeddings, if any were computed, were
    only ever in-memory and are lost along with everything else that crashed),
    and redoes the final `jobs` write. Does NOT retry a second time on parse
    failure (that would need a new LLM call, defeating "without re-calling the
    API") — anything that still fails to parse here is marked stale directly.
    Does NOT write run_metrics — the normal path already wrote it (right after
    usage tallying, before the write that crashed), so writing it again here
    would double-count cost that was already recorded.

    Returns {"selected", "enriched", "stale"}.
    """
    staged = fetch_all(
        client.table("batch_results").select("custom_id,raw").eq("batch_id", batch_id).eq("processed", False)
    )
    if not staged:
        log.info("reprocess_staged: no unprocessed batch_results for batch %s", batch_id)
        return {"selected": 0, "enriched": 0, "stale": 0}

    job_ids = [row["custom_id"] for row in staged]
    raw_by_job_id = {row["custom_id"]: row["raw"] for row in staged}
    rows = fetch_all(client.table("jobs").select(_JOB_SELECT_COLUMNS).in_("id", job_ids))
    by_id = {row["id"]: row for row in rows}

    enriched: dict[str, JobEnrichment] = {}
    failures: dict[str, str] = {}
    for row in staged:
        job_id = row["custom_id"]
        if job_id not in by_id:
            log.error("reprocess_staged: staged result for job %s but that job no longer exists — skipping", job_id)
            continue
        enrichment, _usage, error = _parse_one_result(row["raw"])
        if enrichment is not None:
            enriched[job_id] = enrichment
        else:
            failures[job_id] = error

    for job_id, reason in failures.items():
        log.error("reprocess_staged: job %s permanently failed enrichment, marking stale: %s", job_id, reason)

    embeddings = _embed_enriched(enriched, by_id)
    _write_enrichment(client, enriched=enriched, failures=failures, by_id=by_id, embeddings=embeddings)

    # raw is NOT NULL with no default — same upsert-INSERT-branch quirk as
    # `jobs`' canonical_url/source_board/content_hash, so it must be carried
    # through here too even though this write only means to touch `processed`.
    processed_ids = list(enriched.keys()) + list(failures.keys())
    if processed_ids:
        chunked_upsert(
            client.table("batch_results"),
            [{"batch_id": batch_id, "custom_id": jid, "raw": raw_by_job_id[jid], "processed": True} for jid in processed_ids],
            on_conflict="batch_id,custom_id",
        )

    log.info(
        "reprocess_staged: batch %s -> %d selected, %d enriched, %d stale",
        batch_id, len(staged), len(enriched), len(failures),
    )
    return {"selected": len(staged), "enriched": len(enriched), "stale": len(failures)}


def backfill_embeddings(client) -> dict:
    """Catch-up embedding pass: embeds title+summary for every status='enriched'
    job whose embedding is still NULL — covers jobs enriched before
    VOYAGE_API_KEY existed (or that hit a transient Voyage failure during their
    original enrich_jobs() run, per that step's own best-effort handling above).
    Idempotent and safe to re-run: only ever touches rows missing an embedding,
    and a failed call here writes nothing, so a re-run just retries the same set.
    """
    if not embeddings_available():
        log.warning("VOYAGE_API_KEY not set — backfill_embeddings is a no-op")
        return {"selected": 0, "embedded": 0, "failed": 0}

    rows = fetch_all(
        client.table("jobs")
        .select("id,title,canonical_url,source_board,content_hash,enrichment")
        .eq("status", "enriched")
        .is_("embedding", "null")
    )
    if not rows:
        log.info("backfill_embeddings: no enriched jobs missing an embedding")
        return {"selected": 0, "embedded": 0, "failed": 0}

    texts = [f"{row['title']} {(row.get('enrichment') or {}).get('summary', '')}" for row in rows]
    try:
        vectors = embed_texts(texts)
    except httpx.HTTPError as exc:
        log.error("backfill_embeddings: Voyage call failed, nothing written: %s", exc)
        return {"selected": len(rows), "embedded": 0, "failed": len(rows)}

    updates = [
        {
            "id": row["id"], "canonical_url": row["canonical_url"],
            "source_board": row["source_board"], "content_hash": row["content_hash"],
            "embedding": vector,
        }
        for row, vector in zip(rows, vectors)
    ]
    chunked_upsert(client.table("jobs"), updates, on_conflict="id")

    log.info("backfill_embeddings: %d selected, %d embedded", len(rows), len(updates))
    return {"selected": len(rows), "embedded": len(updates), "failed": 0}
