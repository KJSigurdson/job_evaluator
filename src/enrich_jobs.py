"""True North v2 enrichment worker: jobs with status='pending_enrich' -> structured
JobEnrichment (jsonb) + title/summary embedding, via the Anthropic Message Batches
API (cron work, not real-time — batch pricing, ~50% of the synchronous rate).

Additive only — reads/writes exclusively the v2 `jobs`/`run_metrics` tables.
Nothing here touches any v1 table or v1 code path.

Flow per run:
  1. Select every job with status='pending_enrich' (optionally capped via `limit`,
     for sampling before committing to the full backlog).
  2. Submit ONE Anthropic Batches API request per job (custom_id = job id).
  3. Poll until the batch finishes, fetch results.
  4. Parse each result via _merge_tool_use_blocks (same pattern as scoring.py's/
     enrich.py's identical helper — duplicated here per this codebase's established
     no-shared-utils convention) and validate against
     src.models.job_enrichment.JobEnrichment (strict).
  5. Any job that failed to parse/validate, or whose batch result wasn't
     "succeeded", gets ONE retry: a second, smaller batch containing only those
     jobs, with the validation/parse error appended to the prompt. A job that
     fails AGAIN is marked status='stale' and logged — NEVER written with partial
     enrichment.
  6. Successfully enriched jobs get a title+summary embedding via Voyage AI
     (voyage-3, 1024 dims) if VOYAGE_API_KEY is set; if not, the embedding step is
     skipped (embedding stays NULL) and this is reported clearly — enrichment
     itself is never blocked on the embedding key being absent.
  7. Batched writes to `jobs` (uniform-payload update, same NULL-fill-avoidance
     rule as src/ingest.py) and one `run_metrics` row with jobs_enriched, the
     real tier1_calls/tier2_calls-equivalent count, and estimated_cost_usd
     computed from the batch's actual reported token usage.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError

from src.models.job_enrichment import JobEnrichment

log = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"
_TOOL_NAME = "enrich_job"
_MAX_TOKENS = 1024
_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 3600  # batches can legitimately take up to 24h; this run waits up to 1h before giving up

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
# Response parsing — pure, unit-testable without hitting the Anthropic API
# ---------------------------------------------------------------------------

def _merge_tool_use_blocks(response, tool_name: str) -> dict:
    """Merge every tool_use content block named *tool_name* into one dict — same
    fix as scoring.py's/enrich.py's identical helper (a model can split structured
    output across multiple tool_use blocks instead of one). Duplicated, not
    imported, per this codebase's established no-shared-utils convention."""
    merged: dict = {}
    found = False
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            merged.update(block.input)
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in response.content")
    return merged


def _parse_enrichment(response) -> JobEnrichment:
    tool_input = _merge_tool_use_blocks(response, _TOOL_NAME)
    return JobEnrichment.model_validate(tool_input)


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


def _run_batch(anthropic_client, requests: list[dict]) -> dict:
    """Submit *requests* (custom_id -> params) as one Batches API call, poll until
    it finishes, and return {custom_id: MessageBatchIndividualResponse.result}.
    Blocking (time.sleep polling) — this module is invoked as a cron/manual job,
    not from within the real-time v1 pipeline.
    """
    batch = anthropic_client.messages.batches.create(requests=requests)
    log.info("Submitted batch %s with %d request(s)", batch.id, len(requests))

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while True:
        batch = anthropic_client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch {batch.id} did not finish within {_POLL_TIMEOUT_SECONDS}s")
        log.info("Batch %s still %s — polling again in %ds", batch.id, batch.processing_status, _POLL_INTERVAL_SECONDS)
        time.sleep(_POLL_INTERVAL_SECONDS)

    results = {}
    for item in anthropic_client.messages.batches.results(batch.id):
        results[item.custom_id] = item.result
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def enrich_jobs(
    *, client, anthropic_client, limit: int | None = None, dry_run: bool = False,
    job_ids: list[str] | None = None,
) -> dict:
    """Enrich every job with status='pending_enrich' (capped at *limit* if given —
    used for the sample run before committing to the full backlog). dry_run=True
    selects and reports counts without submitting any batch or writing anything.
    job_ids, if given, restricts selection to that specific set (still gated on
    status='pending_enrich') — used to hand-pick a representative sample (e.g. one
    that deliberately includes "Remote" postings) rather than taking arbitrary DB
    order.

    Returns a summary dict: {selected, enriched, stale, embeddings_skipped}.
    """
    query = client.table("jobs").select(
        "id,title,org,location_field,raw_description,canonical_url,source_board,content_hash"
    ).eq("status", "pending_enrich")
    if job_ids is not None:
        query = query.in_("id", job_ids)
    rows = query.execute().data or []
    if limit is not None:
        rows = rows[:limit]

    if dry_run:
        log.info("DRY RUN — would enrich %d job(s)", len(rows))
        return {"selected": len(rows), "enriched": 0, "stale": 0, "embeddings_skipped": not embeddings_available()}

    if not rows:
        log.info("No pending_enrich jobs found — nothing to do")
        return {"selected": 0, "enriched": 0, "stale": 0, "embeddings_skipped": not embeddings_available()}

    by_id = {row["id"]: row for row in rows}

    # Round 1
    requests = [
        _batch_request(row["id"], build_prompt(
            title=row["title"], org=row["org"],
            location_field=row.get("location_field"), description=row["raw_description"] or "",
        ))
        for row in rows
    ]
    results = _run_batch(anthropic_client, requests)
    total_api_calls = len(requests)

    enriched: dict[str, JobEnrichment] = {}
    failures: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for job_id, result in results.items():
        if result.type != "succeeded":
            failures[job_id] = f"batch result type={result.type!r}"
            continue
        usage = result.message.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        try:
            enriched[job_id] = _parse_enrichment(result.message)
        except (IndexError, AttributeError, ValidationError) as exc:
            failures[job_id] = str(exc)

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
        retry_results = _run_batch(anthropic_client, retry_requests)
        total_api_calls += len(retry_requests)

        still_failing: dict[str, str] = {}
        for job_id, result in retry_results.items():
            if result.type != "succeeded":
                still_failing[job_id] = f"batch result type={result.type!r} (retry)"
                continue
            usage = result.message.usage
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            try:
                enriched[job_id] = _parse_enrichment(result.message)
            except (IndexError, AttributeError, ValidationError) as exc:
                still_failing[job_id] = f"{exc} (retry)"

        failures = still_failing

    for job_id, reason in failures.items():
        log.error("Job %s permanently failed enrichment, marking stale: %s", job_id, reason)

    # Embeddings — skipped (not fatal) if VOYAGE_API_KEY absent OR the Voyage call
    # itself fails; a Voyage outage must not lose the enrichment (and its Anthropic
    # spend) that already succeeded. embedding stays NULL until a later backfill.
    embeddings: dict[str, list[float]] = {}
    if enriched and embeddings_available():
        job_ids = list(enriched.keys())
        texts = [f"{by_id[jid]['title']} {enriched[jid].summary}" for jid in job_ids]
        try:
            vectors = embed_texts(texts)
            embeddings = dict(zip(job_ids, vectors))
        except httpx.HTTPError as exc:
            log.warning(
                "Voyage embedding call failed — skipping embeddings for %d enriched job(s) this "
                "run (enrichment itself is unaffected): %s", len(enriched), exc,
            )
    elif enriched:
        log.warning(
            "VOYAGE_API_KEY not set — skipping embeddings for %d enriched job(s). "
            "Enrichment itself is NOT blocked; embedding stays NULL until the key exists.",
            len(enriched),
        )

    # Batched writes — uniform payload per group, same NULL-fill-avoidance rule as
    # src/ingest.py: enriched-job updates and stale-job updates are separate calls.
    now = datetime.now(timezone.utc).isoformat()
    # canonical_url/source_board/content_hash are NOT NULL with no default — Postgres
    # validates NOT NULL on an upsert's INSERT branch even though every one of these
    # rows already exists and only the ON CONFLICT DO UPDATE branch actually fires
    # (same quirk already hit and fixed in src/ingest.py), so every payload here must
    # carry them through even though this step conceptually only touches enrichment
    # fields (confirmed the hard way against production).
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

    if enriched_updates:
        client.table("jobs").upsert(enriched_updates, on_conflict="id").execute()
    if stale_updates:
        client.table("jobs").upsert(stale_updates, on_conflict="id").execute()

    # tier1_calls carries the total number of actual Anthropic requests made
    # (round 1 + the round-2 retry batch, if any) — this v2 enrichment worker only
    # has one call tier (Haiku), so tier2_calls stays NULL rather than overloading
    # it with an unrelated number.
    client.table("run_metrics").insert({
        "run_at": now,
        "jobs_enriched": len(enriched),
        "tier1_calls": total_api_calls,
        "estimated_cost_usd": round(
            total_input_tokens / 1_000_000 * _HAIKU_SYNC_INPUT_USD_PER_MTOK * _BATCH_DISCOUNT
            + total_output_tokens / 1_000_000 * _HAIKU_SYNC_OUTPUT_USD_PER_MTOK * _BATCH_DISCOUNT,
            4,
        ),
    }).execute()

    log.info(
        "Enrichment complete: %d selected, %d enriched, %d stale, embeddings_skipped=%s",
        len(rows), len(enriched), len(failures), not embeddings_available(),
    )
    return {
        "selected": len(rows), "enriched": len(enriched), "stale": len(failures),
        "embeddings_skipped": not embeddings_available(),
    }


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

    rows = (
        client.table("jobs")
        .select("id,title,canonical_url,source_board,content_hash,enrichment")
        .eq("status", "enriched")
        .is_("embedding", "null")
        .execute()
        .data
        or []
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
    client.table("jobs").upsert(updates, on_conflict="id").execute()

    log.info("backfill_embeddings: %d selected, %d embedded", len(rows), len(updates))
    return {"selected": len(rows), "embedded": len(updates), "failed": 0}
