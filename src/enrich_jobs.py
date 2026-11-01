"""True North v2 enrichment worker: jobs with status='pending_enrich' -> structured
JobEnrichment (jsonb) + title/summary embedding, via the Anthropic Message Batches
API (cron work, not real-time — batch pricing, ~50% of the synchronous rate).

Additive only — reads/writes exclusively the v2 `jobs`/`run_metrics`/
`batch_submissions`/`batch_results` tables. Nothing here touches any v1 table or
v1 code path.

Flow per run:
  1. Select every job with status='pending_enrich' (optionally capped via `limit`,
     for sampling before committing to the full backlog).
  2. Submit ONE Anthropic Batches API request per job (custom_id = job id) via
     src.batch_poll.resume_or_submit — which resumes a prior run's outstanding
     (not yet retrieved) batch instead of submitting duplicate work if one exists,
     and NEVER blocks past its poll budget (~600s, see src.batch_poll). If the
     batch hasn't finished within budget, this returns a 'pending' outcome — not
     an error — and enrich_jobs() stops cleanly, propagating 'pending' to its
     caller (src.pipeline_v2.run(), which then skips matching on partial data
     this run). The next run, or src/drain_batches.py's hourly check, resumes it.
  3. Once the batch ends, results are fetched and immediately persisted to
     `batch_results` (src.supabase_client.persist_batch_results) — BEFORE any
     parsing/validation/DB-write risk, so a crash anywhere after this point never
     loses paid-for Anthropic output; see reprocess_staged() for the recovery path.
  4. Each result is classified by its terminal state (_classify_batch_result)
     before parsing:
       - succeeded -> parsed via _merge_tool_use_blocks (same pattern as
         scoring.py's/enrich.py's identical helper — duplicated here per this
         codebase's established no-shared-utils convention) and validated
         against src.models.job_enrichment.JobEnrichment (strict). A
         parse/validation failure here gets ONE retry: a second, smaller batch
         with the error appended to the prompt; still-failing after that retry
         is marked status='stale' — NEVER written with partial enrichment.
       - expired, canceled, or an errored item whose error type is NOT
         invalid_request_error (e.g. api_error, overloaded_error — transient/
         infra-side) -> requeued: left at status='pending_enrich' with
         enriched_at bumped to now, so src._split_cooldown's EXISTING
         re-enrich-cooldown mechanism naturally defers re-selecting it for
         _RE_ENRICH_COOLDOWN_DAYS rather than burning an LLM call on it again
         next run. No new "attempt counter" column — this reuses that
         mechanism deliberately.
       - errored with error type invalid_request_error -> genuinely terminal,
         marked status='stale' directly, no retry (a differently-worded prompt
         can't fix a request the API itself rejected).
  5. run_metrics is written right after usage is tallied — BEFORE the final
     `jobs` write — so a crash in that final write no longer loses the cost
     record, only requires the write to be retried (see reprocess_staged()). A
     run_metrics row is now ALSO written on the 'pending' path (previously this
     row simply never existed if the batch didn't finish in time).
  6. Successfully enriched jobs get a title+summary embedding via Voyage AI
     (voyage-3.5, 1024 dims) if VOYAGE_API_KEY is set; if not, the embedding step is
     skipped (embedding stays NULL) and this is reported clearly — enrichment
     itself is never blocked on the embedding key being absent.
  7. Batched writes to `jobs` (uniform-payload update, same NULL-fill-avoidance
     rule as src/ingest.py) — tolerant of a job_id with no matching `jobs` row
     (logged and skipped, not a KeyError): src/drain_batches.py's apply_drained_batch()
     derives job metadata fresh from `jobs` by id rather than carrying it over from
     whatever run originally submitted the batch, so a job deleted in between is a
     real, expected possibility there, not just defensive paranoia.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from pydantic import ValidationError

from src.batch_poll import DEFAULT_POLL_BUDGET_SECONDS, resume_or_submit
from src.models.job_enrichment import JobEnrichment
from src.supabase_client import chunked_upsert, fetch_all, fetch_all_in, mark_batch_results_processed

log = logging.getLogger(__name__)

_MODULE = "enrich_jobs"
_MODEL = "claude-haiku-4-5"
_TOOL_NAME = "enrich_job"
_MAX_TOKENS = 1024

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

_JOB_SELECT_COLUMNS = "id,title,org,location_field,raw_description,canonical_url,source_board,content_hash,enriched_at,reenrich_requested_at"


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
# Operates on plain dicts (result.model_dump(mode="json") — see src.batch_poll's
# _poll_claimed_batch()),
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
    """Given one raw batch result dict of type 'succeeded', return
    (enrichment, usage, error) — exactly one of (enrichment, error) is
    non-None. Callers must only pass results already classified as
    'succeeded' by _classify_batch_result — this function no longer branches
    on result["type"] itself (see _classify_batch_result for that)."""
    usage = result["message"]["usage"]
    try:
        return _parse_enrichment(result["message"]), usage, None
    except (IndexError, AttributeError, ValidationError) as exc:
        return None, usage, str(exc)


# ---------------------------------------------------------------------------
# Per-request terminal-state classification. The Batches API's per-item
# result "type" is one of succeeded/errored/canceled/expired (confirmed
# against the installed anthropic SDK's type stubs — anthropic.types.messages.
# message_batch_result.MessageBatchResult); an "errored" item additionally
# carries error.error.type, one of invalid_request_error/authentication_error/
# permission_error/not_found_error/rate_limit_error/timeout_error/
# overloaded_error/api_error/billing_error (anthropic.types.shared.error_type.
# ErrorType).
# ---------------------------------------------------------------------------

def _classify_batch_result(result: dict) -> str:
    """Classify one raw batch result into a disposition:
      - 'succeeded': call _parse_one_result on it as normal.
      - 'requeue': expired, canceled, or errored with a transient/infra-side
        error type (api_error, overloaded_error, or anything not explicitly
        invalid_request_error — fail safe toward retryable rather than
        silently dropping a future error type this codebase hasn't seen yet).
        No LLM call is spent on these — see _write_jobs_for_round's requeued
        handling for how they get back to pending_enrich.
      - 'stale': errored with invalid_request_error — the API itself
        rejected the request; a differently-worded retry prompt can't fix
        that, so this is genuinely terminal, unlike a parse/validation
        failure on a 'succeeded' result (which still gets one retry).
    """
    result_type = result["type"]
    if result_type == "succeeded":
        return "succeeded"
    if result_type in ("expired", "canceled"):
        return "requeue"
    if result_type == "errored":
        error_type = ((result.get("error") or {}).get("error") or {}).get("type")
        if error_type == "invalid_request_error":
            return "stale"
        return "requeue"
    return "requeue"  # unrecognized future type — fail safe, never silently drop


def _requeue_reason(result: dict) -> str:
    if result["type"] == "errored":
        error_type = ((result.get("error") or {}).get("error") or {}).get("type")
        return f"errored ({error_type})"
    return f"batch result type={result['type']!r}"


def _stale_reason(result: dict) -> str:
    message = ((result.get("error") or {}).get("error") or {}).get("message", "")
    return f"invalid_request_error: {message}"


def _process_round(results: dict) -> dict:
    """Classify+parse every item in one Batches API round's raw *results*
    ({custom_id: raw_dict}). Pure — no I/O, no retry submission (that's the
    caller's decision). Returns {'enriched': {job_id: JobEnrichment},
    'parse_failures': {job_id: error_str} (a 'succeeded' result that failed
    JobEnrichment validation — the ONLY bucket callers may choose to retry),
    'requeued': {job_id: reason_str}, 'stale': {job_id: reason_str},
    'input_tokens': int, 'output_tokens': int}."""
    enriched: dict[str, JobEnrichment] = {}
    parse_failures: dict[str, str] = {}
    requeued: dict[str, str] = {}
    stale: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for job_id, result in results.items():
        disposition = _classify_batch_result(result)
        if disposition == "requeue":
            requeued[job_id] = _requeue_reason(result)
            continue
        if disposition == "stale":
            stale[job_id] = _stale_reason(result)
            continue
        enrichment, usage, error = _parse_one_result(result)
        if usage is not None:
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
        if enrichment is not None:
            enriched[job_id] = enrichment
        else:
            parse_failures[job_id] = error

    return {
        "enriched": enriched, "parse_failures": parse_failures,
        "requeued": requeued, "stale": stale,
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
    }


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
                       requeued: dict[str, str], by_id: dict[str, dict],
                       embeddings: dict[str, list[float]]) -> set[str]:
    """Batched writes to `jobs` — uniform payload per group, same
    NULL-fill-avoidance rule as src/ingest.py: enriched-job updates,
    stale-job updates, and requeued-job updates are separate calls.
    canonical_url/source_board/content_hash are NOT NULL with no default —
    Postgres validates NOT NULL on an upsert's INSERT branch even though
    every one of these rows already exists and only the ON CONFLICT DO
    UPDATE branch actually fires (same quirk already hit and fixed in
    src/ingest.py), so every payload here must carry them through even
    though this step conceptually only touches enrichment fields (confirmed
    the hard way against production).

    A job_id with no by_id entry is logged and skipped, not a KeyError — the
    caller may be src.drain_batches.apply_drained_batch(), which derives
    by_id fresh from `jobs` by id rather than carrying it over from whatever
    run originally submitted the batch, so a job deleted in between (or,
    theoretically, any other by_id gap) is a real, expected possibility here,
    not just defensive paranoia. No job_ids column/schema change needed for
    this — by_id is always freshly queried, never persisted.

    Returns the set of job_ids skipped entirely (across all three buckets) —
    callers use this to report counts reflecting what was actually WRITTEN,
    not just what was classified before the by_id check.
    """
    now = datetime.now(timezone.utc).isoformat()
    skipped: set[str] = set()

    def _row(job_id: str, bucket: str) -> dict | None:
        row = by_id.get(job_id)
        if row is None:
            log.error("_write_enrichment: no job metadata for %s (bucket=%r) — skipping this write", job_id, bucket)
            skipped.add(job_id)
        return row

    enriched_updates = []
    for job_id, enrichment in enriched.items():
        row = _row(job_id, "enriched")
        if row is None:
            continue
        enriched_updates.append({
            "id": job_id,
            "canonical_url": row["canonical_url"],
            "source_board": row["source_board"],
            "content_hash": row["content_hash"],
            "enrichment": enrichment.model_dump(mode="json"),
            "embedding": embeddings.get(job_id),
            "enriched_at": now,
            "status": "enriched",
            # Cleared on every successful enrichment — a job selected via the
            # reenrich_requested_at path (see enrich_jobs()'s selection query)
            # has its request fully satisfied by this write; a job selected via
            # the plain pending_enrich path has this already null, so clearing
            # it unconditionally is a no-op there.
            "reenrich_requested_at": None,
        })

    stale_updates = []
    for job_id in failures:
        row = _row(job_id, "stale")
        if row is None:
            continue
        stale_updates.append({
            "id": job_id,
            "canonical_url": row["canonical_url"],
            "source_board": row["source_board"],
            "content_hash": row["content_hash"],
            "status": "stale",
        })

    requeued_updates = []
    for job_id in requeued:
        row = _row(job_id, "requeued")
        if row is None:
            continue
        # status/enrichment deliberately NOT referenced — the job stays at
        # whatever status it already has (pending_enrich); only enriched_at
        # is bumped, piggybacking on _split_cooldown's EXISTING enriched_at-
        # based deferral so a transient batch failure (expired/canceled/
        # api_error/overloaded_error) doesn't get re-attempted every single
        # run forever. See _split_cooldown's docstring.
        requeued_updates.append({
            "id": job_id,
            "canonical_url": row["canonical_url"],
            "source_board": row["source_board"],
            "content_hash": row["content_hash"],
            "enriched_at": now,
        })

    # chunked_upsert, not one .upsert().execute() for the whole list: confirmed
    # the hard way against production — an unchunked ~1136-row upsert here (each
    # row carrying a jsonb enrichment blob) hit Postgres's statement timeout and
    # lost the entire batch's enrichment (Anthropic spend included).
    if enriched_updates:
        chunked_upsert(client.table("jobs"), enriched_updates, on_conflict="id")
    if stale_updates:
        chunked_upsert(client.table("jobs"), stale_updates, on_conflict="id")
    if requeued_updates:
        chunked_upsert(client.table("jobs"), requeued_updates, on_conflict="id")

    return skipped


def _finalize_round(
    client, *, by_id: dict[str, dict], enriched: dict[str, JobEnrichment],
    stale: dict[str, str], requeued: dict[str, str],
    total_input_tokens: int, total_output_tokens: int, enrichment_calls: int,
    run_id: str | None, trigger: str | None, write_run_metrics: bool,
    batch_id_by_job: dict[str, str] | None = None,
) -> dict:
    """Shared tail: write run_metrics (unless write_run_metrics=False —
    reprocess_staged()'s case, where the normal live path already wrote it
    before the crash being recovered from, so writing it again would
    double-count cost already recorded), then always write `jobs`.

    Shared by enrich_jobs()'s own synchronous round-1(+round-2) flow and
    src.drain_batches.apply_drained_batch() — everything upstream of this
    (how results were classified, whether a retry was attempted) differs
    between those callers; this function only ever sees the FINAL,
    fully-resolved tally for one enrichment attempt.

    batch_id_by_job, if given, maps each job_id in enriched/stale/requeued to the
    batch_id whose raw result ultimately decided its outcome — used to mark that
    (batch_id, job_id) row processed=True in `batch_results` immediately AFTER
    `_write_enrichment` commits (never before — the whole point of `processed`
    is "safely applied", so it must only flip once the write it's gating on has
    actually happened). A job_id `_write_enrichment` itself skipped (no matching
    `jobs` row) is excluded from marking, same as reprocess_staged()'s existing
    missing_ids exclusion — nothing was actually applied for it, so its raw
    result should stay available for a future recovery attempt.

    Returns {"enriched", "stale", "requeued", "embeddings_skipped"} (int/int/int/bool).
    """
    if write_run_metrics:
        # Run the embedding backfill BEFORE counting the gap below, not after
        # — so enriched_without_embedding reflects what's STILL missing once
        # this run has done everything it can about it, not the stale
        # pre-backfill backlog. Independent of (and safe to run regardless
        # of) this round's own enriched/stale/requeued buckets below: it only
        # ever touches OTHER already-enriched jobs still missing an
        # embedding (e.g. ones enriched before VOYAGE_API_KEY existed, or
        # that hit a transient Voyage failure on their own original run) —
        # never the jobs_enriched cost record above, so a crash here can't
        # lose anything this function already safely recorded. Best-effort:
        # backfill_embeddings() never raises (a Voyage outage leaves the
        # backlog count unchanged, logged, not fatal to this run).
        backfill_result = backfill_embeddings(client)
        if backfill_result["embedded"]:
            log.info("Embedding backfill (enrich stage): %s", backfill_result)

        # jobs_enriched here reflects the CLASSIFIED count, not the post-write
        # count returned below — run_metrics is deliberately written before
        # _write_enrichment (crash-safety: a crash in that write must not
        # lose the cost record), so the rare by_id-miss skip isn't yet known
        # at this point. The gap only matters for a job_id with no matching
        # `jobs` row at all, an edge case this count very slightly overstates.
        now = datetime.now(timezone.utc).isoformat()
        # Backlog gauge, not a per-run delta: counts every status='enriched' job
        # still missing an embedding RIGHT NOW — AFTER the backfill attempt
        # just above, so this reflects what's genuinely still missing (e.g.
        # Voyage itself down this run), not a stale pre-backfill count.
        # Surfaces the gap match_v2_top_jobs's embedding IS NOT NULL filter
        # creates silently otherwise.
        enriched_without_embedding = len(fetch_all(
            client.table("jobs").select("id").eq("status", "enriched").is_("embedding", "null")
        ))
        client.table("run_metrics").insert({
            "run_at": now,
            "run_id": run_id,
            "stage": "enrich",
            "trigger": trigger,
            "jobs_enriched": len(enriched),
            "enrichment_calls": enrichment_calls,
            "enriched_without_embedding": enriched_without_embedding,
            "estimated_cost_usd": round(
                total_input_tokens / 1_000_000 * _HAIKU_SYNC_INPUT_USD_PER_MTOK * _BATCH_DISCOUNT
                + total_output_tokens / 1_000_000 * _HAIKU_SYNC_OUTPUT_USD_PER_MTOK * _BATCH_DISCOUNT,
                4,
            ),
        }).execute()

    for job_id, reason in stale.items():
        log.error("Job %s permanently failed enrichment, marking stale: %s", job_id, reason)
    for job_id, reason in requeued.items():
        log.warning("Job %s returned to pending_enrich (no LLM retry spent): %s", job_id, reason)

    embeddings = _embed_enriched(enriched, by_id)
    skipped = _write_enrichment(client, enriched=enriched, failures=stale, requeued=requeued, by_id=by_id, embeddings=embeddings)

    if batch_id_by_job:
        applied_ids = (enriched.keys() | stale.keys() | requeued.keys()) - skipped
        by_batch: dict[str, list[str]] = {}
        for job_id in applied_ids:
            bid = batch_id_by_job.get(job_id)
            if bid is not None:
                by_batch.setdefault(bid, []).append(job_id)
        for bid, job_ids in by_batch.items():
            mark_batch_results_processed(client, bid, job_ids)

    applied_enriched = len(enriched.keys() - skipped)
    applied_stale = len(stale.keys() - skipped)
    applied_requeued = len(requeued.keys() - skipped)

    log.info(
        "Enrichment applied: %d enriched, %d stale, %d requeued, embeddings_skipped=%s",
        applied_enriched, applied_stale, applied_requeued, not embeddings_available(),
    )
    return {
        "enriched": applied_enriched, "stale": applied_stale, "requeued": applied_requeued,
        "embeddings_skipped": not embeddings_available(),
    }


def apply_drained_batch(client, results: dict, *, batch_id: str | None = None, run_id: str | None = None, trigger: str | None = None) -> dict:
    """Apply an already-ended batch's *results* ({custom_id: raw_dict}) with
    NO retry attempt — used by src/drain_batches.py, which must never submit
    new work (see that module's docstring: no ingest, no matching, no digest
    — draining only ever finishes applying work someone else already decided
    to submit). A parse/validation failure encountered here goes straight to
    'stale' — the same terminal outcome an enrich_jobs()-driven round-2 retry
    would likely reach anyway if that retry also failed; draining just
    doesn't spend a second LLM call finding that out.

    Derives job metadata fresh from `jobs` by id (results.keys() ARE the job
    ids — custom_id = job_id, see _batch_request) rather than requiring a
    caller-supplied by_id, mirroring reprocess_staged()'s existing pattern —
    this is what lets this function apply a batch whose submitting run's own
    in-memory state (its pending-jobs query result) is long gone.

    batch_id, if given, marks every applied job_id's batch_results row
    processed=True once this call's write commits (see _finalize_round's
    batch_id_by_job) — optional only so existing callers/tests that don't care
    about that bookkeeping aren't forced to supply it.
    """
    job_ids = list(results.keys())
    # One batch can cover the whole pending_enrich backlog — potentially
    # thousands of ids — so this MUST chunk its .in_() (see
    # fetch_all_in/_IN_FILTER_CHUNK_SIZE; confirmed for real that an
    # unchunked equivalent in src.closure.close_jobs() blew up a GET URL
    # into a Cloudflare 520 at 1,665 ids).
    rows = fetch_all_in(client.table("jobs"), "id", job_ids, select_cols=_JOB_SELECT_COLUMNS)
    by_id = {row["id"]: row for row in rows}

    round_result = _process_round(results)
    stale = dict(round_result["stale"])
    for job_id, error in round_result["parse_failures"].items():
        stale[job_id] = f"{error} (parse failure — no retry attempted while draining)"

    batch_id_by_job = {job_id: batch_id for job_id in job_ids} if batch_id else None
    return _finalize_round(
        client, by_id=by_id, enriched=round_result["enriched"], stale=stale,
        requeued=round_result["requeued"],
        total_input_tokens=round_result["input_tokens"], total_output_tokens=round_result["output_tokens"],
        enrichment_calls=len(results), run_id=run_id, trigger=trigger, write_run_metrics=True,
        batch_id_by_job=batch_id_by_job,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _split_cooldown(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Split *rows* into (eligible, deferred_job_ids). A row is deferred when its
    enriched_at is more recent than _RE_ENRICH_COOLDOWN_DAYS ago. enriched_at is
    bumped in two distinct situations, both intentional (see _write_enrichment):
    a real successful enrichment, OR a 'requeued' outcome (expired/canceled/
    transient-errored batch item — never actually enriched, but still worth
    deferring so a persistently-failing job doesn't burn an LLM call every
    single run). Either way: its content_hash changed again since then (status
    flipped back to pending_enrich by src/ingest.py, or was never changed away
    from pending_enrich to begin with) but the cooldown caps actual re-enrich
    attempts to at most once per window. status stays pending_enrich for a
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
    """Enrich every job with status='pending_enrich', PLUS every status='enriched'
    job with reenrich_requested_at set (content changed again after a prior
    successful enrichment — see src/ingest.py's real-content-change branch,
    which sets this instead of flipping status back to pending_enrich). The
    second set is what keeps a re-enriching job matchable in the meantime: it
    stays status='enriched' — and thus in _fetch_enriched_jobs_pool's matching
    pool — for the entire cooldown window, instead of disappearing from
    matching for up to _RE_ENRICH_COOLDOWN_DAYS the way a direct pending_enrich
    flip used to (confirmed in production: 393 jobs stuck cycling through this
    exact loop, 310 of them at precisely 7 cycles — the cooldown period —
    across July-November).

    Both sets go through the SAME _RE_ENRICH_COOLDOWN_DAYS deferral
    (_split_cooldown, keyed on enriched_at either way) — capped at *limit* if
    given (used for the sample run before committing to the full backlog).
    dry_run=True selects and reports counts without submitting any batch or
    writing anything. job_ids, if given, restricts selection to that specific
    set (still gated on the same status/reenrich_requested_at criteria) — used
    to hand-pick a representative sample (e.g. one that deliberately includes
    "Remote" postings) rather than taking arbitrary DB order.

    Returns a summary dict. status='complete': {status, selected, enriched, stale,
    requeued, embeddings_skipped, deferred}. status='pending' (the batch — round 1
    OR round 2 — didn't finish within src.batch_poll's budget; NOT an error, NEVER
    raises): {status, batch_id, request_counts, selected, deferred, ...whatever of
    enriched/stale/requeued/embeddings_skipped round 1 already committed before
    round 2 went pending, if that's the round that stalled}. Callers (src.pipeline_v2.
    run()) must check result["status"] and stop before matching on a 'pending' result
    — enrichment is incomplete, not failed.
    """
    # fetch_all, not a plain .execute(): PostgREST silently truncates a response
    # at db.max_rows (1000) — the pending_enrich backlog can exceed that.
    if job_ids is not None:
        # Caller-supplied sample set — typically small, but chunked via
        # fetch_all_in regardless (see _IN_FILTER_CHUNK_SIZE) rather than
        # trusting every caller to keep it that way.
        pending_rows = fetch_all_in(
            client.table("jobs"), "id", job_ids, select_cols=_JOB_SELECT_COLUMNS,
            filter_fn=lambda q: q.eq("status", "pending_enrich"),
        )
        reenrich_rows_raw = fetch_all_in(
            client.table("jobs"), "id", job_ids, select_cols=_JOB_SELECT_COLUMNS,
            filter_fn=lambda q: q.eq("status", "enriched"),
        )
    else:
        pending_rows = fetch_all(client.table("jobs").select(_JOB_SELECT_COLUMNS).eq("status", "pending_enrich"))
        reenrich_rows_raw = fetch_all(client.table("jobs").select(_JOB_SELECT_COLUMNS).eq("status", "enriched"))
    # reenrich_requested_at IS NOT NULL filtered client-side, not via the
    # query builder — simpler than a raw PostgREST NOT filter string and needs
    # no new capability from the (already paginating) fetch_all/fake-client path.
    reenrich_rows = [row for row in reenrich_rows_raw if row.get("reenrich_requested_at") is not None]
    rows = pending_rows + reenrich_rows
    rows, deferred_ids = _split_cooldown(rows)
    if limit is not None:
        rows = rows[:limit]

    if dry_run:
        log.info("DRY RUN — would enrich %d job(s) (%d deferred by re-enrich cooldown)", len(rows), len(deferred_ids))
        return {
            "status": "complete", "selected": len(rows), "enriched": 0, "stale": 0, "requeued": 0,
            "embeddings_skipped": not embeddings_available(), "deferred": len(deferred_ids),
        }

    if not rows:
        log.info("No pending_enrich jobs found — nothing to do (%d deferred by cooldown)", len(deferred_ids))
        # Still worth running even with nothing new to enrich — this is the
        # one path that otherwise never calls backfill_embeddings() at all,
        # which would leave an existing embedding gap unaddressed on any day
        # the pending_enrich/reenrich_requested backlog happens to be empty.
        backfill_embeddings(client)
        return {
            "status": "complete", "selected": 0, "enriched": 0, "stale": 0, "requeued": 0,
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
    outcome = resume_or_submit(
        anthropic_client, client, module=_MODULE, requests=requests,
        budget_seconds=DEFAULT_POLL_BUDGET_SECONDS,
    )
    if outcome.status == "pending":
        # Nothing was retrieved, so nothing to write to `jobs`. Still write a
        # run_metrics row for the enrich stage on this run_id — previously
        # this row simply never existed on a timeout (confirmed against
        # production: the 2026-09-14 incident this whole mechanism exists
        # for left zero 'enrich' stage rows for its run_id).
        client.table("run_metrics").insert({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "stage": "enrich", "trigger": trigger, "jobs_enriched": 0,
        }).execute()
        log.info(
            "Enrichment pending — batch %s not finished within budget (elapsed %.0fs, "
            "counts=%s) — resuming next run.",
            outcome.batch_id, outcome.elapsed_seconds, outcome.request_counts,
        )
        return {
            "status": "pending", "batch_id": outcome.batch_id, "request_counts": outcome.request_counts,
            "selected": len(rows), "deferred": len(deferred_ids),
        }

    round1 = _process_round(outcome.results)
    enriched = round1["enriched"]
    requeued = round1["requeued"]
    stale = round1["stale"]
    total_input_tokens = round1["input_tokens"]
    total_output_tokens = round1["output_tokens"]
    enrichment_calls = len(requests)
    retry_pending = None
    # Maps each job_id to whichever batch ultimately decided its outcome — round
    # 2's batch_id overwrites round 1's below for any job_id that got retried,
    # since round 2's result (not round 1's parse failure) is what's actually
    # applied. Used only to mark batch_results.processed=True after the write
    # below commits — see _finalize_round's batch_id_by_job param.
    batch_id_by_job: dict[str, str] = {job_id: outcome.batch_id for job_id in outcome.results}

    # Round 2 — one retry for parse/validation failures only, with the error
    # appended. Requeued/stale items from round 1 are NOT retried (see this
    # module's docstring on per-request terminal-state classification).
    if round1["parse_failures"]:
        log.warning(
            "%d job(s) failed round 1, retrying once: %s",
            len(round1["parse_failures"]), list(round1["parse_failures"].keys()),
        )
        retry_requests = [
            _batch_request(job_id, build_retry_prompt(
                title=by_id[job_id]["title"], org=by_id[job_id]["org"],
                location_field=by_id[job_id].get("location_field"),
                description=by_id[job_id]["raw_description"] or "",
                validation_error=error,
            ))
            for job_id, error in round1["parse_failures"].items()
        ]
        enrichment_calls += len(retry_requests)
        retry_outcome = resume_or_submit(
            anthropic_client, client, module=f"{_MODULE}_retry", requests=retry_requests,
            budget_seconds=DEFAULT_POLL_BUDGET_SECONDS,
        )
        if retry_outcome.status == "pending":
            retry_pending = retry_outcome
        else:
            round2 = _process_round(retry_outcome.results)
            enriched.update(round2["enriched"])
            requeued.update(round2["requeued"])
            stale.update(round2["stale"])
            for job_id, error in round2["parse_failures"].items():
                stale[job_id] = f"{error} (retry)"
            total_input_tokens += round2["input_tokens"]
            total_output_tokens += round2["output_tokens"]
            batch_id_by_job.update({job_id: retry_outcome.batch_id for job_id in retry_outcome.results})

    # run_metrics written NOW — right after usage is tallied, BEFORE the final
    # `jobs` write below — so a crash in that write no longer loses the cost
    # record (confirmed the hard way: this used to be written last, and a
    # crashed write meant ~$2.65 of paid Anthropic output went unrecorded).
    # Round 1's results are committed here even if round 2 itself went
    # pending — never lose already-resolved output waiting on a slower part.
    summary = _finalize_round(
        client, by_id=by_id, enriched=enriched, stale=stale, requeued=requeued,
        total_input_tokens=total_input_tokens, total_output_tokens=total_output_tokens,
        enrichment_calls=enrichment_calls, run_id=run_id, trigger=trigger, write_run_metrics=True,
        batch_id_by_job=batch_id_by_job,
    )

    if retry_pending is not None:
        log.info(
            "Enrichment partially applied (round 1 committed) — retry batch %s not "
            "finished within budget (elapsed %.0fs, counts=%s) — resuming next run.",
            retry_pending.batch_id, retry_pending.elapsed_seconds, retry_pending.request_counts,
        )
        return {
            "status": "pending", "batch_id": retry_pending.batch_id,
            "request_counts": retry_pending.request_counts,
            "selected": len(rows), "deferred": len(deferred_ids), **summary,
        }

    log.info(
        "Enrichment complete: %d selected, %d deferred by cooldown, %s",
        len(rows), len(deferred_ids), summary,
    )
    return {"status": "complete", "selected": len(rows), "deferred": len(deferred_ids), **summary}


def reprocess_staged(client, batch_id: str) -> dict:
    """Recovery path: reprocess unprocessed `batch_results` rows for *batch_id*
    WITHOUT calling the Anthropic API again — the raw results are already
    staged from a prior run's poll that didn't finish (crashed somewhere
    after retrieval, e.g. during the final `jobs` write). Re-derives each
    job's original metadata (title/org/canonical_url/source_board/
    content_hash) from `jobs` by id, re-runs the SAME classification
    (_classify_batch_result/_process_round) and parsing logic the live path
    uses, recomputes embeddings for anything that parses successfully (the
    original run's embeddings, if any were computed, were only ever
    in-memory and are lost along with everything else that crashed), and
    redoes the final `jobs` write. Does NOT retry a second time on parse
    failure (that would need a new LLM call, defeating "without re-calling
    the API") — anything that still fails to parse here is marked stale
    directly, same as a requeue/stale-classified item. Does NOT write
    run_metrics — the normal path already wrote it (right after usage
    tallying, before the write that crashed), so writing it again here would
    double-count cost that was already recorded.

    Returns {"selected", "enriched", "stale"}.
    """
    staged = fetch_all(
        client.table("batch_results").select("custom_id,raw").eq("batch_id", batch_id).eq("processed", False)
    )
    if not staged:
        log.info("reprocess_staged: no unprocessed batch_results for batch %s", batch_id)
        return {"selected": 0, "enriched": 0, "stale": 0}

    results = {row["custom_id"]: row["raw"] for row in staged}
    raw_by_job_id = results

    # Same unbounded-backlog concern as apply_drained_batch() above — chunk
    # via fetch_all_in, never a single .in_() with every staged id.
    rows = fetch_all_in(client.table("jobs"), "id", list(results.keys()), select_cols=_JOB_SELECT_COLUMNS)
    by_id = {row["id"]: row for row in rows}

    round_result = _process_round(results)
    stale = dict(round_result["stale"])
    for job_id, error in round_result["parse_failures"].items():
        stale[job_id] = f"{error} (parse failure during reprocess — no retry)"

    # Computed BEFORE _finalize_round (which will itself log+skip these at
    # the jobs-write step), purely so processed_ids below can exclude them:
    # nothing more can be done for a job that no longer exists, and leaving
    # its batch_results row unprocessed (rather than silently discarding the
    # evidence) matches this function's pre-existing behavior for that case.
    all_ids = round_result["enriched"].keys() | stale.keys() | round_result["requeued"].keys()
    missing_ids = all_ids - by_id.keys()
    for job_id in missing_ids:
        log.error("reprocess_staged: staged result for job %s but that job no longer exists — skipping", job_id)

    summary = _finalize_round(
        client, by_id=by_id, enriched=round_result["enriched"], stale=stale,
        requeued=round_result["requeued"], total_input_tokens=0, total_output_tokens=0,
        enrichment_calls=0, run_id=None, trigger=None, write_run_metrics=False,
    )

    # raw is NOT NULL with no default — same upsert-INSERT-branch quirk as
    # `jobs`' canonical_url/source_board/content_hash, so it must be carried
    # through here too even though this write only means to touch `processed`.
    processed_ids = list(all_ids - missing_ids)
    if processed_ids:
        chunked_upsert(
            client.table("batch_results"),
            [{"batch_id": batch_id, "custom_id": jid, "raw": raw_by_job_id[jid], "processed": True} for jid in processed_ids],
            on_conflict="batch_id,custom_id",
        )

    log.info(
        "reprocess_staged: batch %s -> %d selected, %d enriched, %d stale, %d requeued",
        batch_id, len(staged), summary["enriched"], summary["stale"], summary["requeued"],
    )
    return {"selected": len(staged), "enriched": summary["enriched"], "stale": summary["stale"]}


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
