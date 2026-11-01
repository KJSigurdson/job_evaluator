"""True North v2 Tier 2 enrichment: one Sonnet call per (user, job) pair whose
Tier 1 rerank score clears a threshold (default 65), via the Anthropic Batches
API. Reuses src.schemas.Tier2EnrichmentOutput — v1's exact output shape
(org_summary/why_fits/why_not_fits/emphasize_in_cv/deemphasize) — and mirrors
v1's src/enrich.py prompt structure/format instructions closely, since the
frontend already renders that exact bullet/phrase format; this is a new call
site producing the same shape, not a new format.

Phase 2: candidates now come from fetch_candidates() reading tier1_verdicts
directly (score >= threshold, valid, no existing matches_v2 row) — not from
an in-memory dict a same-run Tier 1 call handed down. This makes Tier 2
resumable from whatever Tier 1 has written so far, INCLUDING verdicts the
external hourly drainer applied between runs (src.tier1_rerank.
apply_drained_batch writes to the exact same tier1_verdicts table) — Tier 2
has no way to tell a "live this-run" verdict from a "drained from
yesterday" one, by design, since both are equally valid.

Shadow mode: writes to the new `matches_v2` table only (user_id, job_id,
tier1_score, tier2 jsonb, created_at) — NOT v1's `matches` table. Also writes
run_metrics (tier2_calls, estimated_cost_usd). No v1 table touched.

Non-blocking via src.batch_poll.resume_or_submit, same as src/enrich_jobs.py/
src/tier1_rerank.py — exhausting the poll budget returns status='pending',
NEVER raises.

Crash-safety note specific to this module: custom_id is an anonymous
enumeration index (see tier2_enrich()'s by_key), not user_id/job_id — the
Batches API caps custom_id at 64 characters and two joined UUIDs is 73. That
means, unlike enrich_jobs.py (custom_id=job_id) and tier1_rerank.py
(custom_id=f"{user_id}:{chunk_index}"), custom_id alone isn't enough to
recover which (user, job) pair a staged raw result belongs to. So each
result is staged wrapped as {"llm_result": ..., "context": {...}} via
batch_poll.resume_or_submit's persist_transform hook — the context needed to
rebuild the matches_v2 row and re-run Tier 2's own parser without any
re-fetch. See reprocess_staged().
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import yaml
from pydantic import ValidationError

from src.batch_poll import DEFAULT_POLL_BUDGET_SECONDS, resume_or_submit
from src.match_v2 import fetch_matched_job_ids
from src.schemas import Tier2EnrichmentOutput
from src.supabase_client import chunked_upsert, fetch_all, fetch_all_in, mark_batch_results_processed

log = logging.getLogger(__name__)

_MODULE = "tier2_enrich"
_MODEL = "claude-sonnet-4-6"
_TOOL_NAME = "enrich_posting"
_MAX_TOKENS = 2048
_DEFAULT_THRESHOLD = 65  # fallback only — pipeline_v2.py always passes each user's own
                         # scoring_weights.insert_threshold (scaled to 0-100) explicitly
_LIST_FIELDS = ("emphasize_in_cv", "deemphasize")
_JOB_SELECT_COLUMNS = "id,title,org,raw_description,enrichment,enriched_at"

# Sonnet is priced well above Haiku — VERIFY AGAINST https://www.anthropic.com/pricing
# BEFORE TRUSTING estimated_cost_usd, same caveat as enrich_jobs.py/tier1_rerank.py.
_SONNET_SYNC_INPUT_USD_PER_MTOK = 3.00
_SONNET_SYNC_OUTPUT_USD_PER_MTOK = 15.00
_BATCH_DISCOUNT = 0.5


# ---------------------------------------------------------------------------
# Prompt — mirrors src/enrich.py's _call_llm prompt structure/format
# instructions closely (same FORMAT rules, same "ground in experience_inventory"
# instruction); JOB POSTING and the score section are adapted to v2's shape
# (a jobs row + enrichment jsonb, and a single 0-100 Tier 1 score instead of
# v1's 7-dimension breakdown).
# ---------------------------------------------------------------------------

def _job_text(job: dict) -> str:
    enrichment = job.get("enrichment") or {}
    return (
        f"Title: {job.get('title')}\n"
        f"Organization: {job.get('org')}\n"
        f"Summary: {enrichment.get('summary')}\n"
        f"Seniority: {enrichment.get('seniority')}\n"
        f"Cause area: {enrichment.get('cause_area')}\n"
        f"Skill tags: {enrichment.get('skill_tags')}\n"
        f"Work arrangement: {enrichment.get('work_arrangement')}\n\n"
        f"Full description:\n{job.get('raw_description') or '(not available)'}"
    )


def build_prompt(*, profile: dict, job: dict, tier1_score: int, tier1_reason: str) -> str:
    """Pure/testable — no API call. profile: the UserContext.profile dict
    (full, untrimmed — same as v1's Tier 2 prompt, which doesn't strip any
    profile fields). job: a jobs row dict with at least title/org/
    raw_description/enrichment."""
    profile_text = yaml.dump(profile, allow_unicode=True, default_flow_style=False)

    return (
        "Generate enrichment content for a high-fit job posting.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"JOB POSTING:\n{_job_text(job)}\n\n"
        f"TIER 1 SCORE: {tier1_score}/100 — {tier1_reason}\n\n"
        "Ground why_fits and emphasize_in_cv in specific items from the candidate's "
        "experience_inventory. Be concise and role-specific.\n\n"
        "FORMAT — why_fits and why_not_fits: each is a single string containing a "
        "short bulleted list, not a paragraph. Every line starts with \"- \" and "
        "lines are separated by newlines within the string. Plain bullets only — "
        "no bold text, no markdown headers, no ** formatting. 3-6 bullets per "
        "field. Each bullet covers exactly one distinct claim (e.g. mission fit, "
        "technical fit, seniority/stakeholder fit, location) as its own line — do "
        "not compress multiple claims into one bullet, and do not write "
        "multi-sentence bullets.\n\n"
        "FORMAT — emphasize_in_cv and deemphasize: each list item is a short "
        "phrase (roughly 3-8 words) naming a specific experience or skill, not a "
        "sentence explaining why it matters. Example: \"IVC Evidensia "
        "volume-growth modelling\", not \"Emphasize the volume growth program you "
        "built at IVC Evidensia because it demonstrates quantitative modelling "
        "skills relevant to the role.\""
    )


def build_retry_prompt(*, profile: dict, job: dict, tier1_score: int, tier1_reason: str, validation_error: str) -> str:
    base = build_prompt(profile=profile, job=job, tier1_score=tier1_score, tier1_reason=tier1_reason)
    return (
        f"{base}\n\n"
        "NOTE: A previous attempt at this task failed validation with the "
        f"following error — correct it in this attempt:\n{validation_error}"
    )


# ---------------------------------------------------------------------------
# Response parsing — pure, unit-testable without hitting the Anthropic API.
# Operates on plain dicts (result.model_dump(mode="json") — see _run_batch()),
# not Anthropic SDK objects, so the live path and reprocess_staged()'s replay
# path share identical parsing code with no dual-shape handling.
# ---------------------------------------------------------------------------

def _coerce_list_field(value):
    """Same defensive coercion as src/enrich.py's identical helper: a model can
    return emphasize_in_cv/deemphasize as a JSON-array-in-a-string instead of a
    native array — parse it back rather than burn the retry on a formatting
    quirk. Duplicated, not imported, per this codebase's established
    no-shared-utils convention for LLM-call plumbing."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    return parsed if isinstance(parsed, list) else value


def _merge_tool_use_blocks(message: dict, tool_name: str) -> dict:
    merged: dict = {}
    found = False
    for block in message["content"]:
        if block["type"] == "tool_use" and block["name"] == tool_name:
            merged.update(block["input"])
            found = True
    if not found:
        raise IndexError(f"No tool_use block named {tool_name!r} found in message content")
    return merged


def _parse_tier2(message: dict) -> Tier2EnrichmentOutput:
    tool_input = _merge_tool_use_blocks(message, _TOOL_NAME)
    normalized = dict(tool_input)
    for field in _LIST_FIELDS:
        if field in normalized:
            normalized[field] = _coerce_list_field(normalized[field])
    return Tier2EnrichmentOutput.model_validate(normalized)


def _parse_one_result(result: dict) -> tuple[Tier2EnrichmentOutput | None, dict | None, str | None]:
    """Given one raw batch result dict, return (output, usage, error) — shared
    by the live round-1/round-2 loops and reprocess_staged()."""
    if result["type"] != "succeeded":
        return None, None, f"batch result type={result['type']!r}"
    usage = result["message"]["usage"]
    try:
        return _parse_tier2(result["message"]), usage, None
    except (IndexError, AttributeError, ValidationError) as exc:
        return None, usage, str(exc)


_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Generate enrichment content for a high-fit job posting.",
    "input_schema": Tier2EnrichmentOutput.model_json_schema(),
}


# ---------------------------------------------------------------------------
# Batches API plumbing — duplicated from enrich_jobs.py's/tier1_rerank.py's
# identical helpers, per this codebase's established no-shared-utils convention.
# ---------------------------------------------------------------------------

def _batch_request(custom_id: str, prompt: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _run_batch(
    anthropic_client, requests: list[dict], *, client, module: str = _MODULE,
    context_by_custom_id: dict[str, dict] | None = None,
):
    """Submit *requests*, or resume an outstanding batch for *module* if a
    prior run left one, via src.batch_poll.resume_or_submit — non-blocking:
    exhausting the poll budget returns a BatchOutcome with status='pending',
    NEVER raises. Callers must check outcome.status themselves (same
    pattern as src/enrich_jobs.py).

    context_by_custom_id, if given, is staged ALONGSIDE each raw result as
    {"llm_result": ..., "context": ...} instead of the bare result (via
    resume_or_submit's persist_transform hook) — this module's custom_id is
    an anonymous index (see module docstring), so recovery needs this
    context to know which (user, job) pair a staged result belongs to;
    enrich_jobs.py/tier1_rerank.py don't need this since their custom_id
    already IS the job_id/f"{user_id}:{chunk}".
    """
    persist_transform = None
    if context_by_custom_id is not None:
        def persist_transform(results):
            return {
                cid: {"llm_result": raw, "context": context_by_custom_id.get(cid)}
                for cid, raw in results.items()
            }

    return resume_or_submit(
        anthropic_client, client, module=module, requests=requests,
        budget_seconds=DEFAULT_POLL_BUDGET_SECONDS, persist_transform=persist_transform,
    )


# ---------------------------------------------------------------------------
# Candidate derivation — reads tier1_verdicts directly (Phase 2), not an
# in-memory dict a same-run Tier 1 call handed down.
# ---------------------------------------------------------------------------

def fetch_candidates(
    client, *, user_id: str, profile: dict, profile_content_hash: str | None, threshold: float,
) -> list[dict]:
    """Tier 2 candidates for *user_id*: tier1_verdicts rows scoring >=
    *threshold*, VALID (job_enriched_at/profile_content_hash still match the
    job's/this user's CURRENT state — same validity check as
    src.tier1_rerank.fetch_valid_verdict_job_ids), with no existing
    matches_v2 row. This is what makes Tier 2 resumable from whatever Tier 1
    has written so far, INCLUDING verdicts src.tier1_rerank.
    apply_drained_batch applied from a drained batch between runs — Tier 2
    has no way to tell a "scored this run" verdict from a "drained from
    yesterday's stuck batch" one, by design, since both are equally valid.

    A verdict (or the caller's *profile_content_hash*) with a null hash is
    always treated as stale/invalid, never as a match — see the loop below.

    Returns [{"user_id", "profile", "job", "tier1_score", "tier1_reason"}, ...]
    — tier2_enrich()'s own expected candidate shape.
    """
    verdicts = fetch_all(
        client.table("tier1_verdicts").select("*").eq("user_id", user_id).gte("score", threshold)
    )
    if not verdicts:
        return []

    matched_ids = fetch_matched_job_ids(client, user_id)
    verdicts = [v for v in verdicts if v["job_id"] not in matched_ids]
    if not verdicts:
        return []

    # >= threshold verdicts for one user, but that can still grow into the
    # hundreds over time — chunk the .in_() rather than fetch all job_ids in
    # one request (see fetch_all_in/_IN_FILTER_CHUNK_SIZE).
    jobs_rows = fetch_all_in(client.table("jobs"), "id", [v["job_id"] for v in verdicts], select_cols=_JOB_SELECT_COLUMNS)
    jobs_by_id = {j["id"]: j for j in jobs_rows}

    candidates = []
    for v in verdicts:
        job = jobs_by_id.get(v["job_id"])
        if job is None:
            continue  # job deleted since scoring
        if job.get("enriched_at") != v.get("job_enriched_at"):
            continue  # stale verdict — job re-enriched since scoring
        verdict_hash = v.get("profile_content_hash")
        # profile_content_hash is nullable on both sides (profiles.
        # profile_content_hash and tier1_verdicts.profile_content_hash) — a
        # null must never compare equal to a null (None != None is False in
        # Python), or "we don't know this user's hash" would incorrectly
        # validate against "the verdict's hash is also unknown". See
        # src.tier1_rerank.fetch_valid_verdict_job_ids's identical check.
        if profile_content_hash is None or verdict_hash is None:
            log.warning(
                "fetch_candidates: null profile_content_hash for user %s / job %s "
                "(current=%r, verdict=%r) — treating as stale, not a match",
                user_id, v["job_id"], profile_content_hash, verdict_hash,
            )
            continue
        if verdict_hash != profile_content_hash:
            continue  # stale verdict — profile changed since scoring
        candidates.append({
            "user_id": user_id, "profile": profile, "job": job,
            "tier1_score": v["score"], "tier1_reason": (v.get("rationale") or {}).get("reason", ""),
        })
    return candidates


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _candidate_context(key: str, c: dict) -> dict:
    return {
        "user_id": c["user_id"], "job": c["job"], "tier1_score": c["tier1_score"],
        "tier1_reason": c["tier1_reason"], "profile": c["profile"],
    }


def tier2_enrich(
    client, anthropic_client, *, candidates: list[dict], threshold: float = _DEFAULT_THRESHOLD,
    run_id: str | None = None, trigger: str | None = None,
) -> dict:
    """candidates: [{"user_id", "profile", "job", "tier1_score", "tier1_reason"}, ...]
    — one entry per (user, job) pair surviving the Tier 1 rerank. Only entries
    with tier1_score >= threshold get a Tier 2 call; the rest are skipped
    (not written anywhere, not an error — most candidates are expected to be
    below the bar).

    One retry for any (user, job) pair whose output fails validation, error
    appended to a second, smaller batch — same pattern as enrich_jobs.py/
    tier1_rerank.py. A pair that fails both rounds is logged and dropped —
    never written partially.

    run_metrics is written right after usage is tallied (round 1 + round 2, if
    any) — BEFORE the final `matches_v2` write — so a crash in that write no
    longer loses the cost record.

    Returns {"considered", "enriched", "skipped_below_threshold", "failed"}.
    """
    qualifying = [c for c in candidates if c["tier1_score"] >= threshold]
    skipped = len(candidates) - len(qualifying)

    if not qualifying:
        log.info(
            "tier2_enrich: 0/%d candidate(s) clear threshold %s — nothing to do",
            len(candidates), threshold,
        )
        return {"considered": len(candidates), "enriched": 0, "skipped_below_threshold": skipped, "failed": 0}

    # custom_id is capped at 64 characters by the Batches API — "{user_id}:{job_id}"
    # (two UUIDs) is 73 and gets rejected outright (confirmed for real: "String
    # should have at most 64 characters"). A plain enumeration index is short,
    # unique within this call — see _run_batch()'s context_by_custom_id for how
    # recovery still knows which (user, job) pair each index corresponds to.
    by_key = {str(i): c for i, c in enumerate(qualifying)}
    context_by_key = {key: _candidate_context(key, c) for key, c in by_key.items()}

    requests = [
        _batch_request(key, build_prompt(
            profile=c["profile"], job=c["job"], tier1_score=c["tier1_score"], tier1_reason=c["tier1_reason"],
        ))
        for key, c in by_key.items()
    ]
    outcome = _run_batch(anthropic_client, requests, client=client, context_by_custom_id=context_by_key)
    if outcome.status == "pending":
        # Nothing was retrieved — mirrors src.enrich_jobs.enrich_jobs()'s
        # identical reasoning: still write a run_metrics row for the tier2
        # stage so this run_id has SOME record, rather than a silent gap.
        client.table("run_metrics").insert({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "stage": "tier2", "trigger": trigger, "tier2_calls": 0,
        }).execute()
        log.info(
            "tier2_enrich pending — batch %s not finished within budget (elapsed %.0fs, "
            "counts=%s) — resuming next run.",
            outcome.batch_id, outcome.elapsed_seconds, outcome.request_counts,
        )
        return {
            "status": "pending", "batch_id": outcome.batch_id, "request_counts": outcome.request_counts,
            "considered": len(candidates), "enriched": 0, "skipped_below_threshold": skipped, "failed": 0,
        }

    results, batch_id = outcome.results, outcome.batch_id
    total_api_calls = len(requests)
    # Maps each key to whichever batch's raw result it came from — marked
    # processed=True only AFTER the matches_v2 write below commits (this
    # module, unlike tier1_rerank.py, has a real write to gate the timing on).
    batch_id_by_key: dict[str, str] = {key: batch_id for key in results}

    enriched: dict[str, Tier2EnrichmentOutput] = {}
    failures: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    retry_pending = None

    for key, result in results.items():
        output, usage, error = _parse_one_result(result)
        if usage is not None:
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
        if output is not None:
            enriched[key] = output
        else:
            failures[key] = error

    if failures:
        log.warning("%d pair(s) failed round 1, retrying once: %s", len(failures), list(failures.keys()))
        retry_requests = [
            _batch_request(key, build_retry_prompt(
                profile=by_key[key]["profile"], job=by_key[key]["job"],
                tier1_score=by_key[key]["tier1_score"], tier1_reason=by_key[key]["tier1_reason"],
                validation_error=error,
            ))
            for key, error in failures.items()
        ]
        retry_context = {key: context_by_key[key] for key in failures}
        retry_outcome = _run_batch(
            anthropic_client, retry_requests, client=client, module=f"{_MODULE}_retry",
            context_by_custom_id=retry_context,
        )
        total_api_calls += len(retry_requests)

        if retry_outcome.status == "pending":
            retry_pending = retry_outcome
            failures = {}  # not yet known — round 2 hasn't resolved
        else:
            retry_results, retry_batch_id = retry_outcome.results, retry_outcome.batch_id
            batch_id_by_key.update({key: retry_batch_id for key in retry_results})

            still_failing: dict[str, str] = {}
            for key, result in retry_results.items():
                output, usage, error = _parse_one_result(result)
                if usage is not None:
                    total_input_tokens += usage["input_tokens"]
                    total_output_tokens += usage["output_tokens"]
                if output is not None:
                    enriched[key] = output
                else:
                    still_failing[key] = f"{error} (retry)"

            failures = still_failing

    for key, reason in failures.items():
        log.error("Pair %s permanently failed Tier 2 enrichment: %s", key, reason)

    # run_metrics written NOW — right after usage is tallied, BEFORE the final
    # `matches_v2` write below — so a crash in that write no longer loses the
    # cost record (same fix as enrich_jobs.py, same real incident class).
    # Round 1's results are committed here even if round 2 itself went
    # pending — never lose already-resolved output waiting on a slower part.
    now = datetime.now(timezone.utc).isoformat()
    client.table("run_metrics").insert({
        "run_at": now,
        "run_id": run_id,
        "stage": "tier2",
        "trigger": trigger,
        "tier2_calls": total_api_calls,
        "estimated_cost_usd": round(
            total_input_tokens / 1_000_000 * _SONNET_SYNC_INPUT_USD_PER_MTOK * _BATCH_DISCOUNT
            + total_output_tokens / 1_000_000 * _SONNET_SYNC_OUTPUT_USD_PER_MTOK * _BATCH_DISCOUNT,
            4,
        ),
    }).execute()

    updates = [
        {
            "user_id": by_key[key]["user_id"],
            "job_id": by_key[key]["job"]["id"],
            "tier1_score": by_key[key]["tier1_score"],
            "tier2": output.model_dump(mode="json"),
            "created_at": now,
        }
        for key, output in enriched.items()
    ]
    if updates:
        chunked_upsert(client.table("matches_v2"), updates, on_conflict="user_id,job_id")

    # Marked processed=True only now, AFTER the matches_v2 write above commits
    # — a crash before this point leaves these rows unprocessed and recoverable
    # via reprocess_staged(), same crash-safety principle as enrich_jobs.py.
    # Every resolved key (enriched or permanently failed) is marked, same
    # "resolved, not just succeeded" policy reprocess_staged() already uses.
    by_batch: dict[str, list[str]] = {}
    for key in (enriched.keys() | failures.keys()):
        bid = batch_id_by_key.get(key)
        if bid is not None:
            by_batch.setdefault(bid, []).append(key)
    for bid, keys in by_batch.items():
        mark_batch_results_processed(client, bid, keys)

    if retry_pending is not None:
        log.info(
            "tier2_enrich: round 1 committed (%d enriched) — retry batch %s not finished "
            "within budget (elapsed %.0fs, counts=%s) — resuming next run.",
            len(enriched), retry_pending.batch_id, retry_pending.elapsed_seconds, retry_pending.request_counts,
        )
        return {
            "status": "pending", "batch_id": retry_pending.batch_id,
            "request_counts": retry_pending.request_counts,
            "considered": len(candidates), "enriched": len(enriched),
            "skipped_below_threshold": skipped, "failed": 0,
        }

    log.info(
        "tier2_enrich complete: %d considered, %d skipped (below threshold), %d enriched, %d failed",
        len(candidates), skipped, len(enriched), len(failures),
    )
    return {
        "considered": len(candidates), "enriched": len(enriched),
        "skipped_below_threshold": skipped, "failed": len(failures),
    }


def reprocess_staged(client, batch_id: str) -> dict:
    """Recovery path: reprocess unprocessed `batch_results` rows for *batch_id*
    WITHOUT calling the Anthropic API again. Each staged row's `raw` is
    {"llm_result": ..., "context": {"user_id", "job", "tier1_score",
    "tier1_reason", "profile"}} (see _run_batch()'s context_by_custom_id) —
    fully self-sufficient, no re-fetch needed, since this module's custom_id
    alone can't identify the (user, job) pair. Does NOT retry a second time on
    parse failure (would need a new LLM call) — anything that still fails to
    parse here is dropped, logged, not written. Does NOT write run_metrics —
    the normal path already wrote it before whatever crashed necessitated
    recovery; re-inserting here would double-count cost.

    Returns {"selected", "enriched", "failed"}.
    """
    staged = fetch_all(
        client.table("batch_results").select("custom_id,raw").eq("batch_id", batch_id).eq("processed", False)
    )
    if not staged:
        log.info("reprocess_staged: no unprocessed batch_results for batch %s", batch_id)
        return {"selected": 0, "enriched": 0, "failed": 0}

    now = datetime.now(timezone.utc).isoformat()
    updates = []
    failed_count = 0
    processed_rows = []

    for row in staged:
        custom_id = row["custom_id"]
        wrapped = row["raw"]
        context = wrapped.get("context")
        if not context:
            log.error("reprocess_staged: staged result %s has no context — cannot recover, skipping", custom_id)
            processed_rows.append({"batch_id": batch_id, "custom_id": custom_id, "raw": wrapped, "processed": True})
            failed_count += 1
            continue

        output, _usage, error = _parse_one_result(wrapped["llm_result"])
        if output is not None:
            updates.append({
                "user_id": context["user_id"],
                "job_id": context["job"]["id"],
                "tier1_score": context["tier1_score"],
                "tier2": output.model_dump(mode="json"),
                "created_at": now,
            })
        else:
            log.error("reprocess_staged: pair %s permanently failed Tier 2 enrichment: %s", custom_id, error)
            failed_count += 1

        processed_rows.append({"batch_id": batch_id, "custom_id": custom_id, "raw": wrapped, "processed": True})

    if updates:
        chunked_upsert(client.table("matches_v2"), updates, on_conflict="user_id,job_id")
    if processed_rows:
        chunked_upsert(client.table("batch_results"), processed_rows, on_conflict="batch_id,custom_id")

    log.info(
        "reprocess_staged: batch %s -> %d staged, %d enriched, %d failed",
        batch_id, len(staged), len(updates), failed_count,
    )
    return {"selected": len(staged), "enriched": len(updates), "failed": failed_count}
