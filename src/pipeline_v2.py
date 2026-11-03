"""True North v2 production orchestrator — replaces v1's src/pipeline.py as the
active daily matching + digest-email path (see .github/workflows/daily.yml: v1's
step is now if: false; v1's code and tables stay completely untouched).

Usage:
    python -m src.pipeline_v2 [--dry-run] [--only-user-id ID]

No --only-user-id (scheduled/cron mode) — the full pipeline:
    1. Active adapters (see _ADAPTER_REGISTRY — currently 80k, ProbablyGood;
       IAP disabled) -> ingest.ingest() (dedup/alias into `jobs`). Each
       adapter's fetch-completeness is logged to `source_fetch_log`
       (src.adapters.base.FetchResult) regardless of success/failure.
    1b. closure.close_jobs() — status='enriched'/'pending_enrich' jobs not
       seen in a while are marked 'closed' (see that module for the two
       rules); rule (a) only applies to a source whose fetch THIS RUN (step
       1, just above) was complete=true — derived from fetch_log_rows
       directly, not a historical source_fetch_log query, so "this run"
       means exactly that. ingest.ingest() (step 1) already reopens a closed
       job the moment it's seen again, so running closure right after
       ingest, before enrich, never races a job that's actually still live
       and was just re-ingested this same run.
    2. enrich_jobs.enrich_jobs() — status='pending_enrich' rows (new or
       content-changed with no enrichment yet) PLUS status='enriched' rows
       with reenrich_requested_at set (content changed again after a prior
       successful enrichment — see ingest.ingest()'s real-change branch);
       steady-state stays small by design (the shared jobs pool persists
       across runs, unlike v1's fresh-scrape-every-day model).
    3. profile_embed.backfill_profile_embeddings() — only hash-changed profiles.
    4. Per active user with a complete profile: gate_v2 -> vector top-N
       (src.match_v2.TOP_N) -> Tier 1 -> Tier 2 -> matches_v2, excluding jobs
       already matched to that user (src.match_v2.fetch_matched_job_ids) from
       the candidate pool BEFORE gating — without this, a user's existing
       strong matches would keep winning top-N vector-rerank slots forever,
       starving new jobs from ever reaching Tier 1, on top of re-spending
       Tier 2 and re-emailing the same jobs every run. Only candidates with
       tier1_score >= that user's own scoring_weights.insert_threshold
       (0-1 in storage, scaled to tier1_score's 0-100 range — see
       _match_and_notify_user) ever reach Tier 2 or matches_v2; anything below
       is dropped outright — no near-miss tier, no partial-fit row of any kind.
    5. Digest email per user with new matches this run (email_on_match, not
       dry-run) — reuses src.notify.send_match_digest() UNCHANGED, same
       best-effort/never-raise contract as v1; v2's `matches_v2` fields are
       mapped into v1's exact digest payload shape (see _build_digest_payload)
       since the Edge Function on the other end of that call is unchanged.

--only-user-id ID (manual DB-only mode): skips steps 1-3 entirely — no
    scraping, no job enrichment, no job embedding. Just refreshes that one
    user's own profile embedding if it changed since last embed (free/cheap
    when nothing changed — profile_embed's hash check makes this a no-op most
    of the time), then steps 4-5 for that user only. Wrapped in the same
    running/complete/failed search_quota bookkeeping v1's --only-user-id path
    already uses (src.quota_store.set_status), unchanged. Deliberate cost
    cap: a manual run costs roughly one Tier 1 call + a handful of Tier 2
    calls (~$0.03), never a full scrape+enrich+embed pass (~$2+).

A user with an empty profile (src.profile_embed.build_embedding_text returning
"" — the same "incomplete profile" signal already used for embedding) or no
profile_embedding at all is skipped with a logged reason — never crashes the
run. There are archived and empty profiles in `profiles` in production.

dry_run scope: ingest.ingest()/enrich_jobs.enrich_jobs() both already support
compute-without-writing. src.match_v2/src.tier1_rerank/src.tier2_enrich do
not (they always write) — retrofitting that was out of scope here — so
dry_run skips the entire per-user matching+digest step rather than spending
real LLM cost on a preview it can't actually withhold from being written.

`pipeline_runs` (shared with v1 — see src/run_store.py): run() opens one row
here (_open_pipeline_run) BEFORE anything else, using the same run_id every
run_metrics write below stamps — run_metrics.run_id is a foreign key to
pipeline_runs.run_id, so every stage write this run makes (ingest/enrich/
tier1/tier2/match) depends on that parent row already existing. Closed
(_close_pipeline_run) in a finally, status='complete'/'failed' — same
open/close shape as v1's start_run()/complete_run(), but v2 owns run_id
generation itself (needed so every stage shares one id), so it can't reuse
those v1 functions directly. No-op in dry_run, same as every other write here.
"""
from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone

import anthropic

from src.adapters.aac import AACAdapter
from src.adapters.eighty_thousand_hours import EightyThousandHoursAdapter
from src.adapters.iap import IAPAdapter
from src.adapters.probably_good import ProbablyGoodAdapter
from src.closure import close_jobs
from src.enrich_jobs import enrich_jobs as run_enrich_jobs
from src.ingest import ingest as run_ingest
from src.match_v2 import fetch_matched_job_ids, gate_survivors_with_rejections, vector_rerank
from src.notify import send_match_digest
from src.profile_embed import backfill_profile_embeddings, build_embedding_text
from src.quota_store import set_status
from src.supabase_client import fetch_all, fetch_all_in, get_client, insert_run_metrics
from src.tier1_rerank import TIER1_MAX_PER_USER, fetch_valid_verdict_job_ids, rerank_tier1_batch
from src.tier2_enrich import fetch_candidates as fetch_tier2_candidates
from src.tier2_enrich import tier2_enrich as run_tier2_enrich
from src.user_store import fetch_users

log = logging.getLogger(__name__)

# Source adapter registry — every known adapter, with an explicit enabled flag.
# Toggle here to disable a source without deleting its adapter code (its
# module, tests, and the underlying src/sources/*.py fetch code it wraps all
# stay intact — see each adapter module for that reuse). Re-enable by flipping
# back to True; no other change needed. Disabling an adapter here also means
# its .fetch() is never called, so any credentials that adapter alone needs
# (e.g. IAPAdapter's GOOGLE_SERVICE_ACCOUNT_JSON/IAP_SHEET_ID/IAP_SHEET_TAB —
# read lazily inside src/sources/iap.py's fetch_raw_rows(), not at import
# time) are never required, even if the environment still provides them.
#
# IAP disabled 2026-08-18 — the referral sheet's postings are lower-signal
# and no longer worth the per-run Sheets API call; the sheet itself is
# untouched, so re-enabling is a pure flag flip if that changes.
_ADAPTER_REGISTRY: list[tuple[type, bool]] = [
    (EightyThousandHoursAdapter, True),
    (ProbablyGoodAdapter, True),
    (IAPAdapter, False),
    # AAC last, deliberately: it's the newest/least-audited source (no v1
    # history, no production run yet) — if it ever needs to be pulled
    # quickly, every other board's fetch/ingest has already run first.
    (AACAdapter, True),
]

_ADAPTERS = [cls for cls, enabled in _ADAPTER_REGISTRY if enabled]

_JOBS_POOL_COLUMNS = "id,title,org,canonical_url,location_field,raw_description,enrichment,source_board,enriched_at"


def _fetch_enriched_jobs_pool(client) -> list[dict]:
    return fetch_all(client.table("jobs").select(_JOBS_POOL_COLUMNS).eq("status", "enriched"))


def _check_source_fetch_health(client, source_boards: list[str]) -> list[str]:
    """Return the subset of *source_boards* that have logged complete=False
    on their last TWO source_fetch_log rows (including the one this run just
    wrote) — two consecutive incomplete fetches means something is actually
    broken, not a one-off Algolia hiccup. Checks only the sources passed in
    (the currently-enabled ones) — a disabled source's old rows are
    irrelevant and must not fail a run.

    Deliberately does NOT raise itself — see _raise_if_source_unhealthy()
    and run()'s call sites for why the actual raise is deferred to the end
    of the run, after ingest/enrich/matching/digest have already run for
    every healthy source and active user. A broken source's failure must
    fail the step (so it's impossible to miss, unlike an email that can
    bounce/get filtered — which is exactly how the ProbablyGood adapter's
    AlgoliaSearchKey outage went unnoticed for 7 weeks), but must NOT block
    everything else this run could still legitimately accomplish.
    """
    broken: list[str] = []
    for board in source_boards:
        rows = fetch_all(
            client.table("source_fetch_log").select("complete,fetched_at").eq("source_board", board)
        )
        rows.sort(key=lambda r: r["fetched_at"], reverse=True)
        last_two = rows[:2]
        if len(last_two) == 2 and all(r["complete"] is False for r in last_two):
            broken.append(board)
    return broken


def _raise_if_source_unhealthy(broken_sources: list[str]) -> None:
    """Raises (see _check_source_fetch_health's docstring for why this is a
    separate, deferred step) iff *broken_sources* is non-empty. Call sites in
    run() place this at the END of each path — after whatever
    ingest/enrich/matching/digest work that path does has already completed
    — never before it, so one broken source can't starve a healthy source's
    jobs or an active user's matches of a run they'd otherwise have gotten.
    """
    if broken_sources:
        raise RuntimeError(
            f"Source(s) {broken_sources} logged complete=False for 2 consecutive runs "
            f"(source_fetch_log) — likely broken, not a transient hiccup. Failing this "
            f"run's step now that every healthy source/active user has already been "
            f"processed."
        )


def _open_pipeline_run(client, *, run_id: str, trigger: str, dry_run: bool) -> None:
    """Insert the `pipeline_runs` parent row for *run_id* — MUST happen before
    any run_metrics write this run makes (ingest/enrich/tier1/tier2/match all
    stamp run_metrics.run_id with this same run_id), because run_metrics.run_id
    is a foreign key to pipeline_runs.run_id. Without a matching parent row,
    every one of those stage inserts 23503s (confirmed: this table/FK is shared
    with v1's pipeline.py — see src/run_store.py — which has always opened its
    own row via start_run() before writing anything; v2 generates its own
    run_id up front instead of letting start_run() mint one, so it needs its
    own open/close pair here rather than reusing that v1 function).

    Deliberately NOT wrapped in try/except like run_store.py's telemetry
    writes: this row isn't just observability for v2 — it's a hard prerequisite
    every later run_metrics write depends on via the FK, so a failure here
    should surface immediately and stop the run, not be swallowed into a
    confusing 23503 several stages later.

    No-op in dry_run — nothing downstream writes run_metrics in that mode
    either (ingest.py/enrich_jobs.py/_match_and_notify_user all skip their own
    writes when dry_run=True), so there would be nothing for this row to
    parent.
    """
    if dry_run:
        return
    client.table("pipeline_runs").insert({
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
    }).execute()


def _close_pipeline_run(client, *, run_id: str, status: str, user_count: int | None, dry_run: bool) -> None:
    """Finalize the pipeline_runs row _open_pipeline_run wrote — status is
    'complete' or 'failed' (matching v1's src/run_store.py vocabulary — the
    two pipelines now share this table), or 'incomplete' — enrichment
    (src.enrich_jobs.enrich_jobs) returned a 'pending' result (its Batches
    API poll budget was exhausted, not an error) and run() stopped cleanly
    before matching on partial data; distinct from 'failed' so an ops
    dashboard doesn't page on something that's expected to just resume next
    run (or via src/drain_batches.py's hourly check). user_count is the only v1-style total
    populated here (len(user_results) — the rest of v1's total_* columns, e.g.
    total_scored/total_inserted, describe a fresh-scrape-per-run model that
    doesn't map cleanly onto v2's shared, persistent job pool; left NULL rather
    than guessed). No-op in dry_run (nothing was opened).

    Unlike _open_pipeline_run, failures here ARE swallowed (logged, not
    raised) — by the time this runs, every stage that actually depended on the
    parent row existing has already succeeded or the run is already failing
    for its own reason; a telemetry-only failure to record the final status
    must not mask that outcome or crash an otherwise-successful run, same
    principle as run_store.py's own writes.
    """
    if dry_run:
        return
    try:
        client.table("pipeline_runs").update({
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "user_count": user_count,
        }).eq("run_id", run_id).execute()
    except Exception as exc:
        log.warning("pipeline_runs close (run %s, status=%s) failed — telemetry only, ignored: %r", run_id, status, exc)


# 'application_support_v1' experiment (experiment_assignments table) — arm
# only ever affects _build_digest_payload's output, never gating/Tier 1/
# Tier 2 generation/threshold/the matches_v2 insert itself. A user with no
# experiment_assignments row (existing users, admins) defaults to _FULL_ARM.
_APPLICATION_SUPPORT_EXPERIMENT = "application_support_v1"
_FULL_ARM = "full"
_SHORTLIST_ONLY_ARM = "shortlist_only"


def _build_digest_payload(
    matches_v2_rows: list[dict], jobs_by_id: dict[str, dict], *, arm: str = _FULL_ARM,
) -> list[dict]:
    """Map v2's matches_v2 rows into v1's exact digest payload shape
    ({title, org, location, fit_score, why_fits, url}) — notify.send_match_digest
    is reused UNCHANGED, so this must match the field names its Edge Function
    contract already expects. location comes from jobs.location_field (v1's
    posting.location has no cleaner v2 equivalent).

    fit_score is tier1_score / 100: matches_v2.tier1_score is stored on the
    0-100 scale (see scoring_weights.insert_threshold's own x100 comment
    above), but v1's contract — and this Edge Function's existing template —
    expects 0.0-1.0 (schemas.py's ScoredPosting.fit_score). Sending the raw
    0-100 value unscaled was a real bug (confirmed 2026-11-03, never caught
    because MATCH_DIGEST_URL has been 401ing since before this field existed
    — see src.pipeline_v2._finish_user_matching's notified handling).

    arm: the 'application_support_v1' experiment's per-user arm (see
    _fetch_experiment_arms) — _FULL_ARM (default, including every existing
    user/admin with no experiment_assignments row) includes why_fits;
    _SHORTLIST_ONLY_ARM omits the key entirely (not null/empty-string — a
    true omission). why_fits is the ONLY Tier 2 output field that reaches
    this payload at all today: org_summary/why_not_fits/emphasize_in_cv/
    deemphasize are written to matches_v2.tier2 (tier2_enrich.py) but never
    read here, by any arm — see this experiment's Phase 0 report. Nothing
    about which matches reach this function (gating, Tier 1, Tier 2
    generation/threshold, the matches_v2 insert itself) depends on arm —
    this is the only place arm has any effect, by design: ranking/insert
    stays byte-identical across arms.

    summary is jobs.enrichment.summary verbatim (NOT application support —
    a neutral role description shared across the whole org/job pool, not
    LLM-generated per-user fit reasoning) and is sent for EVERY arm,
    unconditionally — deliberately decided 2026-11-04 against adding a new
    enrichment field after confirming the existing summary (already
    neutral/descriptive, "do not editorialize or assess fit" per its
    prompt) already satisfies the requirement; see src.enrich_jobs.
    build_prompt field 1. Must never be touched by the arm branch above, and
    must never feed back into _job_card_text/gate_v2.py/the embedding-text
    builders (enrich_jobs.py's _embed_enriched/_catch_up_embed) — those stay
    exactly as they were, reading enrichment.summary for their own,
    unrelated purposes. Omitted entirely (not null/empty-string) when a job
    lacks a summary — shouldn't happen for status='enriched' jobs (summary
    is a required JobEnrichment field), but guarded defensively anyway."""
    payload = []
    for row in matches_v2_rows:
        job = jobs_by_id.get(row["job_id"])
        if job is None:
            continue
        tier2 = row.get("tier2") or {}
        enrichment = job.get("enrichment") or {}
        entry = {
            "title": job.get("title"),
            "org": job.get("org"),
            "location": job.get("location_field"),
            "fit_score": row["tier1_score"] / 100,
            "url": job.get("canonical_url"),
        }
        summary = enrichment.get("summary")
        if summary:
            entry["summary"] = summary
        if arm != _SHORTLIST_ONLY_ARM:
            entry["why_fits"] = tier2.get("why_fits")
        payload.append(entry)
    return payload


_DIGEST_BATCH_CAP = 20


def _fetch_experiment_arms(client, user_ids: list[str], experiment: str) -> dict[str, str]:
    """{user_id: arm} for every experiment_assignments row matching
    *experiment* among *user_ids* — ONE query (chunked via fetch_all_in,
    same as every other id-list lookup in this module) covering every user
    about to go through Phase C this run, not a per-user query. A user with
    no row here is simply absent from the returned dict; callers default
    that to _FULL_ARM (see _build_digest_payload) rather than this function
    inventing a sentinel value."""
    rows = fetch_all_in(
        client.table("experiment_assignments"), "user_id", user_ids,
        select_cols="user_id,arm", filter_fn=lambda q: q.eq("experiment", experiment),
    )
    return {row["user_id"]: row["arm"] for row in rows}


def _send_queued_digest(client, user, *, arm: str = _FULL_ARM) -> bool:
    """Send (at most) this user's _DIGEST_BATCH_CAP oldest never-delivered
    matches_v2 rows — NOT just whatever Tier 2 wrote THIS run. Deliberately
    decoupled from this run's own candidates: a match that failed to send,
    or never even attempted (e.g. MATCH_DIGEST_URL unset in the calling
    process's environment — confirmed for real, 2026-11-03), must resurface
    on a LATER run rather than vanish. Once a matches_v2 row exists, the
    already-matched skip-set (match_v2.fetch_matched_job_ids) means that
    job can never reach Tier 1/2 again — so matches_v2.notified_at IS NULL
    is the only durable record that a match still needs delivering.

    On a confirmed 2xx only, stamps notified_at = now() for EXACTLY the row
    ids included in the payload actually sent. A failed send (or nothing
    queued, or email_on_match off) leaves notified_at untouched, so the
    same rows are picked up again next run.

    A queued row whose job has closed, or whose job can no longer be found
    at all, is excluded from the payload AND stamped notified_at = now()
    (with notified_skip_reason recording why) unconditionally — regardless
    of whether the rest of this batch's send succeeds. Without this, a
    dead-job row never leaves the notified_at-IS-NULL queue (it's always
    among the oldest, so it would re-occupy a _DIGEST_BATCH_CAP slot every
    run forever, permanently starving genuinely sendable matches behind
    it) — confirmed worth guarding against 2026-11-03, before any real
    instance of it has actually happened.
    """
    if not user.profile.get("email_on_match"):
        return False

    backlog_rows = (
        client.table("matches_v2").select("id,job_id,tier1_score,tier2")
        .eq("user_id", user.user_id).is_("notified_at", "null")
        .order("created_at").limit(_DIGEST_BATCH_CAP).execute().data
    )
    if not backlog_rows:
        return False

    job_ids = [row["job_id"] for row in backlog_rows]
    jobs = fetch_all_in(
        client.table("jobs"), "id", job_ids,
        select_cols="id,title,org,location_field,canonical_url,status,enrichment",
    )
    jobs_by_id = {j["id"]: j for j in jobs}

    skip_ids_by_reason: dict[str, list[str]] = {}
    sendable_rows = []
    for row in backlog_rows:
        job = jobs_by_id.get(row["job_id"])
        if job is None:
            skip_ids_by_reason.setdefault("job_missing", []).append(row["id"])
        elif job.get("status") == "closed":
            skip_ids_by_reason.setdefault("job_closed", []).append(row["id"])
        else:
            sendable_rows.append(row)

    now_iso = datetime.now(timezone.utc).isoformat()
    for reason, skip_ids in skip_ids_by_reason.items():
        client.table("matches_v2").update({
            "notified_at": now_iso, "notified_skip_reason": reason,
        }).in_("id", skip_ids).execute()

    if not sendable_rows:
        return False

    payload = _build_digest_payload(sendable_rows, jobs_by_id, arm=arm)
    try:
        # send_match_digest() never raises by contract, but this try/except
        # is a deliberate second layer of defense (see CLAUDE.md) — notified
        # reflects its actual return value (True only on a confirmed 2xx),
        # not just "was it called".
        notified = send_match_digest(user.user_id, payload)
    except Exception as exc:
        log.warning("Match-digest email failed for user %s: %s", user.user_id, exc)
        notified = False

    if notified:
        sent_ids = [row["id"] for row in sendable_rows]
        client.table("matches_v2").update({
            "notified_at": datetime.now(timezone.utc).isoformat(),
        }).in_("id", sent_ids).execute()  # notified_skip_reason untouched — this is a real send, not a skip

    return notified


def _prepare_tier1_request(
    client, user, jobs_pool: list[dict], *, gate_rejections_out: dict | None = None,
) -> dict:
    """Phase A for one user (no LLM calls, no writes): gate_v2 survivors,
    excluding already-matched jobs, THEN excluding jobs this user already
    has a VALID tier1_verdicts row for (src.tier1_rerank.
    fetch_valid_verdict_job_ids) — re-scoring an already-validly-scored job
    is exactly the waste Phase 2 exists to stop — THEN vector-rerank the
    remainder to TIER1_MAX_PER_USER.

    Runs its gate+vector computation unconditionally (even in dry_run — it's
    free, no LLM cost) so dry_run still reports real gate_survivors counts;
    callers skip Phase B/C in dry_run instead, same as before.

    Returns {"user_id", "skipped": str|None, "gate_survivor_count": int,
    "profile_content_hash": str|None, "job_cards": list[dict]}. "skipped" is
    "empty_profile" or "no_profile_embedding" for a user with an incomplete
    profile — job_cards is empty in that case, nothing else is computed.

    gate_rejections_out, if given, is mutated in place: this user's per-rule
    gate rejection counts are recorded under user.user_id, for run()'s
    aggregate run_metrics.gate_rejections write — kept out of the return
    dict deliberately, same reasoning as before (observability side-channel,
    not part of the matching contract several tests assert exact equality on).
    """
    text = build_embedding_text(user.profile)
    if not text:
        log.warning("User %s: empty profile — skipping matching", user.user_id)
        return {"user_id": user.user_id, "skipped": "empty_profile", "gate_survivor_count": 0,
                "profile_content_hash": None, "job_cards": []}

    profile_rows = (
        client.table("profiles").select("profile_embedding,profile_content_hash")
        .eq("user_id", user.user_id).execute().data
    )
    profile_row = profile_rows[0] if profile_rows else {}
    profile_embedding = profile_row.get("profile_embedding")
    if not profile_embedding:
        log.warning("User %s: no profile_embedding yet — skipping matching", user.user_id)
        return {"user_id": user.user_id, "skipped": "no_profile_embedding", "gate_survivor_count": 0,
                "profile_content_hash": None, "job_cards": []}

    profile_content_hash = profile_row.get("profile_content_hash")

    matched_ids = fetch_matched_job_ids(client, user.user_id)
    candidate_jobs = [job for job in jobs_pool if job["id"] not in matched_ids]

    survivors, rejection_counts = gate_survivors_with_rejections(candidate_jobs, user.profile)
    if gate_rejections_out is not None and rejection_counts:
        gate_rejections_out[user.user_id] = rejection_counts

    valid_verdict_ids = fetch_valid_verdict_job_ids(client, user.user_id, profile_content_hash)
    unverdicted_survivors = [job for job in survivors if job["id"] not in valid_verdict_ids]

    ranked = vector_rerank(
        client, profile_embedding=profile_embedding,
        candidate_job_ids=[job["id"] for job in unverdicted_survivors], top_n=TIER1_MAX_PER_USER,
    )
    by_id = {job["id"]: job for job in unverdicted_survivors}
    job_cards = [
        {
            "id": job["id"], "title": job["title"], "org": job["org"], "enrichment": job["enrichment"],
            # Carried through (not used in the Tier 1 prompt itself, see
            # tier1_rerank._job_card_text) so tier1_rerank._build_manifest
            # can snapshot the REQUEST-TIME enriched_at for every job sent —
            # see that module's docstring on why write-time state is wrong.
            "enriched_at": job.get("enriched_at"),
        }
        for row in ranked if (job := by_id.get(row["id"])) is not None
    ]

    log.info(
        "User %s: %d gate survivor(s), %d still needing Tier 1 (valid-verdict-excluded, top-%d by similarity)",
        user.user_id, len(survivors), len(job_cards), TIER1_MAX_PER_USER,
    )
    return {
        "user_id": user.user_id, "skipped": None, "gate_survivor_count": len(survivors),
        "profile_content_hash": profile_content_hash, "job_cards": job_cards,
    }


def _finish_user_matching(
    client, anthropic_client, user, prep: dict, *, dry_run: bool, run_id=None, trigger=None, arm: str = _FULL_ARM,
) -> dict:
    """Phase C for one user: Tier 2 candidates = valid tier1_verdicts rows
    >= this user's threshold with no existing matches_v2 row
    (tier2_enrich.fetch_candidates — reads the DB directly, so this is
    resumable from whatever Tier 1 has written so far, including verdicts
    the external hourly drainer applied between runs), then Tier 2 itself,
    then digest email for whatever's new. Returns a small per-user result
    dict for run-level reporting — same shape _match_and_notify_user used to
    return, before the Phase A/B/C split.
    """
    base_result = {"user_id": user.user_id, "gate_survivors": prep["gate_survivor_count"]}
    if prep["skipped"] is not None:
        return {"user_id": user.user_id, "skipped": prep["skipped"]}
    if dry_run:
        return {**base_result, "tier1_scored": 0, "tier2_enriched": 0, "notified": False}

    # scoring_weights.insert_threshold is stored 0-1 (fractional); tier1_score is
    # 0-100 (see src/models/tier1_rerank.py's JobScore). Scale up before comparing
    # — this is this user's own per-user bar, read fresh off their rubric every
    # run, not a global constant shared across users.
    #
    # math.ceil(), not a plain *100: tier1_verdicts.score is an INTEGER column,
    # and a fractional threshold (e.g. 0.725 * 100 = 72.5) sent as a .gte()
    # filter value gets stringified into the request URL as "72.5" — Postgres
    # then rejects it outright trying to parse "72.5" as an integer (confirmed
    # for real: 22P02 invalid input syntax for type integer). Rounding UP (not
    # down, not banker's rounding) preserves ">=" semantics against an integer
    # column: "score >= 72.5" and "score >= 73" select exactly the same set of
    # integer scores, so ceil is the one rounding direction that can't
    # accidentally admit a score the fractional threshold would have excluded.
    threshold = math.ceil(user.rubric["threshold"] * 100)

    # tier1_scored reports how many candidates Tier 1 actually scored for
    # this user THIS run (the chunks Phase A built) — not how many Tier 2
    # candidates resulted, which is a different, threshold-filtered number.
    tier1_scored = len(prep["job_cards"])

    candidates = fetch_tier2_candidates(
        client, user_id=user.user_id, profile=user.profile,
        profile_content_hash=prep["profile_content_hash"], threshold=threshold,
    )
    if not candidates:
        # Even with zero new Tier 2 candidates THIS run, this user may still
        # have an older never-delivered backlog (see _send_queued_digest) —
        # must still attempt it, not short-circuit to notified=False.
        return {**base_result, "tier1_scored": tier1_scored, "tier2_enriched": 0,
                "notified": _send_queued_digest(client, user, arm=arm)}

    tier2_result = run_tier2_enrich(
        client, anthropic_client, candidates=candidates, threshold=threshold,
        run_id=run_id, trigger=trigger,
    )
    if tier2_result.get("status") == "pending":
        # This user's own Tier 2 batch didn't finish within budget — NOT an
        # error, and NOT a reason to block any OTHER user's Tier 2/digest
        # (each user's Tier 2 batch is independent, unlike Tier 1's single
        # shared submission). Whatever round 1 already committed for this
        # user is real; just skip the digest for now, resume next run.
        log.info("User %s: Tier 2 pending (batch %s) — digest deferred to next run", user.user_id, tier2_result.get("batch_id"))
        return {**base_result, "tier1_scored": tier1_scored, "tier2_enriched": tier2_result["enriched"], "notified": False}

    return {
        **base_result, "tier1_scored": tier1_scored,
        "tier2_enriched": tier2_result["enriched"], "notified": _send_queued_digest(client, user, arm=arm),
    }


# Consent gating — a user is processed at all (gating, scoring, Tier 2,
# matches_v2 writes, digest) only if their LATEST user_consents row for
# this key has granted=True. Fail closed: no row, a withdrawn (granted=
# False) latest row, or any other key is irrelevant here are all treated
# identically — "not processed", never "processed by default".
_REQUIRED_CONSENT_KEY = "data_storage"


def _fetch_consented_user_ids(client, user_ids: list[str]) -> set[str]:
    """{user_id} for every id in *user_ids* whose LATEST user_consents row
    (by created_at) for _REQUIRED_CONSENT_KEY has granted=True — ONE
    chunked query (fetch_all_in) covering every candidate user this run,
    not a per-user query, mirroring _fetch_experiment_arms's shape.

    user_consents is append-only (a new row per consent change, see
    20261104100000_match_events_user_consents_baseline.sql — no update-in-
    place), so "latest" must be resolved client-side by comparing
    created_at across every row this user has for this key, not just
    taking whichever row the API happens to return first.

    A user_id with NO data_storage row at all, or whose latest row has
    granted=False (consent withdrawn after being granted), is simply
    absent from the returned set — callers must treat absence as "do not
    process" (fail closed), never assume absence means "never asked,
    default to allowed"."""
    rows = fetch_all_in(
        client.table("user_consents"), "user_id", user_ids,
        select_cols="user_id,granted,created_at",
        filter_fn=lambda q: q.eq("consent_key", _REQUIRED_CONSENT_KEY),
    )
    latest_by_user: dict[str, dict] = {}
    for row in rows:
        current = latest_by_user.get(row["user_id"])
        if current is None or row["created_at"] > current["created_at"]:
            latest_by_user[row["user_id"]] = row
    return {user_id for user_id, row in latest_by_user.items() if row["granted"] is True}


def run(
    *, dry_run: bool = False, only_user_id: str | None = None,
    _client=None, _anthropic_client=None,
) -> dict:
    """Run the v2 pipeline. only_user_id=None is the scheduled/cron full
    pipeline; only_user_id=X is the manual DB-only single-user path (see
    module docstring for both). Returns a summary dict for reporting.
    """
    client = _client if _client is not None else get_client()
    anthropic_client = _anthropic_client if _anthropic_client is not None else anthropic.Anthropic()
    track_quota = only_user_id is not None and not dry_run
    # Pessimistic by default, upgraded only once we're sure — matches v1's
    # src/pipeline.py's run_status convention. 'incomplete' (distinct from
    # 'failed') is reached only when enrichment itself returns 'pending' (its
    # Batches API poll budget was exhausted, not an error) — see the pending
    # gate below. only_user_id mode can never reach 'incomplete': it skips
    # enrichment entirely (steps 1-3 below only run when only_user_id is None).
    run_outcome = "failed"
    user_results: list[dict] = []  # defined before try/finally — _close_pipeline_run
                                    # in finally needs this even if we never reach the
                                    # per-user loop (e.g. a failure during ingest)
    # One run_id ties every stage's run_metrics row (ingest/enrich/tier1/tier2/
    # match) back to this single invocation of run(); trigger distinguishes the
    # cron-scheduled full pipeline from a manual --only-user-id call.
    run_id = str(uuid.uuid4())
    trigger = "manual" if only_user_id is not None else "scheduled"

    try:
        # MUST happen before anything writes run_metrics — every stage below
        # (ingest/enrich/tier1/tier2/match) stamps run_metrics.run_id with this
        # same run_id, and that column is a foreign key to pipeline_runs.run_id.
        _open_pipeline_run(client, run_id=run_id, trigger=trigger, dry_run=dry_run)

        if track_quota:
            set_status(client, only_user_id, "running")

        ingest_counts = None
        enrich_result = None
        # Populated below (scheduled mode only) and checked via
        # _raise_if_source_unhealthy() at the END of every return path this
        # function has — never checked eagerly, so a broken source never
        # blocks ingest/enrich/matching for a healthy one or an active user.
        source_health_broken: list[str] = []
        if only_user_id is None:
            active_names = [cls.__name__ for cls, enabled in _ADAPTER_REGISTRY if enabled]
            skipped_names = [cls.__name__ for cls, enabled in _ADAPTER_REGISTRY if not enabled]
            log.info("Adapters active: %s | skipped (disabled): %s", active_names, skipped_names)

            postings = []
            fetch_log_rows = []
            for adapter_cls in _ADAPTERS:
                adapter = adapter_cls()
                fetched_at = datetime.now(timezone.utc).isoformat()
                try:
                    result = adapter.fetch()
                    postings.extend(result.postings)
                    fetch_log_rows.append({
                        "run_id": run_id, "source_board": adapter.source_board,
                        "fetched": result.fetched, "nb_hits": result.nb_hits,
                        "complete": result.complete, "fetched_at": fetched_at,
                    })
                    log.info(
                        "Adapter %s fetched %d posting(s) (fetched=%d, nb_hits=%s, complete=%s)",
                        adapter_cls.__name__, len(result.postings), result.fetched, result.nb_hits, result.complete,
                    )
                except Exception:
                    log.exception("Adapter %s failed to fetch — skipping this source", adapter_cls.__name__)
                    fetch_log_rows.append({
                        "run_id": run_id, "source_board": adapter.source_board,
                        "fetched": 0, "nb_hits": None, "complete": False, "fetched_at": fetched_at,
                    })

            # source_fetch_log row per adapter per run, even on failure (see
            # FetchResult's docstring) — written here, not inside ingest.py,
            # since fetch-completeness is a property of the ADAPTER call, not
            # of ingest()'s dedup step. Skipped in dry_run, same convention as
            # every other write in this module.
            if fetch_log_rows and not dry_run:
                client.table("source_fetch_log").insert(fetch_log_rows).execute()
                # Computed here (so "2 consecutive" includes the row just
                # inserted above) but NOT acted on yet — see
                # _raise_if_source_unhealthy()'s call sites below, all of
                # which come after ingest/enrich/matching/digest. Skipped
                # entirely in dry_run (no row was written this run to check
                # against).
                source_health_broken = _check_source_fetch_health(client, [cls.source_board for cls in _ADAPTERS])

            ingest_counts_obj = run_ingest(postings, client=client, dry_run=dry_run, run_id=run_id, trigger=trigger)
            ingest_counts = ingest_counts_obj.as_dict()
            log.info("Ingest: %s", ingest_counts)

            if not dry_run:
                # complete_sources: derived from fetch_log_rows directly —
                # "this run's fetch was complete", not a historical
                # source_fetch_log query (see src.closure.plan_closures's
                # docstring on why that distinction matters for rule (a)).
                complete_sources = frozenset(r["source_board"] for r in fetch_log_rows if r["complete"])
                # disabled_boards: every registry entry NOT enabled (IAP,
                # today) — see closure.compute_closure_health()'s docstring
                # for why its jobs are reported separately from the
                # actionable closure-health signal, not folded into it.
                disabled_boards = frozenset(cls.source_board for cls, enabled in _ADAPTER_REGISTRY if not enabled)
                closure_result = close_jobs(
                    client, complete_sources=complete_sources, disabled_boards=disabled_boards,
                    run_id=run_id, trigger=trigger,
                )
                log.info("Closure: %s", closure_result)

            enrich_result = run_enrich_jobs(
                client=client, anthropic_client=anthropic_client, dry_run=dry_run,
                run_id=run_id, trigger=trigger,
            )
            log.info("Enrich: %s", enrich_result)

            if enrich_result.get("status") == "pending":
                # The enrichment batch (round 1 or round 2 — see
                # src.enrich_jobs.enrich_jobs) didn't finish within its poll
                # budget. NOT an error — the Anthropic Batches API only
                # guarantees completion within 24h, and matching on a
                # partially-enriched jobs pool would be wrong regardless of
                # why enrichment is incomplete. Stop here, before profile
                # embedding refresh and the per-user matching loop, and let
                # the next scheduled run (or src/drain_batches.py's hourly
                # check) resume it — the batch itself, and the jobs still at
                # status='pending_enrich', are untouched and safe to resume.
                log.info("Enrichment pending — stopping this run before matching (run_id=%s)", run_id)
                # Matching was already going to be skipped this run
                # regardless of source health (enrichment isn't done) — so
                # raising here blocks nothing an unhealthy source wouldn't
                # already be blocking, and still fails the step.
                _raise_if_source_unhealthy(source_health_broken)
                run_outcome = "incomplete"
                return {"ingest": ingest_counts, "enrich": enrich_result, "users": [], "status": run_outcome}

            profile_embed_result = backfill_profile_embeddings(client) if not dry_run else {
                "selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0,
            }
            log.info("Profile embed refresh (all users): %s", profile_embed_result)

        users, excluded = fetch_users(client, only_user_id=only_user_id)
        if excluded:
            log.warning(
                "fetch_users excluded %d user(s) with malformed gate config: %s",
                len(excluded), [e.user_id for e in excluded],
            )
        if not users:
            if track_quota:
                raise RuntimeError(f"only_user_id={only_user_id}: no profile+scoring_weights row found")
            log.warning("No users with both a profile and scoring_weights row — nothing to do")
            # No matching happens in this branch either way (no users) — same
            # reasoning as the pending-enrichment return above.
            _raise_if_source_unhealthy(source_health_broken)
            run_outcome = "complete"
            return {"ingest": ingest_counts, "enrich": enrich_result, "users": []}

        # Consent gating — fail closed: a user is processed further this
        # run only if their LATEST user_consents row for
        # _REQUIRED_CONSENT_KEY has granted=True. Deliberately placed AFTER
        # the "no users" check above (that check's error message is about
        # a missing profile+scoring_weights row specifically, not consent —
        # keeping them separate avoids a misleading error for the manual
        # --only-user-id path) but BEFORE every other per-user step below,
        # including the single-user profile-embed refresh just past this
        # block — "skip them entirely" means no per-user work of any kind,
        # not just no gating/scoring/Tier 2/matches_v2/digest.
        consented_ids = _fetch_consented_user_ids(client, [user.user_id for user in users])
        users_before_consent = users
        users = [user for user in users_before_consent if user.user_id in consented_ids]
        skipped_for_consent = len(users_before_consent) - len(users)
        if skipped_for_consent:
            log.info(
                "Consent gating: skipping %d user(s) this run — no granted '%s' consent "
                "(no gating, scoring, Tier 2, matches_v2 writes, or digest for them)",
                skipped_for_consent, _REQUIRED_CONSENT_KEY,
            )

        # Manual/single-user mode: the one user fetch_users() found just got
        # filtered out above — without this, search_quota would otherwise be
        # left at 'complete' (zero matches, same as a real run that found
        # nothing) by the finally block below, indistinguishable from "we
        # processed you and found no matches" when nothing was actually
        # processed at all. Mirrors the "no profile+scoring_weights row"
        # RuntimeError just above this block — same track_quota/except
        # handling, distinct message so logs show the real cause.
        if track_quota and not users:
            raise RuntimeError(f"only_user_id={only_user_id}: consent not granted for '{_REQUIRED_CONSENT_KEY}'")

        if only_user_id is not None and not dry_run and users:
            profile_embed_result = backfill_profile_embeddings(client, only_user_id=only_user_id)
            log.info("Profile embed refresh (single user): %s", profile_embed_result)

        jobs_pool = _fetch_enriched_jobs_pool(client)
        log.info("Matching against %d enriched job(s)", len(jobs_pool))

        # Phase A — per user, no LLM calls: gate -> exclude valid-verdict
        # jobs -> vector top-TIER1_MAX_PER_USER. Always computed, even in
        # dry_run (free, no LLM cost) — see _prepare_tier1_request's
        # docstring.
        gate_rejections: dict[str, dict] = {}
        prep_by_user: dict[str, dict] = {}
        for user in users:
            prep_by_user[user.user_id] = _prepare_tier1_request(
                client, user, jobs_pool, gate_rejections_out=gate_rejections,
            )

        # Phase B — ONE Batches submission across every user's every chunk.
        # Skipped entirely in dry_run (never call Tier 1/Tier 2 — same gate
        # as before the Phase A/B/C split).
        if not dry_run:
            tier1_user_requests = {
                user.user_id: {
                    "profile": user.profile, "gate_constraints": user.profile["hard_constraints"],
                    "job_cards": prep_by_user[user.user_id]["job_cards"],
                    "profile_content_hash": prep_by_user[user.user_id]["profile_content_hash"],
                }
                for user in users
                if prep_by_user[user.user_id]["skipped"] is None and prep_by_user[user.user_id]["job_cards"]
            }
            tier1_outcome = rerank_tier1_batch(
                client, anthropic_client, user_requests=tier1_user_requests, run_id=run_id, trigger=trigger,
            )
            log.info("Tier 1: %s", tier1_outcome)
            if tier1_outcome["status"] == "pending":
                # Mirrors enrich_jobs.enrich_jobs()'s pending gate: stop here,
                # before Tier 2/digest, and let the next scheduled run (or
                # src/drain_batches.py's hourly check) resume it. No matching
                # was going to run for anyone this call anyway once Tier 1
                # itself hasn't resolved, so raising the source-health check
                # here blocks nothing it wouldn't already be blocking.
                log.info("Tier 1 pending — stopping this run before Tier 2/digest (run_id=%s)", run_id)
                _raise_if_source_unhealthy(source_health_broken)
                run_outcome = "incomplete"
                return {"ingest": ingest_counts, "enrich": enrich_result, "users": [], "status": run_outcome}

        # Phase C — per user: Tier 2 candidates from tier1_verdicts (fresh
        # DB read, resumable from anything Tier 1/the drainer has written),
        # Tier 2 itself, digest. Arms fetched ONCE here for every digest
        # user this run (not a per-user query) — see _fetch_experiment_arms.
        arms_by_user = _fetch_experiment_arms(
            client, [user.user_id for user in users], _APPLICATION_SUPPORT_EXPERIMENT,
        )
        for user in users:
            result = _finish_user_matching(
                client, anthropic_client, user, prep_by_user[user.user_id],
                dry_run=dry_run, run_id=run_id, trigger=trigger,
                arm=arms_by_user.get(user.user_id, _FULL_ARM),
            )
            log.info("User %s: %s", user.user_id, result)
            user_results.append(result)

        # One row per run aggregating every user's gate rejection counts
        # (see match_v2.gate_survivors_with_rejections) — the per-rule
        # diagnostics run_metrics.gate_rejections was originally spec'd for and
        # never wired up. Skipped in dry_run (no writes at all in that mode) and
        # when nothing was rejected (nothing to report).
        if gate_rejections and not dry_run:
            insert_run_metrics(client, {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "stage": "match",
                "trigger": trigger,
                "gate_rejections": gate_rejections,
            })

        # Only now — after ingest/enrich/profile-embed/matching/digest have
        # ALL already run for every healthy source and active user — does a
        # broken source actually fail the step.
        _raise_if_source_unhealthy(source_health_broken)
        run_outcome = "complete"
        return {"ingest": ingest_counts, "enrich": enrich_result, "users": user_results}

    except Exception:
        if track_quota:
            set_status(client, only_user_id, "failed")
        raise
    finally:
        if track_quota and run_outcome == "complete":
            set_status(client, only_user_id, "complete", completed=True)
        _close_pipeline_run(
            client, run_id=run_id, status=run_outcome,
            user_count=len(user_results), dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="True North v2 pipeline (multi-user)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip ingest/enrich/matches writes; compute and log only",
    )
    parser.add_argument(
        "--only-user-id", type=str, default=None, metavar="ID",
        help="Manual DB-only path: match+notify a single user, no scraping/enrichment; writes search_quota",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    result = run(dry_run=args.dry_run, only_user_id=args.only_user_id)
    print(json.dumps(result, indent=2, default=str))
