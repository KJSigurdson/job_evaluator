"""True North v2 production orchestrator — replaces v1's src/pipeline.py as the
active daily matching + digest-email path (see .github/workflows/daily.yml: v1's
step is now if: false; v1's code and tables stay completely untouched).

Usage:
    python -m src.pipeline_v2 [--dry-run] [--only-user-id ID]

No --only-user-id (scheduled/cron mode) — the full pipeline:
    1. Active adapters (see _ADAPTER_REGISTRY — currently 80k, ProbablyGood;
       IAP disabled) -> ingest.ingest() (dedup/alias into `jobs`).
    2. enrich_jobs.enrich_jobs() — only status='pending_enrich' rows (new or
       content-changed); steady-state stays small by design (the shared jobs
       pool persists across runs, unlike v1's fresh-scrape-every-day model).
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
import uuid
from datetime import datetime, timezone

import anthropic

from src.adapters.eighty_thousand_hours import EightyThousandHoursAdapter
from src.adapters.iap import IAPAdapter
from src.adapters.probably_good import ProbablyGoodAdapter
from src.enrich_jobs import enrich_jobs as run_enrich_jobs
from src.ingest import ingest as run_ingest
from src.match_v2 import TOP_N, fetch_matched_job_ids, match_user
from src.notify import send_match_digest
from src.profile_embed import backfill_profile_embeddings, build_embedding_text
from src.quota_store import set_status
from src.supabase_client import fetch_all, get_client
from src.tier1_rerank import rerank_tier1
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
]

_ADAPTERS = [cls for cls, enabled in _ADAPTER_REGISTRY if enabled]

_JOBS_POOL_COLUMNS = "id,title,org,canonical_url,location_field,raw_description,enrichment,source_board"


def _fetch_enriched_jobs_pool(client) -> list[dict]:
    return fetch_all(client.table("jobs").select(_JOBS_POOL_COLUMNS).eq("status", "enriched"))


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


def _build_digest_payload(matches_v2_rows: list[dict], jobs_by_id: dict[str, dict]) -> list[dict]:
    """Map v2's matches_v2 rows into v1's exact digest payload shape
    ({title, org, location, fit_score, why_fits, url}) — notify.send_match_digest
    is reused UNCHANGED, so this must match the field names its Edge Function
    contract already expects. location comes from jobs.location_field (v1's
    posting.location has no cleaner v2 equivalent)."""
    payload = []
    for row in matches_v2_rows:
        job = jobs_by_id.get(row["job_id"])
        if job is None:
            continue
        tier2 = row.get("tier2") or {}
        payload.append({
            "title": job.get("title"),
            "org": job.get("org"),
            "location": job.get("location_field"),
            "fit_score": row["tier1_score"],
            "why_fits": tier2.get("why_fits"),
            "url": job.get("canonical_url"),
        })
    return payload


def _match_and_notify_user(
    client, anthropic_client, user, jobs_pool: list[dict], *, dry_run: bool,
    run_id: str | None = None, trigger: str | None = None, gate_rejections_out: dict | None = None,
) -> dict:
    """Steps 4-5 for one user: gate_v2 -> vector top-N -> Tier 1 -> Tier 2 ->
    matches_v2, then digest email for whatever's new this run. Returns a
    small per-user result dict for run-level reporting.

    gate_rejections_out, if given, is mutated in place: this user's per-rule
    gate rejection counts (see match_v2.gate_survivors_with_rejections) are
    recorded under user.user_id, for run()'s aggregate run_metrics.gate_rejections
    write. Kept out of this function's own return dict deliberately — several
    existing tests assert exact equality on it, and this is purely an
    observability side-channel, not part of the per-user matching contract.
    """
    text = build_embedding_text(user.profile)
    if not text:
        log.warning("User %s: empty profile — skipping matching", user.user_id)
        return {"user_id": user.user_id, "skipped": "empty_profile"}

    profile_rows = client.table("profiles").select("profile_embedding").eq("user_id", user.user_id).execute().data
    profile_embedding = profile_rows[0].get("profile_embedding") if profile_rows else None
    if not profile_embedding:
        log.warning("User %s: no profile_embedding yet — skipping matching", user.user_id)
        return {"user_id": user.user_id, "skipped": "no_profile_embedding"}

    matched_ids = fetch_matched_job_ids(client, user.user_id)
    candidate_jobs = [job for job in jobs_pool if job["id"] not in matched_ids]

    match_result = match_user(
        client, user_id=user.user_id, profile=user.profile,
        profile_embedding=profile_embedding, jobs=candidate_jobs, top_n=TOP_N,
    )
    if gate_rejections_out is not None:
        rejections = match_result.get("gate_rejection_counts") or {}
        if rejections:
            gate_rejections_out[user.user_id] = rejections
    top = match_result["top"]
    base_result = {"user_id": user.user_id, "gate_survivors": match_result["gate_survivor_count"]}
    if not top or dry_run:
        return {**base_result, "tier1_scored": 0, "tier2_enriched": 0, "notified": False}

    job_cards = [{"id": job["id"], "title": job["title"], "org": job["org"], "enrichment": job["enrichment"]} for job in top]
    tier1_results = rerank_tier1(client, anthropic_client, users={
        user.user_id: {
            "profile": user.profile, "gate_constraints": user.profile["hard_constraints"], "job_cards": job_cards,
        },
    }, run_id=run_id, trigger=trigger)

    top_by_id = {job["id"]: job for job in top}
    scores = tier1_results.get(user.user_id, {}).get("scores", [])
    candidates = [
        {"user_id": user.user_id, "profile": user.profile, "job": top_by_id[s.job_id],
         "tier1_score": s.score, "tier1_reason": s.reason}
        for s in scores if s.job_id in top_by_id
    ]

    # scoring_weights.insert_threshold is stored 0-1 (fractional); tier1_score is
    # 0-100 (see src/models/tier1_rerank.py's JobScore). Scale up before comparing
    # — this is this user's own per-user bar, read fresh off their rubric every
    # run, not a global constant shared across users.
    threshold = user.rubric["threshold"] * 100

    tier2_result = run_tier2_enrich(
        client, anthropic_client, candidates=candidates, threshold=threshold,
        run_id=run_id, trigger=trigger,
    )

    # Everything tier2_enrich() writes to matches_v2 for this user THIS run is,
    # by construction, new — matched_ids already excluded every pre-existing
    # match from the candidate pool before gating even ran, so no before/after
    # diffing is needed to know what's new for the digest.
    qualifying_job_ids = [c["job"]["id"] for c in candidates if c["tier1_score"] >= threshold]
    notified = False
    if qualifying_job_ids and user.profile.get("email_on_match") and not dry_run:
        new_rows = fetch_all(
            client.table("matches_v2").select("job_id,tier1_score,tier2")
            .eq("user_id", user.user_id).in_("job_id", qualifying_job_ids)
        )
        if new_rows:
            payload = _build_digest_payload(new_rows, top_by_id)
            try:
                send_match_digest(user.user_id, payload)
                notified = True
            except Exception as exc:
                log.warning("Match-digest email failed for user %s: %s", user.user_id, exc)

    return {
        **base_result, "tier1_scored": len(scores),
        "tier2_enriched": tier2_result["enriched"], "notified": notified,
    }


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
        if only_user_id is None:
            active_names = [cls.__name__ for cls, enabled in _ADAPTER_REGISTRY if enabled]
            skipped_names = [cls.__name__ for cls, enabled in _ADAPTER_REGISTRY if not enabled]
            log.info("Adapters active: %s | skipped (disabled): %s", active_names, skipped_names)

            postings = []
            for adapter_cls in _ADAPTERS:
                try:
                    fetched = adapter_cls().fetch()
                    postings.extend(fetched)
                    log.info("Adapter %s fetched %d posting(s)", adapter_cls.__name__, len(fetched))
                except Exception:
                    log.exception("Adapter %s failed to fetch — skipping this source", adapter_cls.__name__)

            ingest_counts_obj = run_ingest(postings, client=client, dry_run=dry_run, run_id=run_id, trigger=trigger)
            ingest_counts = ingest_counts_obj.as_dict()
            log.info("Ingest: %s", ingest_counts)

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
                run_outcome = "incomplete"
                log.info("Enrichment pending — stopping this run before matching (run_id=%s)", run_id)
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
            run_outcome = "complete"
            return {"ingest": ingest_counts, "enrich": enrich_result, "users": []}

        if only_user_id is not None and not dry_run:
            profile_embed_result = backfill_profile_embeddings(client, only_user_id=only_user_id)
            log.info("Profile embed refresh (single user): %s", profile_embed_result)

        jobs_pool = _fetch_enriched_jobs_pool(client)
        log.info("Matching against %d enriched job(s)", len(jobs_pool))

        gate_rejections: dict[str, dict] = {}
        for user in users:
            result = _match_and_notify_user(
                client, anthropic_client, user, jobs_pool, dry_run=dry_run,
                run_id=run_id, trigger=trigger, gate_rejections_out=gate_rejections,
            )
            log.info("User %s: %s", user.user_id, result)
            user_results.append(result)

        # One row per run aggregating every user's gate rejection counts
        # (see match_v2.gate_survivors_with_rejections) — the per-rule
        # diagnostics run_metrics.gate_rejections was originally spec'd for and
        # never wired up. Skipped in dry_run (no writes at all in that mode) and
        # when nothing was rejected (nothing to report).
        if gate_rejections and not dry_run:
            client.table("run_metrics").insert({
                "run_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "stage": "match",
                "trigger": trigger,
                "gate_rejections": gate_rejections,
            }).execute()

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
