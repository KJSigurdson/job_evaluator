-- True North v2 Phase 1 — job lifecycle (requeue-loop fix, closure, enrichment
-- observability). Additive/idempotent throughout (IF NOT EXISTS / safe
-- constraint replace), same convention as every prior v2 migration. Applied
-- by hand via the Supabase dashboard (no Supabase CLI in this workflow).

-- ---------------------------------------------------------------------------
-- 1. jobs.reenrich_requested_at — src/ingest.py's real-content-change branch
-- sets this (instead of flipping status back to pending_enrich) when a job
-- already has an enrichment, so it stays in the matching pool
-- (src/pipeline_v2.py's _fetch_enriched_jobs_pool, status='enriched' only)
-- for the entire _RE_ENRICH_COOLDOWN_DAYS window instead of disappearing from
-- it. src/enrich_jobs.py's selection query picks up status='enriched' rows
-- with this set, alongside status='pending_enrich' rows, subject to the same
-- cooldown. Cleared on a successful re-enrichment.
-- ---------------------------------------------------------------------------
alter table jobs add column if not exists reenrich_requested_at timestamptz;

-- One-time data fix: every job CURRENTLY stuck in the requeue loop Phase 0
-- identified (status='pending_enrich' despite already having a real
-- enrichment from some prior successful run — confirmed in production: 393
-- jobs had been through this cycle more than 4 times, 310 of them at exactly
-- 7 cycles, i.e. _RE_ENRICH_COOLDOWN_DAYS). Restoring status='enriched' here
-- makes them matchable again immediately; reenrich_requested_at=now() queues
-- them for a real re-enrich on the very next enrich_jobs.py run rather than
-- waiting out a stale cooldown clock that was never actually tracking a
-- pending request before this column existed.
update jobs
    set status = 'enriched', reenrich_requested_at = now()
    where status = 'pending_enrich' and enrichment is not null;

-- ---------------------------------------------------------------------------
-- 2. jobs.status — add 'closed' (src/closure.py). Report-only in Phase 1:
-- close_jobs() is implemented and tested but NOT yet called from
-- src/pipeline_v2.py's run() — see the Phase 1 report's dry-run counts and
-- sequencing caveat before wiring it in. Widening the CHECK constraint here
-- is still required even before that, though, so a manual/one-off close_jobs()
-- call (or a dashboard UPDATE) doesn't immediately violate it.
-- ---------------------------------------------------------------------------
alter table jobs drop constraint if exists jobs_status_check;
alter table jobs add constraint jobs_status_check
    check (status in ('pending_enrich', 'enriched', 'stale', 'closed'));

-- ---------------------------------------------------------------------------
-- 3. source_fetch_log — one row per adapter per scheduled pipeline_v2.py run
-- (src/pipeline_v2.py's adapter-fetch loop, via each adapter's new
-- FetchResult return shape — src/adapters/base.py). complete=false on a
-- failed adapter call, or when a source's own fetch logic (e.g. 80k's
-- fetch_raw_hits_bucketed()) detects it may have missed postings. Feeds
-- src/closure.py's rule (a): a source's missing postings are only closed on
-- the strength of a RECENT complete=true signal here, never assumed.
-- ---------------------------------------------------------------------------
create table if not exists source_fetch_log (
    id uuid primary key default gen_random_uuid(),
    run_id uuid references pipeline_runs(run_id),
    source_board text not null,
    fetched int not null,
    nb_hits int,
    complete boolean not null,
    fetched_at timestamptz not null default now()
);

create index if not exists source_fetch_log_source_board_idx on source_fetch_log (source_board);
create index if not exists source_fetch_log_fetched_at_idx on source_fetch_log (fetched_at);

-- ---------------------------------------------------------------------------
-- 4. run_metrics.enriched_without_embedding — src/enrich_jobs.py's stage='enrich'
-- insert now reports how many status='enriched' jobs are still missing a
-- Voyage embedding (and therefore invisible to match_v2_top_jobs's pgvector
-- rerank — see that function's `embedding is not null` filter). Surfaces the
-- gap Phase 0 found silently hiding 17 jobs from every user's matching.
-- ---------------------------------------------------------------------------
alter table run_metrics add column if not exists enriched_without_embedding int;
