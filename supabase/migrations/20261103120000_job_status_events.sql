-- DRAFT — NOT APPLIED. For dashboard-paste review only.
--
-- job_status_events: append-only history of jobs.status transitions.
-- Today, src/closure.py (close) and src/ingest.py (reopen) both overwrite
-- jobs.status/last_seen in place with no history kept — "why/when did this
-- job close, and whose fetch last confirmed it" is unanswerable after the
-- fact. Confirmed needed during the AAC go-live: jobs.closed_at doesn't
-- exist at all, and verifying the 4 jobs AAC reopened had been closed in
-- error required a manual per-job web-fetch check, not a DB query.
--
-- Purely additive observability — nothing reads this table yet, and
-- jobs.status remains the single source of truth for CURRENT state. Writing
-- to it (one INSERT alongside each status-changing UPDATE) is follow-up
-- work, not done by this migration.
create table if not exists job_status_events (
    id uuid primary key default gen_random_uuid(),
    job_id uuid not null references jobs(id) on delete cascade,
    -- The NEW status this event records, e.g. 'closed', 'enriched',
    -- 'pending_enrich' — same vocabulary as jobs.status, deliberately not
    -- constrained to its CHECK list here so a future status value doesn't
    -- need two migrations to land.
    status text not null,
    -- Which rule/path produced this transition — e.g. 'complete_source_stale'
    -- or 'safety_ceiling' (src/closure.py's two rules) on close,
    -- 'reopened_cross_board' / 'reopened_same_board' on reopen. Nullable:
    -- not every future writer of this table will have a rule name to give.
    reason text,
    -- Ties this event to the pipeline_v2 run that produced it, same FK every
    -- other per-run table in this schema uses. Nullable for the same reason
    -- run_metrics.run_id is: a standalone/manual invocation has no run_id.
    run_id uuid references pipeline_runs(run_id),
    created_at timestamptz not null default now()
);

create index if not exists job_status_events_job_id_idx on job_status_events (job_id);
create index if not exists job_status_events_created_at_idx on job_status_events (created_at);
