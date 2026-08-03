-- run_metrics schema clarity: rows were stage-writes with no stage identity,
-- and enrich_jobs.py's Haiku call count was being written into tier1_calls
-- (conflating "enrich" stage volume with the actual Tier 1 rerank stage).
-- Additive/idempotent only, same convention as every other v2 migration —
-- IF NOT EXISTS everywhere, safe to re-run. Existing rows are left with NULL
-- run_id/stage/trigger/enrichment_calls; no backfill needed (check constraints
-- below only apply to non-null values, so old rows are unaffected).
--
-- run_id: shared by every stage row (ingest/enrich/tier1/tier2/match) written
--   by one src/pipeline_v2.py run() invocation — see that module's `run_id`.
-- stage: which pipeline stage wrote this row.
-- trigger: 'scheduled' (cron, no --only-user-id) or 'manual' (--only-user-id).
-- enrichment_calls: src/enrich_jobs.py's own Haiku call count, now written
--   under its own column instead of overloading tier1_calls.
alter table run_metrics
    add column if not exists run_id uuid,
    add column if not exists stage text,
    add column if not exists trigger text,
    add column if not exists enrichment_calls int;

alter table run_metrics
    drop constraint if exists run_metrics_stage_check;
alter table run_metrics
    add constraint run_metrics_stage_check
        check (stage is null or stage in ('ingest', 'enrich', 'tier1', 'tier2', 'match'));

alter table run_metrics
    drop constraint if exists run_metrics_trigger_check;
alter table run_metrics
    add constraint run_metrics_trigger_check
        check (trigger is null or trigger in ('scheduled', 'manual'));

create index if not exists run_metrics_run_id_idx on run_metrics (run_id);
create index if not exists run_metrics_stage_idx on run_metrics (stage);
