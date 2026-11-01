-- True North v2: close_jobs() (src/closure.py) is now wired into
-- pipeline_v2.run() after ingest and writes one run_metrics row per run
-- (stage='closure') with the per-source/per-rule breakdown. Additive/
-- idempotent, same convention as every prior v2 migration. Applied by hand
-- via the Supabase dashboard (no Supabase CLI in this workflow).

alter table run_metrics add column if not exists closure_summary jsonb;

alter table run_metrics
    drop constraint if exists run_metrics_stage_check;
alter table run_metrics
    add constraint run_metrics_stage_check
        check (stage is null or stage in ('ingest', 'closure', 'enrich', 'tier1', 'tier2', 'match'));
