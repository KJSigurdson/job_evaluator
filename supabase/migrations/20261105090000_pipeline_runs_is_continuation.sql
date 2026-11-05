-- DRAFT — NOT APPLIED. For dashboard-paste review only.
--
-- Provenance for drainer-triggered continuation runs (daily.yml
-- mode=continue, src.pipeline_v2.run(skip_fetch=True)) — distinguishes a
-- continuation from a fresh daily cron tick without overloading
-- pipeline_runs.trigger (which stays 'scheduled' for a continuation run,
-- same as any full multi-user run — "how was this run started" and "is
-- this continuing an earlier incomplete run" are two different questions;
-- trigger likely has the same two-value CHECK constraint run_metrics.trigger
-- does, which a dedicated boolean sidesteps entirely rather than risking).
-- Set true only by _open_pipeline_run when skip_fetch=True; every other
-- run (including src/drain_batches.py's own trigger='drain' bookkeeping
-- rows) defaults to false.
alter table public.pipeline_runs
  add column is_continuation boolean not null default false;
