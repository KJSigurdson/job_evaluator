-- run_metrics.enrichment_calls: src/enrich_jobs.py's stage='enrich' insert has
-- written this key since the run_id/stage/trigger instrumentation pass (see
-- 20260803120000_run_metrics_stage_columns.sql, which specified this same
-- column but evidently wasn't fully applied — run_id/stage/trigger exist live,
-- enrichment_calls does not). Until this exists, the next real enrich_jobs.py
-- run's run_metrics insert 400s (PostgREST rejects unknown insert columns).
--
-- Audited every run_metrics insert call site in src/ (ingest.py, enrich_jobs.py,
-- tier1_rerank.py, tier2_enrich.py, pipeline_v2.py) against the live schema —
-- this is the only column any of them writes that the table doesn't already have.
--
-- Idempotent, additive only, safe to re-run.
alter table run_metrics
    add column if not exists enrichment_calls int;
