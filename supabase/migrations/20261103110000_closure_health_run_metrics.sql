-- PASTE REQUIRED — not applied. src/closure.py's close_jobs() now writes a
-- closure_health key alongside closure_summary on its existing stage='closure'
-- run_metrics row (see that module's compute_closure_health()): a count of
-- open jobs that are only staying open because a board OTHER than their
-- nominal owner keeps touching last_seen, while the owning board itself
-- hasn't had a complete fetch in 14+ days. insert_run_metrics() never raises
-- (see src/supabase_client.py), so a run before this migration is applied
-- degrades to a logged error, not a crash — but the metric won't actually
-- persist until this is pasted in.
alter table run_metrics add column if not exists closure_health jsonb;
