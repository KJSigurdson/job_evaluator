-- True North v2 Phase 2: cache every Tier 1 score ever produced for a
-- (user, job) pair, above and below that user's insert_threshold, so a
-- re-run never re-spends an LLM call re-scoring a job/profile pair that
-- hasn't changed since it was last scored. See src/tier1_rerank.py's
-- fetch_valid_verdict_job_ids/_write_verdicts.
--
-- A verdict is valid only while BOTH job_enriched_at matches the job's
-- CURRENT jobs.enriched_at AND profile_content_hash matches the user's
-- CURRENT profiles.profile_content_hash — either one changing (new
-- enrichment content, or the user edited their profile) invalidates the
-- cached score and the job becomes a rescore candidate again next run.
--
-- job_enriched_at/profile_content_hash written here are REQUEST-TIME values
-- read back out of that batch's own manifest (batch_submissions.manifest,
-- see below) — not whatever `jobs`/`profiles` say at write time, which can
-- be hours later via the hourly drainer and after either one has changed.
--
-- rationale defaults to '{}' (not left NOT NULL with no default) and
-- profile_content_hash is nullable (mirrors profiles.profile_content_hash
-- itself, nullable with no default per 20260731130000_profiles_content_hash.sql)
-- so that one row missing either value can never fail the whole chunked
-- upsert — chunked_upsert's statement is one transaction; one bad row
-- would otherwise lose every other verdict in that chunk too.
create table if not exists tier1_verdicts (
    user_id uuid not null,
    job_id uuid not null references jobs(id) on delete cascade,
    score int not null,
    rationale jsonb not null default '{}'::jsonb,
    job_enriched_at timestamptz not null,
    profile_content_hash text,
    batch_id text,
    scored_at timestamptz not null default now(),
    primary key (user_id, job_id)
);

create index if not exists tier1_verdicts_user_score_idx on tier1_verdicts (user_id, score);

-- Phase 2 verdict-validity fix: persist, alongside a batch submission, the
-- exact request-time snapshot of what was sent (per custom_id: user_id,
-- profile_content_hash, {job_id: job_enriched_at}) — see
-- src.supabase_client.record_batch_submitted/fetch_batch_manifest and
-- src.tier1_rerank._build_manifest/_write_verdicts. Nullable: every module
-- except tier1_rerank submits without one.
alter table batch_submissions add column if not exists manifest jsonb;
