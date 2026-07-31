-- True North v2: shared job enrichment architecture (jobs / job_aliases /
-- run_metrics), plus a profile_embedding column on the existing profiles table.
--
-- ADDITIVE ONLY. Does not modify, drop, or rename any existing table or column
-- (profiles, scoring_weights, matches, seen, experiences, experience_achievements,
-- search_quota, pipeline_runs, pipeline_run_users). The v1 pipeline (src/pipeline.py)
-- reads/writes none of the objects created here and is unaffected.
--
-- Every statement is idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS) so this
-- migration is safe to re-run if a partial apply is interrupted.

-- 1. pgvector extension, for the embedding columns below.
create extension if not exists vector;

-- 2. jobs — one row per canonical job posting in the v2 shared enrichment pipeline.
create table if not exists jobs (
    id uuid primary key default gen_random_uuid(),
    canonical_url text not null unique,
    source_board text not null,
    org text,
    title text,
    content_hash text not null,
    raw_description text,
    enrichment jsonb,
    embedding vector(1024),
    first_seen timestamptz not null default now(),
    last_seen timestamptz not null default now(),
    enriched_at timestamptz,
    status text not null default 'pending_enrich'
        check (status in ('pending_enrich', 'enriched', 'stale'))
);

-- HNSW index for approximate-nearest-neighbor search over `embedding`.
create index if not exists jobs_embedding_hnsw_idx
    on jobs using hnsw (embedding vector_cosine_ops);

create index if not exists jobs_status_idx on jobs (status);
create index if not exists jobs_source_board_idx on jobs (source_board);

-- 3. job_aliases — a job posting can be cross-listed under multiple URLs across
-- source boards; each alias points back to the one canonical `jobs` row.
-- alias_url is the primary key: it's already required to be unique and not-null,
-- and every other table in this migration has an explicit primary key, so this
-- promotes that constraint to PK rather than leaving the table with none.
create table if not exists job_aliases (
    alias_url text primary key,
    job_id uuid not null references jobs(id) on delete cascade,
    source_board text not null
);

create index if not exists job_aliases_job_id_idx on job_aliases (job_id);

-- 4. run_metrics — one row per v2 pipeline run, for cost/volume observability
-- (distinct from the existing v1 pipeline_runs/pipeline_run_users tables, which
-- track the OLD per-user scoring pipeline and are untouched by this migration).
create table if not exists run_metrics (
    id uuid primary key default gen_random_uuid(),
    run_at timestamptz not null default now(),
    jobs_ingested int,
    jobs_enriched int,
    tier1_calls int,
    tier2_calls int,
    estimated_cost_usd numeric,
    per_board jsonb,       -- ingest counts per source board, e.g. {"80k": 120, "iap": 40}
    gate_rejections jsonb  -- per-rule rejection counts per user, e.g. {"<user_id>": {"location": 12}}
);

-- 5. profile_embedding — semantic embedding of a user's profile, for the v2
-- similarity-based matching path. Additive column on the existing profiles table;
-- every existing column, row, and consumer of `profiles` is unaffected.
alter table profiles add column if not exists profile_embedding vector(1024);
