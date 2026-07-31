-- True North v2 Tier 2 output — shadow mode: NOT src.matches_store.py's v1
-- `matches` table (that cutover is a later, deliberate step). One row per
-- (user, job) pair whose Tier 1 rerank score cleared the enrichment threshold;
-- src/tier2_enrich.py upserts on (user_id, job_id), so a re-run never
-- duplicates a row for the same pair.
create table if not exists matches_v2 (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    job_id uuid not null references jobs(id) on delete cascade,
    tier1_score int not null,
    tier2 jsonb not null,  -- Tier2EnrichmentOutput: org_summary/why_fits/why_not_fits/emphasize_in_cv/deemphasize
    created_at timestamptz not null default now(),
    unique (user_id, job_id)
);

create index if not exists matches_v2_user_id_idx on matches_v2 (user_id);
