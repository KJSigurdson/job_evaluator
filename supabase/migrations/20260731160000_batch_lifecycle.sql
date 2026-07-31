-- True North v2 crash-safety: track Anthropic Batches API calls through their
-- full lifecycle (submitted -> retrieved -> processed), so a crash anywhere
-- after submission never silently loses paid-for LLM output or its cost
-- record. See src/supabase_client.py's record_batch_submitted/
-- record_batch_retrieved/persist_batch_results/find_outstanding_batch.
--
-- batch_submissions: one row per Anthropic Batches API call. status='submitted'
-- right after .create(); 'retrieved' right after .results() succeeds. Each
-- orchestrator (enrich_jobs.py/tier1_rerank.py/tier2_enrich.py) checks for an
-- outstanding 'submitted' row for its own module before submitting new work,
-- so a run resuming after a timed-out prior run doesn't duplicate an
-- already-paid-for in-flight batch.
create table if not exists batch_submissions (
    batch_id text primary key,
    module text not null,           -- 'enrich_jobs' | 'tier1_rerank' | 'tier2_enrich'
    status text not null default 'submitted' check (status in ('submitted', 'retrieved')),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists batch_submissions_module_status_idx on batch_submissions (module, status);

-- batch_results: one row per (batch_id, custom_id) — the raw, unparsed
-- Anthropic result, persisted immediately on retrieval, before any
-- parsing/validation/DB-write risk. processed=false until that module's own
-- parse+write logic (live or via reprocess_staged() recovery) finishes with
-- it, success or terminal failure alike.
create table if not exists batch_results (
    id uuid primary key default gen_random_uuid(),
    batch_id text not null references batch_submissions(batch_id) on delete cascade,
    custom_id text not null,
    raw jsonb not null,
    processed boolean not null default false,
    created_at timestamptz not null default now(),
    unique (batch_id, custom_id)
);

create index if not exists batch_results_unprocessed_idx on batch_results (batch_id) where not processed;
