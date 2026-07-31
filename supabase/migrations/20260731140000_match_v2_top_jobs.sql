-- True North v2: pgvector cosine-similarity rerank, run entirely in SQL (not
-- Python) — src/match_v2.py calls this via client.rpc(). Takes the candidate job
-- ids gate_v2 already filtered down to (hard-gate logic needs location_field/
-- enrichment, which isn't practical to replicate in SQL) and reranks just that
-- set by cosine similarity. `1 - (embedding <=> query_embedding)` is the standard
-- cosine-similarity idiom for pgvector's <=> distance operator regardless of
-- whether the stored vectors are pre-normalized.
--
-- No dedicated index needed at this scale: candidate_job_ids caps the input to
-- at most the size of one user's gate-survivor set (currently well under 1000
-- rows), so a full scan over that filtered set is trivial; jobs already has an
-- HNSW index on embedding from the original v2 schema migration if that ever
-- changes.
create or replace function match_v2_top_jobs(
  query_embedding vector(1024),
  candidate_job_ids uuid[],
  match_limit int default 25
)
returns table (
  id uuid,
  similarity float
)
language sql
stable
as $$
  select id, 1 - (embedding <=> query_embedding) as similarity
  from jobs
  where id = any(candidate_job_ids)
    and embedding is not null
  order by embedding <=> query_embedding
  limit match_limit;
$$;
