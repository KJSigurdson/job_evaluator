-- True North v2: widen the vector-rerank cutoff from top-25 to top-40 (three
-- of a real user's historically good matches were dropping out at the 25
-- cutoff before ever reaching Tier 1; Tier 1's cost delta is negligible since
-- it's already one Haiku call per user regardless of how many job cards are
-- in it). src/match_v2.py's vector_rerank() always passes match_limit
-- explicitly, so this default isn't load-bearing for the app itself — it only
-- matters if the function is ever called directly (dashboard, ad-hoc script)
-- without an explicit limit, and keeping it in sync avoids a surprising
-- mismatch between the Python-side constant and the SQL-side default.
create or replace function match_v2_top_jobs(
  query_embedding vector(1024),
  candidate_job_ids uuid[],
  match_limit int default 40
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
