# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A daily, serverless, **multi-user** pipeline (project name "True North") that ingests EA-aligned job boards (80,000 Hours, Probably Good, IAP referral doc) into a **persistent** Supabase `jobs` table, enriches each posting once with structured LLM output, and matches that shared pool against every registered user's profile embedding + hard constraints, upserting high-fit roles into `matches_v2` with LLM-generated reasoning. Deployed via GitHub Actions cron — no server. Sends a per-user match-digest email (reused unchanged from v1) and supports a cost-capped manual single-user run (`--only-user-id`).

**v2 (`src/pipeline_v2.py`) is the active production pipeline** — wired into `.github/workflows/daily.yml`'s "Run pipeline v2" step. **v1 (`src/pipeline.py`) is disabled** in that same workflow via `if: false` on its step (not deleted — one-line revert to roll back; see "v1 — legacy/disabled" below). v1's scrapers (`src/sources/*.py`) stay alive as the shared fetch code v2's adapters wrap unchanged.

## Commands

```bash
# Activate venv (Python 3.14)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the v2 pipeline (active path)
python -m src.pipeline_v2                              # scheduled/cron mode: scrape+ingest+enrich+match+digest
python -m src.pipeline_v2 --only-user-id <uuid>         # manual DB-only mode: match+digest for one user, no scraping/enrichment (~$0.03)
python -m src.pipeline_v2 --dry-run                     # compute gate/vector for real (free); skip Tier 1/Tier 2/writes

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_gate_v2.py -v
```

## Architecture — v2 (active)

```
src/
  adapters/                       # SourceAdapter wrappers — reuse src/sources/*.py's fetch code UNCHANGED, add a
                                   # second, independent parse step producing JobPosting (not v1's RawPosting)
    base.py                       # JobPosting dataclass + SourceAdapter ABC
    eighty_thousand_hours.py      # wraps src.sources.eightyk.fetch_raw_hits()
    probably_good.py              # wraps src.sources.probablygood.fetch_raw_hits()
    iap.py                        # wraps src.sources.iap.fetch_raw_rows()
  ingest.py                       # dedup/alias JobPostings into `jobs`/`job_aliases` (canonical_url exact match ->
                                   # update-in-place on content change; org/title fuzzy match -> job_aliases row,
                                   # not a new jobs row); writes run_metrics per-board counts
  models/
    job_enrichment.py             # JobEnrichment/WorkArrangement/SalaryRange — strict (extra="forbid") schema for
                                   # jobs.enrichment jsonb, incl. posting_type (job/internship/fellowship/volunteer/
                                   # funding/program/speculative)
    tier1_rerank.py                # JobScore/Tier1RerankOutput — strict schema for the Tier 1 rerank LLM output
  enrich_jobs.py                  # Haiku Batches API: JobEnrichment (incl. posting_type) + Voyage title+summary
                                   # embedding, for jobs.status='pending_enrich' only (new/content-changed jobs —
                                   # steady-state stays small since the pool persists across runs)
  profile_embed.py                # embeds profiles (career_goals/cause_priorities/experience/skills/values_notes —
                                   # deliberately NOT location/comp_needs, which gate_v2 already enforces), re-embeds
                                   # only on profile_content_hash change; only_user_id scopes to one user
  gate_v2.py                      # hard-constraint gate against a jobs row's structured enrichment (not raw text):
                                   # location (work_arrangement mode/remote_scope/eligible_regions, falling back to
                                   # location_field text for hybrid/onsite), seniority (enrichment.seniority buckets),
                                   # posting_type (excludes volunteer/funding by default, null treated as job)
  match_v2.py                     # gate_v2 survivors -> pgvector cosine rerank to TOP_N=40 via the match_v2_top_jobs()
                                   # SQL RPC (similarity computed in Postgres, not Python); fetch_matched_job_ids()
                                   # for the already-matched skip-set (see Data flow)
  tier1_rerank.py                 # one Haiku Batches call per user, scoring all TOP_N job cards 0-100 in one prompt;
                                   # gate constraints included as context only (explicit instruction not to re-apply)
  tier2_enrich.py                 # one Sonnet Batches call per (user,job) pair with tier1_score >= that user's OWN
                                   # scoring_weights.insert_threshold (0-1 in storage, scaled x100 to match
                                   # tier1_score's 0-100 range — see pipeline_v2._match_and_notify_user); no
                                   # near-miss tier — anything below is dropped, never written anywhere. Reuses v1's
                                   # exact Tier2EnrichmentOutput shape (org_summary/why_fits/why_not_fits/
                                   # emphasize_in_cv/deemphasize) so the frontend's existing renderer is untouched
  pipeline_v2.py                  # orchestration — see Data flow below
  supabase_client.py              # get_client(); fetch_all() (paginates past PostgREST's 1000-row response cap —
                                   # a plain .execute() silently truncates, confirmed for real); chunked_upsert()/
                                   # chunked_insert() (500-row chunks — an oversized single upsert hit a Postgres
                                   # statement timeout for real, losing the whole batch atomically); batch lifecycle:
                                   # record_batch_submitted/record_batch_retrieved/find_outstanding_batch/
                                   # persist_batch_results (crash-safety, see Reliability requirements)
  notify.py                       # send_match_digest() — REUSED UNCHANGED from v1, same Edge Function/payload contract
  quota_store.py                  # set_status() — REUSED UNCHANGED from v1, same search_quota semantics
  user_store.py                   # fetch_users() — REUSED UNCHANGED from v1 (profiles+scoring_weights -> profile/rubric dicts)
tests/
supabase/migrations/              # SQL applied by hand via the Supabase dashboard (no Supabase CLI in this workflow)
.github/workflows/daily.yml       # v1 step: if: false (disabled). v2 step: active, timeout-minutes: 350.
```

## Architecture — v1 (legacy/disabled)

`src/pipeline.py` and everything it alone depends on (`src/gate.py`, `src/scoring.py`, `src/enrich.py`, `src/matches_store.py`, `src/seen_store.py`, `src/run_store.py`) are untouched, fully functional, and still tested — just not invoked by the cron anymore. `src/sources/*.py` (`eightyk.py`, `probablygood.py`, `iap.py`) are shared: v1 calls them directly, v2's adapters wrap the same `fetch_raw_hits()`/`fetch_raw_rows()` functions unchanged. No v1 table (`matches`, `seen`, `pipeline_runs`, `pipeline_run_users`) is read or written by any v2 module. Rollback is a one-line edit to `daily.yml`: remove `if: false` from v1's "Run pipeline" step, add it to v2's "Run pipeline v2" step instead.

### Data flow — v2

**Scheduled/cron mode** (`python -m src.pipeline_v2`, no `--only-user-id`):
1. Adapters (80k, ProbablyGood, IAP) fetch once, each wrapped in its own try/except (graceful per-source failure) → `ingest.ingest()` dedupes/aliases into `jobs`/`job_aliases`.
2. `enrich_jobs.enrich_jobs()` — only `status='pending_enrich'` rows (new or content-changed since last ingest); Haiku Batches call, JobEnrichment + Voyage embedding.
3. `profile_embed.backfill_profile_embeddings()` — only users whose rendered embedding text hash changed since last embed.
4. Per active user with a non-empty profile and a `profile_embedding` (both checked; missing either skips that user with a logged reason, never crashes the run): fetch that user's already-matched `job_id`s (`match_v2.fetch_matched_job_ids`) and exclude them from the candidate pool **before** gating — without this, a user's existing matches would keep winning top-40 vector-rerank slots indefinitely, permanently starving new jobs from ever reaching Tier 1, on top of re-spending Tier 2 and re-emailing the same jobs every run. Then: `gate_v2.check` survivors → `match_v2.vector_rerank` top-40 → `tier1_rerank.rerank_tier1` (one Haiku call, all 40 cards) → `tier2_enrich.tier2_enrich` (one Sonnet call per pair with tier1_score ≥ that user's own `scoring_weights.insert_threshold` ×100 — no near-miss tier, anything below is dropped, never written anywhere) → rows land in `matches_v2` (`tier2_enrich`'s own write, `upsert` on `(user_id, job_id)`).
5. Digest: every `matches_v2` row written this run is new by construction (step 4's exclusion already filtered out anything pre-existing) — mapped into v1's exact digest payload shape (`{title, org, location, fit_score, why_fits, url}`) and sent via `notify.send_match_digest()`, unchanged contract, only if `profile["email_on_match"]` and not `--dry-run`.

**Manual DB-only mode** (`--only-user-id X`): skips steps 1–3 entirely — no scraping, no job enrichment, no job embedding. Refreshes just that user's own profile embedding if changed (free when unchanged), then steps 4–5 for that user only. Wrapped in the same `running`→`complete`/`failed` `search_quota` bookkeeping v1's `--only-user-id` path already used (`quota_store.set_status`, unchanged). Deliberate cost cap confirmed in production: ~$0.03–0.05/run vs. a full scheduled run's ~$0.90+.

`--dry-run` still computes `gate_v2`/vector rerank for real (free — no LLM cost) but skips Tier 1/Tier 2 and every write; `ingest`/`enrich_jobs` have their own native dry-run (compute-without-writing).

### Critical invariant: matches_v2 upsert never touches user-owned columns

Same principle as v1's `matches` table: `tier2_enrich.py`'s upsert payload is `{user_id, job_id, tier1_score, tier2, created_at}` — structurally no `status`/`user_notes`/`discarded` fields. The already-matched skip-set (step 4 above) means an existing `matches_v2` row is never re-processed, let alone re-upserted, by a later run.

## Secrets

Required in `.env` / GitHub Actions secrets:
- `ANTHROPIC_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `VOYAGE_API_KEY` — job/profile embeddings (voyage-3.5). No `voyageai` package; `enrich_jobs.embed_texts`/`profile_embed` call Voyage's REST API directly via `httpx` to avoid an unused dependency. Unset = embedding step skipped, not an error (embeddings backfill later once the key exists).

Optional (degrade gracefully when unset — see `.env.example`):
- `MATCH_DIGEST_URL` / `PIPELINE_SHARED_SECRET` — match-digest email, same contract as v1. **Currently returning 401 Unauthorized in production** (confirmed pre-existing, not caused by the v2 cutover — v1's last scheduled run before the cutover hit the identical error on the identical endpoint) — needs the Edge Function/secret checked outside this pipeline's code.

## LLM usage

- **Enrichment (`enrich_jobs.py`):** Haiku, structured JSON tool output, one Batches API call per pending job. Validated against `JobEnrichment` (strict). On validation failure: one retry with the error appended to the prompt, then `status='stale'` — never partial.
- **Tier 1 (`tier1_rerank.py`):** Haiku, one Batches API call per user, scoring all of that user's top-40 job cards 0–100 in a single prompt (not per-job — amortizes the static prompt prefix the same way v1's batched Tier 1 did).
- **Tier 2 (`tier2_enrich.py`):** Sonnet, one Batches API call per (user, job) pair above the Tier 1 threshold. Reuses v1's exact `Tier2EnrichmentOutput` shape and the same defensive `emphasize_in_cv`/`deemphasize` list-coercion.
- All three stages use the **Anthropic Message Batches API** (not synchronous calls) — cron-appropriate, ~50% of synchronous pricing, but each call polls up to `_POLL_TIMEOUT_SECONDS` (3600s) and a batch can legitimately take up to 24h server-side — see Reliability requirements below for how this is made crash-safe.
- All LLM outputs validated against Pydantic schemas (strict, `extra="forbid"`). On parse failure: retry once (enrichment/Tier 1/Tier 2 each independently), then log-and-skip — never crash the run.

## Reliability requirements

- **Idempotency:** re-running the same day → no duplicate `jobs`/`matches_v2` rows (upsert on canonical_url / (user_id,job_id)), no re-enrichment of unchanged content, no re-matching of already-matched jobs.
- **Graceful per-source failure:** one adapter erroring must not abort the others (each wrapped independently in `pipeline_v2.py`).
- **Graceful per-user failure:** a user with no `scoring_weights` row, an empty profile, or no `profile_embedding` yet is skipped with a logged reason — never crashes the run.
- **Pagination:** any Supabase read that could plausibly exceed 1000 rows MUST go through `supabase_client.fetch_all()`, not a plain `.execute()` — PostgREST silently truncates at `db.max_rows` with no error. Confirmed for real once `jobs` crossed 1000 rows.
- **Bulk-write chunking:** any bulk `.upsert()`/`.insert()` that could plausibly exceed a few hundred rows MUST go through `chunked_upsert()`/`chunked_insert()` (500-row chunks) — an oversized single statement hit a real Postgres statement timeout, losing the *entire* batch atomically (not just rows past some cutoff), including already-paid-for LLM output.
- **`.in_()` filter chunking:** PostgREST encodes every filter — `.in_()` included — in the request URL's query string, for GET *and* for PATCH/UPDATE alike (only the row payload goes in the body). An unbounded id/user_id list therefore makes an unbounded URL, which fails as a **proxy-level 5xx (e.g. Cloudflare 520), not a clear PostgREST/Postgres error** — there is no helpful stack trace pointing at the real cause. Confirmed for real: `closure.close_jobs()` re-fetched 1,665 job_ids via a single `.in_("id", job_ids)` (~65KB GET URL) and got a 520; ingest had already completed, so closure silently wrote nothing and enrich/matching never ran that day. Any `.in_()` call whose list can grow with the data (job backlogs, a user's accumulated verdicts, registered users) MUST go through `supabase_client.fetch_all_in()`, which chunks at `_IN_FILTER_CHUNK_SIZE` (100 — deliberately much smaller than `_WRITE_CHUNK_SIZE`'s 500, since that constant bounds POST-body size/statement-timeout risk, a different limit entirely, not URL length).
- **Batch crash-safety** (the core reliability design of this pipeline — confirmed to have actually lost ~$2.65 of paid Anthropic output before this was built): each Batches-API module (`enrich_jobs.py`/`tier1_rerank.py`/`tier2_enrich.py`) tracks its batch through `batch_submissions` (`submitted`→`retrieved`) and stages every raw result to `batch_results` *immediately on retrieval*, before any parsing or final write. `run_metrics` is written right after usage is tallied — **before** the final table write — so a crash in that write no longer loses the cost record. `find_outstanding_batch()` is checked before submitting new work, so a run that resumes after a runner timeout/crash mid-poll resumes the same in-flight (already-paid-for) batch instead of duplicating it. `reprocess_staged(client, batch_id)` (one per module) replays unprocessed staged results without any new API call — response parsing operates on plain dicts (`result.model_dump(mode="json")`, staged immediately) specifically so the live path and the recovery-replay path share identical code, no dual-shape handling.
- **Batches API `custom_id` constraints — length AND charset, and the fake client enforces neither:** confirmed for real (August) that custom_id is capped at 64 characters — `tier2_enrich.py`'s original `f"{user_id}:{job_id}"` design (two joined UUIDs, 73 chars) got "String should have at most 64 characters" back from the API, which is why that module's custom_id is a plain anonymous enumeration index instead (see its module docstring). Confirmed for real AGAIN later that length isn't the only constraint: custom_id must also match `^[a-zA-Z0-9_-]{1,64}$` — `tier1_rerank.py`'s `f"{user_id}:{chunk_index}"` put a `:` in every id, well under 64 chars, and the Batches API still rejected the ENTIRE `.create()` call with a 400 `invalid_request_error` (not just the one bad request). `tests/conftest.py`'s `FakeSupabaseClient`/fake Anthropic clients enforce NEITHER the length nor the charset — both failures only ever surface against the real API. Every Batches module (`enrich_jobs.py`/`tier1_rerank.py`/`tier2_enrich.py`) now calls `batch_poll.validate_custom_id()` inside its own `_batch_request()` (the one chokepoint both the live and retry paths already funnel through) so a bad id raises locally, at request-build time, before spending a network round trip — see `tests/test_batch_poll.py`'s `test_validate_custom_id_*`.
- **Match-digest email:** same never-raise contract as v1 — `notify.send_match_digest()` never raises, and `pipeline_v2.py`'s call site wraps it in try/except as a second layer of defense.
- **Already-matched skip-set:** `match_v2.fetch_matched_job_ids()` must be applied before `gate_v2`/vector rerank in every per-user matching call — omitting this both wastes Tier 2 spend on already-matched jobs and, since vector rerank is capped at `TOP_N`, can permanently starve new jobs from ever reaching Tier 1 for an active user with several existing matches.
- **`run_metrics` writes must never abort a run:** every `run_metrics` insert goes through `supabase_client.insert_run_metrics()` (never raises — logs the full payload at ERROR on failure and returns), not a raw `.table("run_metrics").insert().execute()`. Confirmed for real: `closure.close_jobs()` closed 1,665 jobs for real (every chunked upsert 200 OK), then its own run_metrics insert 400'd with PGRST204 — a migration that existed in the repo but hadn't actually been applied to the live schema — and that exception aborted the run before enrich/matching ever started, even though the closure work itself had already fully committed. Observability must never be able to undo or block on real work that already succeeded. This applies to run_metrics specifically because it's the only *pure*-metrics table in this codebase — `batch_submissions`/`batch_results`/`tier1_verdicts`/`matches_v2`/`jobs`/`job_aliases` are real pipeline state (losing those writes silently would be a correctness bug, not a telemetry gap) and `source_fetch_log` feeds real downstream decisions (closure's rule (a), the loud source-health check) — none of those should get this treatment.
- **Schema drift — a new column written by code needs a migration in the SAME change, and the fake test client won't catch a missing one:** `tests/conftest.py`'s `FakeSupabaseClient` has no real schema — it happily accepts `.insert()`/`.upsert()` with any keys at all, so a column the live Postgres schema doesn't have yet will pass every test and 400 with `PGRST204` ("Could not find the '<col>' column") only against production. When adding a column a write touches, write the migration in the same commit, and verify it was actually APPLIED (not just committed to `supabase/migrations/`) before relying on it in a scheduled run — a migration file existing in the repo is not evidence it was pasted into the dashboard. `tests/conftest.py`'s `_FakeTable.raise_on_execute` (op → exception) exists specifically so a test CAN simulate this failure mode deliberately when it matters (e.g. verifying `insert_run_metrics()`'s never-raise contract) — the fake still won't catch an unannounced drift on its own.
- **Standing rule — reporting migrations:** every report to the user that includes a new or changed file under `supabase/migrations/` MUST list each such file and include its full SQL under a "PASTE REQUIRED" heading — even if the file was committed in an earlier change (the closure_summary incident happened specifically because a migration was committed once, flagged once, and then never re-surfaced; by the time it mattered, "did I actually paste this?" had no clear answer). A migration committed to the repo is not evidence it was applied to the live schema — only a "PASTE REQUIRED" callout that the user acts on (or an explicit live-schema check, e.g. a `.select()` probe against production) closes that gap.
- **Unit tests are required on:** `gate_v2` logic (location/seniority/posting_type, rejection_reason categorisation), `match_v2` (gate filtering, vector-rerank RPC wiring, already-matched exclusion), Tier 1/Tier 2 batch parsing (dict-based, shared between live and recovery paths), the full batch lifecycle (submission tracking, resumption, staging, `reprocess_staged`), `fetch_all`/`fetch_all_in`/`chunked_upsert`/`chunked_insert`/`insert_run_metrics`/`validate_custom_id`, `pipeline_v2.py`'s scheduled vs. manual-mode wiring and digest-payload mapping, and the `posting_type` exclusion default.
