# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A daily, serverless, **multi-user** pipeline that scrapes EA-aligned job boards (80,000 Hours, Probably Good, IAP referral doc) ONCE per run, then scores that shared pool against every registered user's profile and weights (both read from Supabase), and upserts high-fit roles into a Supabase `matches` table per user with LLM-generated reasoning and CV guidance. Deployed via GitHub Actions cron — no server. Optionally sends a per-user match-digest email and can be scoped to a single user for a one-off request (`--only-user-id`).

## Commands

```bash
# Activate venv (Python 3.14)
source .venv/bin/activate

# Install dependencies (once requirements.txt exists)
pip install -r requirements.txt

# Run the pipeline
python -m src.pipeline

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_gate.py -v
```

## Architecture

```
src/
  sources/            # one module per source; all return List[RawPosting]
  gate.py             # hard-constraint logic (location + seniority binary pass/fail) — unchanged per-user, takes a profile dict; tags rejection_reason for diagnostics only
  scoring.py          # Tier 1: cheap rubric scoring via LLM structured output — unchanged per-user, takes profile+rubric dicts
  enrich.py           # Tier 2: org summary + CV guidance (only for fit >= user's insert_threshold); defensively coerces emphasize_in_cv/deemphasize back to list[str] if a model returns a JSON-array-in-a-string
  supabase_client.py  # get_client() from SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (service-role, bypasses RLS)
  user_store.py       # fetch_users(only_user_id=None): reads `profiles` + `scoring_weights` (+ `experiences`/`experience_achievements`, incl. `publication` kind), builds the profile/rubric dicts gate/scoring/enrich expect; parses location_rule/seniority_rule from JSON text
  seen_store.py       # per-user seen-cache backed by the Supabase `seen` table; writes are batched (record_verdict queues in memory, upsert_verdicts flushes once per user in chunks of 500)
  matches_store.py    # query-existing-URLs + upsert-only against Supabase `matches` (never touches status/user_notes/discarded)
  quota_store.py      # set_status(): running/complete/failed write-back to `search_quota` — one-off single-user runs only
  notify.py           # send_match_digest(): best-effort opt-in match-digest email via a Supabase Edge Function; never raises
  schemas.py          # Pydantic models for all LLM outputs and internal types
  pipeline.py         # orchestration: fetch users → scrape ONCE (shared) → per-user: dedup → gate → score → enrich → upsert → batched seen flush → digest email
tests/
.github/workflows/daily.yml
```

### Data flow

1. Fetch every user with a `profiles` row AND a `scoring_weights` row (skip profile-only rows). `--only-user-id X` scopes every query to a single user X instead — a one-off request, not the normal daily run.
2. Scrape all sources ONCE → shared `List[RawPosting]` pool
3. Shared recency filter: drop postings older than `RECENCY_DAYS`. Postings with no `posted_at` are exempt UNLESS they also have no `deadline` (e.g. IAP) — those instead get a 14-day cutoff anchored to the *earliest* `first_seen` for that URL across all users in the `seen` table (see `pipeline.py` docstring for the full reasoning)
4. `--limit N` shuffles the fresh pool (seeded on `run_id`, reproducible per run) before slicing to N, so a capped run draws roughly proportionally across sources instead of always favoring whichever source was scraped first
5. For EACH user: build a skip-set (their `seen` rows ∪ their existing `matches` URLs), drop already-seen postings, then:
   - Hard-gate: binary pass/fail on location + seniority — **bias toward false-positives; when a field is unstated, pass it through**. Every rejection is tagged with `rejection_reason` (`location` / `seniority` / `hard_constraints` for both) — diagnostic only, does not change pass/fail. Aggregated into a `Counter` and logged as one summary line per user after the gate loop.
   - Tier 1 scoring: 7 soft dimensions (0–1 each), weighted by that user's `scoring_weights`, → fit %
   - Roles ≥ user's `insert_threshold` → Tier 2 enrichment → upsert into `matches`
   - Roles between `near_miss_floor` and `insert_threshold` → queued for `seen` as `below_threshold` (not inserted)
   - Every terminal verdict (`gated_out` / `below_threshold` / `inserted`) is queued in memory and flushed to `seen` for that user in ONE batched upsert (chunked at 500 rows) after that user's Tier 2 loop — not one write per posting. Parse failures are NEVER queued — retried next run.
   - If `profile["email_on_match"]` is true and ≥1 match was inserted this run and it's not `--dry-run`: send one match-digest email for that user (best-effort — see notify.py, never affects run outcome).
6. `--only-user-id` (when not `--dry-run`) writes `running` → `complete`/`failed` to `search_quota` for that user around the whole run — the normal multi-user run never touches that table.

### Critical invariant: matches upsert never touches user-owned columns

`matches_store.py` only ever writes job fields + model output (see `MatchRow` in `schemas.py`, which structurally has no `status`/`user_notes`/`discarded` fields). If a URL already exists in a user's `matches`, the per-user skip-set filters it out before scoring even runs, so it's never re-upserted — this preserves user-edited `status` (Draft/Applied/etc.) across runs.

## Secrets

Required in `.env` / GitHub Actions secrets:
- `ANTHROPIC_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `GOOGLE_SERVICE_ACCOUNT_JSON`

Optional (degrade gracefully when unset — see `.env.example`):
- `MATCH_DIGEST_URL` / `PIPELINE_SHARED_SECRET` — match-digest email. Unset = digest sending silently skipped (logged at info), never an error.

## LLM usage

- **Tier 1 (scoring):** structured JSON output, low temperature, pinned model. Runs on all hard-gate survivors.
- **Tier 2 (enrichment):** full profile passed in context; generates org summary, why-fit, why-not-fit, emphasize-in-CV, de-emphasize (defensively coerced back to `list[str]` if a model returns a JSON-array-in-a-string). Runs only on fit ≥ that user's `insert_threshold`.
- Use Claude as an **HTML parser** for sources that don't expose a JSON endpoint (feed page text, extract postings to `RawPosting` schema). Prefer JSON endpoints first.
- All LLM outputs must be validated against Pydantic schemas. On parse failure: retry once, then log-and-skip (never crash the run).

## Reliability requirements

- Idempotency: re-running the same day → no new `matches` rows, no modified rows, no duplicate `seen` entries (all writes upsert on their unique key, whether the seen-table upsert is a single row or a batched chunk).
- Graceful per-source failure: one source erroring must not abort the others.
- Graceful per-user failure: a missing `scoring_weights` row skips that user (logged) without aborting the run.
- Match-digest email is strictly best-effort: `notify.send_match_digest()` must never raise (missing config / network error / non-2xx all log-and-swallow), and its call site in `pipeline.py` wraps it in try/except as a second layer of defense. An email failure must never fail the run, flip `search_quota` to `'failed'`, or stop processing of subsequent users.
- Log every run: shared counts (scraped / recency-dropped / stale-dropped), per-user counts (gated-out / scored / inserted / near-miss / parse-failures), per-user gate-rejection breakdown (location / seniority / hard_constraints), per-source success, pinned model string, temperature.
- Unit tests are required on: hard-gate logic (incl. rejection_reason categorisation), dedup/URL canonicalization, weighted-sum scoring, Pydantic schema validation, per-user seen-cache logic (incl. batching/chunking), matches upsert column allowlist, match-digest never-raise contract.
