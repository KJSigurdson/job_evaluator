# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A daily, serverless pipeline that scrapes EA-aligned job boards (80,000 Hours, Probably Good, IAP referral doc), scores new roles against a version-controlled personal profile, and inserts high-fit roles into a Notion application tracker with LLM-generated reasoning and CV guidance. Deployed via GitHub Actions cron — no server.

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
profile.yaml          # personal profile-as-code (read at runtime, no API call)
rubric.yaml           # soft-dimension weights + fit threshold (default 0.75)
src/
  sources/            # one module per source; all return List[RawPosting]
  gate.py             # hard-constraint logic (location + seniority binary pass/fail)
  scoring.py          # Tier 1: cheap rubric scoring via LLM structured output
  enrich.py           # Tier 2: org summary + CV guidance (only for fit >= 0.75)
  notion_client.py    # query-existing-URLs + insert-only (never modifies existing rows)
  schemas.py          # Pydantic models for all LLM outputs and internal types
  pipeline.py         # orchestration: fetch state → scrape → dedup → gate → score → insert
tests/
.github/workflows/daily.yml
```

### Data flow

1. Query Notion for all existing role URLs (dedup source of truth)
2. Scrape sources → `List[RawPosting]`
3. Drop URLs already in Notion (canonicalize: strip tracking params or hash `org+title`)
4. Hard-gate: binary pass/fail on location + seniority — **bias toward false-positives; when a field is unstated, pass it through**
5. Tier 1 scoring: 7 soft dimensions (0–1 each), weighted sum → fit %
6. Roles ≥ 0.75 → Tier 2 enrichment → Notion INSERT
7. Roles 0.65–0.74 → log as near-miss (do not insert)

### Critical invariant: insert-only

`notion_client.py` must NEVER modify or overwrite an existing Notion row. If a URL already exists in Notion, skip it unconditionally — regardless of what fields differ. This preserves user-edited Status (Draft/Applied/etc.) across runs.

## Secrets

Required in `.env` / GitHub Actions secrets:
- `ANTHROPIC_API_KEY`
- `NOTION_TOKEN`
- `NOTION_DB_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON`

## LLM usage

- **Tier 1 (scoring):** structured JSON output, low temperature, pinned model. Runs on all hard-gate survivors.
- **Tier 2 (enrichment):** full profile passed in context; generates org summary, why-fit, why-not-fit, emphasize-in-CV, de-emphasize. Runs only on fit ≥ 0.75.
- Use Claude as an **HTML parser** for sources that don't expose a JSON endpoint (feed page text, extract postings to `RawPosting` schema). Prefer JSON endpoints first.
- All LLM outputs must be validated against Pydantic schemas. On parse failure: retry once, then log-and-skip (never crash the run).

## Reliability requirements

- Idempotency: re-running the same day → no new rows, no modified rows.
- Graceful per-source failure: one source erroring must not abort the others.
- Log every run: counts (scraped / new / gated-out / scored / inserted), per-source success, parse failures, pinned model string, temperature.
- Unit tests are required on: hard-gate logic, dedup/URL canonicalization, weighted-sum scoring, Pydantic schema validation.

## Build order

Follow this sequence — get reliability skeleton green on stubs before wiring real APIs:

1. Scaffold structure, `schemas.py`, `profile.yaml`, `rubric.yaml`, `.env.example`
2. `gate.py`, `scoring.py`, dedup, insert-only `notion_client.py` — tests passing on stub postings
3. Wire IAP Google Doc source end-to-end (smallest, most structured)
4. Add 80k Hours + Probably Good sources (JSON endpoint first, HTML+LLM fallback)
5. Tier 2 enrichment
6. GitHub Actions workflow + near-miss logging
