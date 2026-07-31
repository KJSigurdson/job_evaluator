-- True North v2: persist each job's raw location_field (as listed by the source
-- board) on `jobs` — needed by gate_v2's hybrid/onsite region-token matching and
-- by the Tier 1 enrichment prompt's location context. Additive, nullable, no
-- default; existing rows are backfilled by src/ingest.py's normal write path on
-- their next ingest, or via a one-off backfill script for rows already ingested
-- before this column existed.
alter table jobs add column if not exists location_field text;
