-- src/batch_poll.py's non-blocking poll/resume rework needs two new
-- batch_submissions.status values:
--
--   'claimed' — the atomic compare-and-swap concurrency guard between the
--     daily pipeline and the hourly drainer (src/drain_batches.py): whoever's
--     UPDATE ... WHERE status = 'submitted' actually matches a row owns that
--     batch until it releases the claim (back to 'submitted' if still
--     pending, or to 'retrieved' once applied). A 'claimed' row untouched
--     for longer than batch_poll._CLAIM_STALE_MINUTES is presumed abandoned
--     by a crashed process and becomes reclaimable again — no separate
--     schema state needed for that, just an updated_at age check.
--
--   'expired' — a submission neither retrieved nor even claimed within
--     SUBMISSION_EXPIRY_HOURS (24h, the Batches API's own completion
--     guarantee) is marked expired and freed up for resubmission, rather
--     than blocking enrichment on that job forever.
--
-- No new columns — updated_at (already present, already maintained by
-- record_batch_retrieved) is reused for the 'claimed' staleness check.
--
-- Idempotent, additive only, safe to re-run.
alter table batch_submissions
    drop constraint if exists batch_submissions_status_check;
alter table batch_submissions
    add constraint batch_submissions_status_check
        check (status in ('submitted', 'claimed', 'retrieved', 'expired'));
