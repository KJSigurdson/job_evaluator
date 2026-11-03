-- DRAFT — NOT APPLIED. For dashboard-paste review only. Do NOT push the
-- accompanying code change until this has been pasted and run.
--
-- matches_v2.notified_at: per-match delivery tracking for the digest email.
-- Confirmed via information_schema that no notified_at (or equivalent)
-- column exists anywhere in this schema today — and pipeline_run_users'
-- own `notified` boolean (src/run_store.py, v1-only) is scoped to one
-- run_id, not one match, so it can't answer "was THIS match ever emailed"
-- across runs. That gap is exactly what let the 9 real Tier 2 matches from
-- run 56a07def-ab01-45e6-a217-cfd4c7d28d74 (Adam Holt, 2026-11-03 10:52:58
-- UTC) go unnotified forever: the pipeline only ever checked "is this job
-- new to matches_v2 THIS run", so once written, a match that failed to
-- send (or, in that run's case, never even attempted — MATCH_DIGEST_URL
-- wasn't configured in that process's environment) had no way to resurface
-- for a future run's digest.
alter table matches_v2 add column if not exists notified_at timestamptz null;

-- Backfill: every pre-existing row is assumed already delivered (notified_at
-- = created_at) EXCEPT the 9 real, confirmed-unsent rows from run
-- 56a07def-ab01-45e6-a217-cfd4c7d28d74 — those are left NULL on purpose so
-- src.pipeline_v2's new notified_at-IS-NULL digest query picks them up and
-- actually emails them on a future run, per the production decision made
-- 2026-11-03 to leave them in place rather than delete+reprocess.
update matches_v2
set notified_at = created_at
where id not in (
    '1ddf3604-8578-46b1-8b76-41571140ba86',
    '36f7a8d2-f14b-4802-8b45-24158f4648d5',
    '3d38dbbe-0b49-46b8-8294-4e981c647ee1',
    '409eec06-e8d1-4b0c-bd4f-a6ef5dd10e7a',
    '4797325c-73f6-4d5f-a750-88aa462f6c19',
    '62f9d7fa-4a16-4c3d-a7c7-092c4f1d5f92',
    '6d91e62d-b9ab-420d-af6a-9924ed2ade95',
    'adda409a-5015-4bfc-ad8a-05ef4fb6124d',
    'ef116a1a-1baf-4eb9-8e6b-6ac37bade183'
);

create index if not exists matches_v2_user_id_notified_at_idx
    on matches_v2 (user_id, notified_at);
