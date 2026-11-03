-- DRAFT — NOT APPLIED to THIS repo's tracked state, but confirmed already
-- applied live in production. Committed verbatim as given — see
-- 20261104100000_match_events_user_consents_baseline.sql for the
-- pre-change state this builds on.
alter table public.match_events drop constraint match_events_event_type_check;
alter table public.match_events add constraint match_events_event_type_check
  check (event_type in ('expanded', 'clicked_posting', 'status_changed', 'discarded',
                        'viewed_fit_rationale', 'viewed_cv_pointers'));

alter table public.match_events add column attribution text;
alter table public.match_events add constraint match_events_attribution_check
  check (attribution is null or attribution in ('tn_match', 'tn_adjacent', 'external', 'unknown'));
