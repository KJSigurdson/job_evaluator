-- DRAFT — NOT APPLIED. For dashboard-paste review only.
--
-- Baseline migration for two tables that already exist live but had no
-- migration file in this repo (same gap as `profiles` before them):
-- match_events and user_consents. Reproduces the PRE-2026-11-04 state
-- exactly, as audited directly against production (information_schema.
-- columns, pg_get_constraintdef, pg_indexes, pg_policies, information_
-- schema.role_table_grants) — NOT reconstructed or improved in any way.
-- In particular: match_events.event_type's CHECK here is the ORIGINAL
-- four-value list; the attribution column and the six-value event_type
-- list are a LATER, separate change — see
-- 20261104110000_match_events_attribution.sql, confirmed applied after
-- this baseline.
--
-- user_consents.user_id deliberately has NO foreign key constraint below
-- — confirmed via the live constraint list (only a PRIMARY KEY and the
-- consent_key CHECK exist), unlike match_events.user_id, which does have
-- one. Reproduced as-is, not added, per "do not reconstruct or improve".
create table public.match_events (
    id uuid not null default gen_random_uuid() constraint match_events_pkey primary key,
    user_id uuid not null constraint match_events_user_id_fkey references public.profiles(user_id) on delete cascade,
    match_id uuid constraint match_events_match_id_fkey references public.matches_v2(id) on delete set null,
    event_type text not null,
    status text,
    source text not null,
    event_date date,
    details jsonb,
    created_at timestamptz not null default now(),
    constraint match_events_event_type_check check (
        event_type = any (array['expanded', 'clicked_posting', 'status_changed', 'discarded'])
    ),
    constraint match_events_source_check check (
        source = any (array['ui', 'digest', 'exit_survey', 'pulse_survey', 'admin'])
    ),
    constraint match_events_status_check check (
        status = any (array['applied', 'screening', 'interview', 'offer', 'hired', 'rejected', 'withdrawn'])
    ),
    constraint status_requires_type check (
        (event_type = 'status_changed') = (status is not null)
    )
);

create index match_events_user_idx on public.match_events using btree (user_id, created_at desc);
create index match_events_match_idx on public.match_events using btree (match_id);

alter table public.match_events enable row level security;

create policy "own events select" on public.match_events
    for select to authenticated using (user_id = auth.uid());
create policy "own events insert" on public.match_events
    for insert to authenticated with check (user_id = auth.uid());

grant all on table public.match_events to anon, authenticated;


create table public.user_consents (
    id uuid not null default gen_random_uuid() constraint user_consents_pkey primary key,
    user_id uuid not null,
    consent_key text not null,
    granted boolean not null,
    policy_version text not null,
    created_at timestamptz not null default now(),
    constraint user_consents_consent_key_check check (
        consent_key = any (array['data_storage', 'behavioral_tracking', 'third_party_aggregate_analytics'])
    )
);

create index user_consents_user_id_idx on public.user_consents using btree (user_id, created_at desc);

alter table public.user_consents enable row level security;

create policy "own consents select" on public.user_consents
    for select to authenticated using (user_id = auth.uid());
create policy "own consents insert" on public.user_consents
    for insert to authenticated with check (user_id = auth.uid());

grant all on table public.user_consents to anon, authenticated;
