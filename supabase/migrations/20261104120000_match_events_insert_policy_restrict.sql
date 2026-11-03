-- DRAFT — NOT APPLIED. For dashboard-paste review only. Committed verbatim
-- as given — this policy change is the user's own, not drafted here.
--
-- Tightens match_events' INSERT policy: a client-side (authenticated,
-- source='ui') insert can no longer set attribution, and can only ever
-- insert source='ui' rows — distinguishing end-user-authenticated UI
-- events from the new track-click Edge Function's inserts, which use the
-- service-role key (bypasses RLS entirely, so this policy never applies
-- to it) and always write source='digest'.
drop policy "own events insert" on public.match_events;
create policy "own events insert" on public.match_events
  for insert to authenticated
  with check (user_id = auth.uid() and source = 'ui' and attribution is null);
