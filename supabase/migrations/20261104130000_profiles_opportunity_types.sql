-- DRAFT — NOT APPLIED. For dashboard-paste review only. Committed verbatim
-- as given.
--
-- profiles.opportunity_types (text[], PostingType vocabulary, null =
-- unanswered) and profiles.open_to_part_time (boolean, nullable) — read by
-- src.user_store._build_profile into the profile dict (top-level keys, not
-- nested under hard_constraints) and consumed by src.gate_v2.check():
-- opportunity_types replaces the global _DEFAULT_EXCLUDED_POSTING_TYPES
-- rule with this user's own exhaustive allow-list once answered;
-- open_to_part_time isn't read by the gate yet, LLM context only.
alter table public.profiles
  add column opportunity_types text[],
  add column open_to_part_time boolean;

alter table public.profiles add constraint profiles_opportunity_types_check
  check (opportunity_types is null or (
    cardinality(opportunity_types) > 0
    and opportunity_types <@ array['job','internship','fellowship','volunteer',
                                   'funding','program','speculative']::text[]
  ));
