create table public.experiment_assignments (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references public.profiles(user_id) on delete cascade,
  experiment  text not null,
  arm         text not null,
  assigned_at timestamptz not null default now(),
  unique (user_id, experiment),
  constraint experiment_assignments_arm_check check (
    experiment <> 'application_support_v1' or arm in ('full', 'shortlist_only')
  )
);

alter table public.experiment_assignments enable row level security;
grant select on public.experiment_assignments to authenticated;
create policy "read own assignments" on public.experiment_assignments
  for select to authenticated using (user_id = auth.uid());
-- no insert/update/delete for authenticated: only the trigger (security definer) and service role write

create or replace function public.assign_application_support_arm()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  n_full  int;
  n_short int;
  chosen  text;
begin
  if new.is_admin then
    return new;
  end if;

  if (tg_op = 'INSERT' and new.onboarding_completed)
     or (tg_op = 'UPDATE' and new.onboarding_completed and not old.onboarding_completed) then

    perform pg_advisory_xact_lock(hashtext('application_support_v1'));

    select count(*) filter (where arm = 'full'),
           count(*) filter (where arm = 'shortlist_only')
      into n_full, n_short
      from experiment_assignments
     where experiment = 'application_support_v1';

    chosen := case
      when n_full < n_short then 'full'
      when n_short < n_full then 'shortlist_only'
      when random() < 0.5   then 'full'
      else 'shortlist_only'
    end;

    insert into experiment_assignments (user_id, experiment, arm)
    values (new.user_id, 'application_support_v1', chosen)
    on conflict (user_id, experiment) do nothing;
  end if;

  return new;
end;
$$;

create trigger trg_assign_application_support_arm
after insert or update of onboarding_completed on public.profiles
for each row execute function public.assign_application_support_arm();
