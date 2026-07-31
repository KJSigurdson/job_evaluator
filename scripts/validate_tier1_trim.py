"""Validation: does the Tier 1 profile trim (scoring._tier1_profile_view) change any
fit_score? Scores the same sample of real postings for one real user TWICE — once
against the full, untrimmed profile, once against the trimmed view (current default
behavior) — and reports fit_score for each posting side by side plus max/mean
absolute delta.

Usage:
    python -m scripts.validate_tier1_trim --user-id ID [--samples N]

Required before relying on src/scoring.py's Tier 1 profile trim: if any posting's
fit_score moves by more than 0.02, something "operational" was actually informing
the score and the trim isn't safe as shipped.
"""
from __future__ import annotations

import argparse
import statistics
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

from src import scoring as scoring_mod  # noqa: E402
from src.gate import check as gate_check  # noqa: E402
from src.sources import eightyk, iap, probablygood  # noqa: E402
from src.supabase_client import get_client  # noqa: E402
from src.user_store import fetch_users  # noqa: E402

_FLAG_THRESHOLD = 0.02


@contextmanager
def _trim_disabled():
    """Temporarily make _tier1_profile_view an identity function, so score() runs
    against the FULL, untrimmed profile — for the "before" comparison run only."""
    original = scoring_mod._tier1_profile_view
    scoring_mod._tier1_profile_view = lambda profile: profile
    try:
        yield
    finally:
        scoring_mod._tier1_profile_view = original


def _fetch_sample_postings(limit: int) -> list:
    """Round-robin across sources rather than concatenate-then-truncate — see
    measure_tier1_tokens.py's identical helper for why a fixed source order would
    silently bias the sample."""
    per_source: list[list] = []
    for name, fetch_fn in [("iap", iap.fetch), ("80k", eightyk.fetch), ("probably_good", probablygood.fetch)]:
        try:
            per_source.append(fetch_fn())
        except Exception as exc:
            print(f"  (source {name} failed, skipping: {exc})")
            per_source.append([])

    postings = []
    for group in zip(*per_source):
        postings.extend(group)
        if len(postings) >= limit:
            break
    return postings[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Tier 1 profile trim doesn't move fit_score")
    parser.add_argument("--user-id", required=True, help="Supabase user_id")
    parser.add_argument("--samples", type=int, default=10, help="Number of postings to sample (default 10)")
    args = parser.parse_args()

    print(f"Loading user {args.user_id} …")
    users, excluded_users = fetch_users(get_client(), only_user_id=args.user_id)
    if excluded_users:
        for e in excluded_users:
            print(f"User {e.user_id} excluded — {e.field}: {e.reason}")
    if not users:
        print(f"No user found with user_id={args.user_id} (needs both a profile and scoring_weights row).")
        return
    user = users[0]

    print(f"Fetching up to {args.samples} sample posting(s) …")
    postings = _fetch_sample_postings(args.samples)
    if not postings:
        print("No postings fetched from any source — nothing to validate.")
        return
    if len(postings) < args.samples:
        print(f"  warning: only found {len(postings)} posting(s) (wanted {args.samples})")
    print(f"Using {len(postings)} posting(s)\n")

    print("Scoring with FULL (untrimmed) profile …")
    with _trim_disabled():
        full_results = []
        for posting in postings:
            gate = gate_check(posting, user.profile)
            scored = scoring_mod.score(posting, gate, user.profile, user.rubric)
            full_results.append(scored.fit_score)

    print("Scoring with TRIMMED profile (current default) …")
    trimmed_results = []
    for posting in postings:
        gate = gate_check(posting, user.profile)
        scored = scoring_mod.score(posting, gate, user.profile, user.rubric)
        trimmed_results.append(scored.fit_score)

    deltas = [abs(a - b) for a, b in zip(full_results, trimmed_results)]

    print()
    print("=" * 78)
    print(f"{'posting':48} {'full':>9} {'trimmed':>9} {'delta':>9}")
    print("=" * 78)
    flagged = []
    for posting, full, trimmed, delta in zip(postings, full_results, trimmed_results, deltas):
        flag = delta > _FLAG_THRESHOLD
        if flag:
            flagged.append(posting.url)
        title = (posting.title or posting.url)[:46]
        marker = f"  <-- FLAG (> {_FLAG_THRESHOLD})" if flag else ""
        print(f"{title:48} {full:>9.4f} {trimmed:>9.4f} {delta:>9.4f}{marker}")

    print("-" * 78)
    print(f"max delta:  {max(deltas):.4f}")
    print(f"mean delta: {statistics.mean(deltas):.4f}")
    if flagged:
        print()
        print(f"!! {len(flagged)} posting(s) moved by more than {_FLAG_THRESHOLD} — the trim is NOT safe as shipped:")
        for url in flagged:
            print(f"   {url}")
    else:
        print()
        print(f"No posting moved by more than {_FLAG_THRESHOLD} — trim looks safe for this sample.")


if __name__ == "__main__":
    main()
