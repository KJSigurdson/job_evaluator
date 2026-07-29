"""One-off measurement: Tier 1 prompt token breakdown for one real user, via
Anthropic's messages.count_tokens endpoint. Read-only — makes zero scoring calls,
just tokenizes what the real Tier 1 prompt WOULD contain. Context: Tier 1 was
observed averaging ~2,483 input tokens/call across 56 real calls in production, with
no prior visibility into which prompt piece drove that.

Usage:
    python -m scripts.measure_tier1_tokens --user-id ID [--samples N]

--user-id ID   required. Supabase user_id to load profile+scoring_weights for
               (via user_store.fetch_users(client, only_user_id=ID)).
--samples N    number of real postings to sample for the posting.raw_text
               min/median/max breakdown (default 10).

Reuses scoring.py's REAL prompt-construction pieces (_build_profile_text,
_build_dimensions_text, _COMP_ADEQUACY_RULE, _tier1_profile_view) rather than
re-deriving these strings, so this measurement cannot silently drift from the
actual Tier 1 prompt in src/scoring.py's _call_llm/_acall_llm.

Every per-piece cost below is a MARGINAL cost: full-prompt tokens minus the same
prompt with that piece blanked out. Pieces can share sub-word tokens at their
boundaries, so the reported pieces will not sum to EXACTLY the true total — this is
normal tokenizer behavior, not a measurement bug, and is called out in the report.
"""
from __future__ import annotations

import argparse
import statistics

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic  # noqa: E402  (must follow load_dotenv)

from src.schemas import Tier1ScoreOutput  # noqa: E402
from src.scoring import (  # noqa: E402
    _COMP_ADEQUACY_RULE,
    _TIER1_MODEL_DEFAULT,
    _build_dimensions_text,
    _build_profile_text,
    _tier1_profile_view,
)
from src.sources import eightyk, iap, probablygood  # noqa: E402
from src.supabase_client import get_client  # noqa: E402
from src.user_store import fetch_users  # noqa: E402

_CACHEABLE_PREFIX_MIN_TOKENS = 4096  # claude-haiku-4-5's minimum cacheable-prefix length

_TOOLS = [{
    "name": "score_posting",
    "description": "Output the seven-dimension rubric score for a job posting.",
    "input_schema": Tier1ScoreOutput.model_json_schema(),
}]


def _build_prompt(profile_text: str, dimensions_text: str, posting_raw_text: str) -> str:
    """Byte-for-byte the same assembly as scoring.py's _call_llm/_acall_llm."""
    return (
        "Score the following job posting against the candidate profile.\n\n"
        f"CANDIDATE PROFILE:\n{profile_text}\n\n"
        f"SCORING DIMENSIONS (score 0.0–1.0 each, one-line rationale required):\n"
        f"{dimensions_text}\n\n"
        f"{_COMP_ADEQUACY_RULE}"
        f"JOB POSTING:\n{posting_raw_text}"
    )


def _count(client: Anthropic, model: str, prompt: str, *, tools: list | None) -> int:
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
    if tools is not None:
        kwargs["tools"] = tools
    return client.messages.count_tokens(**kwargs).input_tokens


def _fetch_sample_postings(limit: int) -> list:
    """Round-robin across sources rather than concatenate-then-truncate: IAP's
    raw_text runs 51-278 chars (short referral-doc snippets) while 80k/
    probably_good run up to ~950/~1970 chars (full scraped descriptions) — taking
    the first N in a fixed source order silently biases the sample toward
    whichever source happens to be queried first."""
    per_source: list[list] = []
    for name, fetch_fn in [("iap", iap.fetch), ("80k", eightyk.fetch), ("probably_good", probablygood.fetch)]:
        try:
            per_source.append(fetch_fn())
        except Exception as exc:
            print(f"  (source {name} failed, skipping: {exc})")
            per_source.append([])

    postings = []
    for group in zip(*per_source):  # round-robin: one from each source per round
        postings.extend(group)
        if len(postings) >= limit:
            break
    return postings[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Tier 1 prompt token breakdown for one real user")
    parser.add_argument("--user-id", required=True, help="Supabase user_id")
    parser.add_argument("--samples", type=int, default=10, help="Number of postings to sample (default 10)")
    args = parser.parse_args()

    model = _TIER1_MODEL_DEFAULT
    anthropic_client = Anthropic()

    print(f"Loading user {args.user_id} …")
    users = fetch_users(get_client(), only_user_id=args.user_id)
    if not users:
        print(f"No user found with user_id={args.user_id} (needs both a profile and scoring_weights row).")
        return
    user = users[0]
    profile = user.profile
    rubric = user.rubric
    tier1_profile = _tier1_profile_view(profile)

    print(f"Fetching up to {args.samples} sample posting(s) …")
    postings = _fetch_sample_postings(args.samples)
    if not postings:
        print("No postings fetched from any source — nothing to measure.")
        return
    print(f"Using {len(postings)} posting(s)\n")

    profile_text = _build_profile_text(tier1_profile)
    dimensions_text = _build_dimensions_text(rubric)

    # ---------------------------------------------------------------------
    # Baselines for marginal-cost isolation
    # ---------------------------------------------------------------------
    empty_prompt = _build_prompt("", "", "")
    full_prompt_no_posting = _build_prompt(profile_text, dimensions_text, "")

    baseline_no_tools = _count(anthropic_client, model, empty_prompt, tools=None)
    baseline_with_tools = _count(anthropic_client, model, empty_prompt, tools=_TOOLS)
    tools_cost = baseline_with_tools - baseline_no_tools

    profile_only_prompt = _build_prompt(profile_text, "", "")
    profile_cost = _count(anthropic_client, model, profile_only_prompt, tools=_TOOLS) - baseline_with_tools

    dims_only_prompt = _build_prompt("", dimensions_text, "")
    dims_and_rule_cost = _count(anthropic_client, model, dims_only_prompt, tools=_TOOLS) - baseline_with_tools

    cacheable_prefix_tokens = _count(anthropic_client, model, full_prompt_no_posting, tools=_TOOLS)

    # ---------------------------------------------------------------------
    # Per-posting raw_text marginal cost, across the sample
    # ---------------------------------------------------------------------
    raw_text_costs = []
    full_totals = []
    for posting in postings:
        full_prompt = _build_prompt(profile_text, dimensions_text, posting.raw_text)
        full_total = _count(anthropic_client, model, full_prompt, tools=_TOOLS)
        full_totals.append(full_total)
        raw_text_costs.append(full_total - cacheable_prefix_tokens)

    # ---------------------------------------------------------------------
    # Per-profile-key marginal cost (against the FULL, untrimmed profile — so the
    # report can show what each stripped key cost too, not just what remains)
    # ---------------------------------------------------------------------
    full_profile_text = _build_profile_text(profile)
    full_profile_baseline = _count(
        anthropic_client, model, _build_prompt(full_profile_text, "", ""), tools=_TOOLS,
    )
    key_costs: dict[str, int] = {}
    for key in profile:
        stripped = {k: v for k, v in profile.items() if k != key}
        stripped_text = _build_profile_text(stripped)
        stripped_total = _count(anthropic_client, model, _build_prompt(stripped_text, "", ""), tools=_TOOLS)
        key_costs[key] = full_profile_baseline - stripped_total

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("TIER 1 PROMPT TOKEN BREAKDOWN")
    print("=" * 70)
    print(f"{'tools schema (static, every call)':45} {tools_cost:>8}")
    print(f"{'profile_text (Tier-1 trimmed view, static/user)':45} {profile_cost:>8}")
    print(f"{'dimensions_text + comp_adequacy rule (static/user)':45} {dims_and_rule_cost:>8}")
    print(
        f"{'posting.raw_text (varies/call): min/median/max':45} "
        f"{min(raw_text_costs):>4} / {statistics.median(raw_text_costs):>4.0f} / {max(raw_text_costs):>4}"
    )
    print("-" * 70)
    print(f"{'TOTAL per call: min/median/max (measured, full prompt)':45} "
          f"{min(full_totals):>4} / {statistics.median(full_totals):>4.0f} / {max(full_totals):>4}")

    print()
    print("=" * 70)
    print("CACHEABLE PREFIX (tools + profile + dimensions + rule, everything before")
    print("the posting) — claude-haiku-4-5 requires >= 4,096 tokens to cache")
    print("=" * 70)
    clears = cacheable_prefix_tokens >= _CACHEABLE_PREFIX_MIN_TOKENS
    print(f"Prefix tokens: {cacheable_prefix_tokens}")
    print(f"Clears the {_CACHEABLE_PREFIX_MIN_TOKENS}-token cacheable minimum: {'YES' if clears else 'NO'}")
    if not clears:
        print(
            f"  -> {_CACHEABLE_PREFIX_MIN_TOKENS - cacheable_prefix_tokens} tokens short. A cache_control "
            "marker here would silently no-op at full price — do not enable caching at this prefix size."
        )

    print()
    print("=" * 70)
    print("PER-PROFILE-KEY TOKEN COST (against the FULL, untrimmed profile — includes")
    print("keys the Tier 1 trim below removes, so you can see what was saved)")
    print("=" * 70)
    for key, cost in sorted(key_costs.items(), key=lambda kv: -kv[1]):
        stripped_marker = " (stripped from Tier 1 prompt)" if key not in tier1_profile else ""
        print(f"{key:35} {cost:>8}{stripped_marker}")


if __name__ == "__main__":
    main()
