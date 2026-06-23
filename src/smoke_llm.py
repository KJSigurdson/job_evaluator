"""End-to-end LLM smoke test — costs almost nothing (3 postings max).

Usage:
    python -m src.smoke_llm

Fetches the first 3 postings from ProbablyGood (no extra env vars needed),
runs Tier 1 scoring on all three, then runs Tier 2 enrichment on the
highest scorer.  Loads secrets from .env automatically.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.enrich import enrich  # noqa: E402  (must follow load_dotenv)
from src.gate import check as gate_check  # noqa: E402
from src.scoring import score  # noqa: E402
from src.sources.probablygood import fetch_raw_hits, parse_hits  # noqa: E402

_ROOT = Path(__file__).parent.parent


def _load_yaml(name: str) -> dict:
    with open(_ROOT / name) as f:
        return yaml.safe_load(f)


def main() -> None:
    profile = _load_yaml("profile.yaml")
    rubric  = _load_yaml("rubric.yaml")

    print("Fetching ProbablyGood postings …")
    raw_hits = fetch_raw_hits()
    postings = parse_hits(raw_hits)[:3]
    print(f"Using {len(postings)} posting(s)\n")

    scored_all = []
    for i, posting in enumerate(postings, 1):
        print(f"{'='*60}")
        print(f"[{i}] {posting.title} @ {posting.org}")
        print(f"    URL: {posting.url}")

        gate = gate_check(posting, profile)
        print(f"    Gate: {'PASS' if gate.passed else 'FAIL'} — {gate.reason or 'ok'}")

        try:
            result = score(posting, gate, profile, rubric)
        except Exception as exc:
            print(f"    Tier 1 FAILED: {exc}")
            continue

        print(f"    fit_score: {result.fit_score:.3f}")
        for dim in type(result.scores).model_fields:
            ds = getattr(result.scores, dim)
            print(f"      {dim:<30} {ds.score:.2f}  {ds.rationale}")

        scored_all.append(result)

    if not scored_all:
        print("\nNo postings scored — nothing to enrich.")
        return

    best = max(scored_all, key=lambda r: r.fit_score)
    print(f"\n{'='*60}")
    print(f"Tier 2 enrichment for highest scorer ({best.fit_score:.3f}):")
    print(f"  {best.posting.title} @ {best.posting.org}")

    try:
        enrichment = enrich(best, profile)
    except Exception as exc:
        print(f"Tier 2 FAILED: {exc}")
        return

    print(f"\n  org_summary:      {enrichment.org_summary}")
    print(f"  why_fits:         {enrichment.why_fits}")
    print(f"  why_not_fits:     {enrichment.why_not_fits}")
    print(f"  emphasize_in_cv:  {enrichment.emphasize_in_cv}")
    print(f"  deemphasize:      {enrichment.deemphasize}")


if __name__ == "__main__":
    main()
