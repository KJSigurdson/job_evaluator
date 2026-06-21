"""Hard-constraint gate (step 2). Binary location + seniority checks against profile.yaml."""
from __future__ import annotations

from src.schemas import HardGateResult, RawPosting


def check(posting: RawPosting, profile: dict) -> HardGateResult:
    raise NotImplementedError
