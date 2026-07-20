"""Shared fixtures and stub factory helpers."""
from __future__ import annotations

import json

import pytest

from src.schemas import DimensionScore, RawPosting, Tier1ScoreOutput
from src.user_store import _build_profile, _build_rubric

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


# ---------------------------------------------------------------------------
# Fake Supabase client — in-memory tables, enough of the query-builder surface
# for user_store.py / seen_store.py / matches_store.py to run against.
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, data: list[dict]):
        self.data = data


class _FakeQuery:
    def __init__(self, table: "_FakeTable", op: str, payload=None, on_conflict: str | None = None):
        self._table = table
        self._op = op
        self._payload = payload
        self._on_conflict = on_conflict
        self._filters: list[tuple[str, object]] = []

    def select(self, _cols: str = "*") -> "_FakeQuery":
        return self

    def eq(self, col: str, val) -> "_FakeQuery":
        self._filters.append(("eq", col, val))
        return self

    def in_(self, col: str, values) -> "_FakeQuery":
        self._filters.append(("in", col, list(values)))
        return self

    def execute(self) -> _FakeResponse:
        if self._op == "select":
            rows = self._table.rows
            for kind, col, val in self._filters:
                if kind == "eq":
                    rows = [r for r in rows if r.get(col) == val]
                elif kind == "in":
                    rows = [r for r in rows if r.get(col) in val]
            return _FakeResponse([dict(r) for r in rows])

        if self._op == "upsert":
            # Supabase's real upsert() accepts either a single dict or a list of dicts
            # (a batch upsert in one HTTP call) — mirror both here.
            key_cols = (self._on_conflict or "").split(",")
            rows = self._payload if isinstance(self._payload, list) else [self._payload]
            self._table.upsert_calls.append(rows)  # one entry per .upsert().execute() — lets tests count round-trips
            upserted = []
            for row_payload in rows:
                match = next(
                    (r for r in self._table.rows if all(r.get(c) == row_payload.get(c) for c in key_cols)),
                    None,
                )
                if match is not None:
                    match.update(row_payload)
                else:
                    self._table.rows.append(dict(row_payload))
                upserted.append(dict(row_payload))
            return _FakeResponse(upserted)

        raise NotImplementedError(self._op)


class _FakeTable:
    def __init__(self):
        self.rows: list[dict] = []
        self.upsert_calls: list[list[dict]] = []  # one entry (list of rows) per .upsert().execute() call

    def select(self, cols: str = "*") -> _FakeQuery:
        return _FakeQuery(self, "select").select(cols)

    def upsert(self, payload: dict, on_conflict: str | None = None) -> _FakeQuery:
        return _FakeQuery(self, "upsert", payload=payload, on_conflict=on_conflict)


class FakeSupabaseClient:
    """Minimal stand-in for a supabase.Client. Backs each table with an in-memory list."""

    def __init__(self):
        self._tables: dict[str, _FakeTable] = {}

    def table(self, name: str) -> _FakeTable:
        return self._tables.setdefault(name, _FakeTable())

    def seed(self, name: str, rows: list[dict]) -> None:
        self.table(name).rows.extend(dict(r) for r in rows)


@pytest.fixture
def fake_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


# ---------------------------------------------------------------------------
# profile / rubric — built the same way pipeline.py builds them at runtime,
# from fake `profiles` + `scoring_weights` rows, so test_gate.py/test_scoring.py
# exercise the real Supabase-row → dict mapping instead of a hand-rolled shape.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def profile_row() -> dict:
    return {
        "user_id": "u-test",
        "experience": {
            "core_strengths": ["Built a BI function from scratch", "Led multi-country rollout"],
            "technical": {"sql": "strong", "python": "working"},
        },
        "skills": ["data leadership", "analytics", "stakeholder management"],
        "career_goals": "Move into an EA-aligned data leadership role.",
        "cause_priorities": ["global health and development", "EA-aligned impact"],
        "location": "Sundsvall, Sweden",
        "seniority_level": "10+ years, director-level",
        "comp_needs": "Flag tjänstepension replicability when comp is discussed.",
        "values_notes": "GWWC pledge, HIP IAP completed, effective giving advocacy.",
    }


@pytest.fixture(scope="session")
def weights_row() -> dict:
    return {
        "user_id": "u-test",
        "cause_mission_fit": 0.25,
        "role_function_fit": 0.25,
        "location_compatibility": 0.15,
        "seniority_match": 0.10,
        "comp_adequacy": 0.10,
        "values_alignment": 0.10,
        "skill_growth": 0.05,
        # location_rule/seniority_rule are `text` columns in real Postgres — PostgREST
        # returns them as JSON strings, not parsed objects. Stored as strings here to
        # match that and exercise user_store._parse_rule end-to-end.
        "location_rule": json.dumps({
            "accept_fully_remote": True,
            "accept_hybrid_in": ["Sweden", "Sundsvall"],
            "accept_onsite_in": ["Sundsvall"],
        }),
        # 10+ years, director-level candidate — doesn't want intern/junior-level roles.
        "seniority_rule": json.dumps({"accept_levels": ["mid", "senior", "director"]}),
        "insert_threshold": 0.75,
        "near_miss_floor": 0.65,
    }


@pytest.fixture(scope="session")
def profile(profile_row: dict, weights_row: dict) -> dict:
    return _build_profile(profile_row, weights_row)


@pytest.fixture(scope="session")
def rubric(weights_row: dict) -> dict:
    return _build_rubric(weights_row)


# ---------------------------------------------------------------------------
# Posting / Tier 1 score stub factories
# ---------------------------------------------------------------------------

def make_posting(**kwargs) -> RawPosting:
    defaults = dict(
        url="https://example.org/job/123",
        title="Head of Data",
        org="GiveDirectly",
        source="80k",
        location=None,
        seniority=None,
        comp=None,
        deadline=None,
        raw_text="Full job description text.",
    )
    defaults.update(kwargs)
    return RawPosting(**defaults)


def make_tier1(
    cause_mission_fit: float = 0.8,
    role_function_fit: float = 0.8,
    location_compatibility: float = 0.8,
    seniority_match: float = 0.8,
    comp_adequacy: float = 0.8,
    values_alignment: float = 0.8,
    skill_growth: float = 0.8,
) -> Tier1ScoreOutput:
    def dim(score: float) -> DimensionScore:
        return DimensionScore(score=score, rationale="stub")

    return Tier1ScoreOutput(
        cause_mission_fit=dim(cause_mission_fit),
        role_function_fit=dim(role_function_fit),
        location_compatibility=dim(location_compatibility),
        seniority_match=dim(seniority_match),
        comp_adequacy=dim(comp_adequacy),
        values_alignment=dim(values_alignment),
        skill_growth=dim(skill_growth),
    )
