"""Tests for src/reformat_match_enrichment.py: the one-off matches-enrichment
reformatting migration. Stub-based — never hits the Anthropic API."""
from __future__ import annotations

import pytest

from src.dedup import canonical_url
from src.matches_store import upsert_match
from src.reformat_match_enrichment import (
    ReformatError,
    _ReformatOutput,
    _build_prompt,
    reformat_many,
    run,
)
from src.schemas import MatchRow


def _match_row(**kwargs) -> MatchRow:
    from datetime import date
    defaults = dict(
        title="Head of Data",
        org="GiveDirectly",
        org_summary="A charity evaluator.",
        source="80k",
        url="https://example.org/job/1",
        canonical_url=canonical_url("https://example.org/job/1"),
        date_found=date.today(),
        fit_score=0.85,
        dimension_scores={"cause_mission_fit": 0.9},
        why_fits="A long paragraph explaining strong mission alignment and technical fit all in one breath.",
        why_not_fits="A paragraph noting the candidate lacks fundraising experience for this particular role.",
        emphasize_in_cv=["Emphasize the volume growth program because it demonstrates modelling skills."],
        deemphasize=["De-emphasize the early career retail work since it isn't relevant here."],
    )
    defaults.update(kwargs)
    return MatchRow(**defaults)


def _row_dict(**kwargs) -> dict:
    defaults = dict(
        user_id="u1",
        canonical_url="https://example.org/job/1",
        why_fits="Prose paragraph about fit.",
        why_not_fits="Prose paragraph about gaps.",
        emphasize_in_cv=["Full sentence about a strength."],
        deemphasize=["Full sentence about something irrelevant."],
    )
    defaults.update(kwargs)
    return defaults


def _stub_output(**kwargs) -> _ReformatOutput:
    defaults = dict(
        why_fits="- bullet one\n- bullet two",
        why_not_fits="- gap one",
        emphasize_in_cv=["short phrase"],
        deemphasize=["other phrase"],
    )
    defaults.update(kwargs)
    return _ReformatOutput(**defaults)


# ---------------------------------------------------------------------------
# _build_prompt — asserts formatting-only, no-re-evaluation instructions
# ---------------------------------------------------------------------------

def test_prompt_instructs_formatting_only_not_re_evaluation():
    prompt = _build_prompt(_row_dict())
    lowered = prompt.lower()
    assert "not evaluating" in lowered or "not re-evaluating" in lowered or "formatting pass only" in lowered
    assert "do not add" in lowered
    assert "do not" in lowered and ("remove" in lowered or "soften" in lowered)


def test_prompt_includes_current_text_verbatim():
    row = _row_dict(why_fits="UNIQUE_MARKER_TEXT")
    prompt = _build_prompt(row)
    assert "UNIQUE_MARKER_TEXT" in prompt


def test_prompt_specifies_bullet_format_rules():
    prompt = _build_prompt(_row_dict())
    assert "- " in prompt
    assert "3-6 bullets" in prompt or "3-6" in prompt


# ---------------------------------------------------------------------------
# reformat_many — concurrency + per-row failure isolation (mirrors score_many)
# ---------------------------------------------------------------------------

def test_reformat_many_preserves_order():
    rows = [_row_dict(canonical_url=f"https://example.org/{i}") for i in range(5)]

    async def stub_llm(row):
        return _stub_output(why_fits=f"- {row['canonical_url']}")

    results = _run(reformat_many(rows, max_concurrency=2, _allm_fn=stub_llm))
    assert [r.why_fits for r in results] == [f"- https://example.org/{i}" for i in range(5)]


def test_reformat_many_isolates_per_row_failure():
    rows = [_row_dict(canonical_url="https://example.org/ok"), _row_dict(canonical_url="https://example.org/bad")]

    async def stub_llm(row):
        if row["canonical_url"] == "https://example.org/bad":
            raise ValueError("boom")
        return _stub_output()

    results = _run(reformat_many(rows, max_concurrency=2, _allm_fn=stub_llm))
    assert isinstance(results[0], _ReformatOutput)
    assert isinstance(results[1], ReformatError)


def test_reformat_many_retries_once_before_failing():
    calls = {"n": 0}

    async def flaky_llm(row):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("transient")
        return _stub_output()

    results = _run(reformat_many([_row_dict()], max_concurrency=1, _allm_fn=flaky_llm))
    assert isinstance(results[0], _ReformatOutput)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# run() — orchestration against the fake Supabase client
# ---------------------------------------------------------------------------

def _run(coro):
    import asyncio
    return asyncio.run(coro)


async def _stub_llm(row):
    return _stub_output(
        why_fits=f"- reformatted {row['canonical_url']}",
        why_not_fits="- reformatted gap",
        emphasize_in_cv=["reformatted emphasize"],
        deemphasize=["reformatted deemphasize"],
    )


def test_run_requires_limit_or_confirm_all(fake_client):
    with pytest.raises(ValueError):
        run(_client=fake_client, _allm_fn=_stub_llm)


def test_run_updates_matching_rows(fake_client):
    row = _match_row()
    upsert_match(fake_client, "u1", row)

    summary = run(_client=fake_client, limit=10, _allm_fn=_stub_llm)

    assert summary == {"processed": 1, "updated": 1, "failed": 0}
    updated = fake_client.table("matches").rows[0]
    assert updated["why_fits"] == f"- reformatted {row.canonical_url}"
    assert updated["fit_score"] == row.fit_score  # untouched


def test_run_scoped_to_user_id(fake_client):
    upsert_match(fake_client, "u1", _match_row(url="https://example.org/a", canonical_url=canonical_url("https://example.org/a")))
    upsert_match(fake_client, "u2", _match_row(url="https://example.org/b", canonical_url=canonical_url("https://example.org/b")))

    summary = run(_client=fake_client, confirm_all=True, user_id="u1", _allm_fn=_stub_llm)

    assert summary["processed"] == 1
    rows_by_user = {r["user_id"]: r for r in fake_client.table("matches").rows}
    assert rows_by_user["u1"]["why_fits"].startswith("- reformatted")
    assert rows_by_user["u2"]["why_fits"] == "A long paragraph explaining strong mission alignment and technical fit all in one breath."


def test_run_dry_run_makes_no_update_calls(fake_client):
    upsert_match(fake_client, "u1", _match_row())

    summary = run(_client=fake_client, limit=10, dry_run=True, _allm_fn=_stub_llm)

    assert summary["updated"] == 0
    assert summary["processed"] == 1
    assert fake_client.table("matches").update_calls == []
    # original row untouched
    assert fake_client.table("matches").rows[0]["why_fits"] == "A long paragraph explaining strong mission alignment and technical fit all in one breath."


def test_run_isolates_per_row_failure(fake_client):
    upsert_match(fake_client, "u1", _match_row(url="https://example.org/ok", canonical_url=canonical_url("https://example.org/ok")))
    upsert_match(fake_client, "u1", _match_row(url="https://example.org/bad", canonical_url=canonical_url("https://example.org/bad")))

    async def sometimes_fails(row):
        if row["canonical_url"] == canonical_url("https://example.org/bad"):
            raise ValueError("boom")
        return _stub_output()

    summary = run(_client=fake_client, confirm_all=True, _allm_fn=sometimes_fails)

    assert summary["processed"] == 2
    assert summary["updated"] == 1
    assert summary["failed"] == 1


def test_run_applies_limit(fake_client):
    for i in range(5):
        upsert_match(fake_client, "u1", _match_row(url=f"https://example.org/{i}", canonical_url=canonical_url(f"https://example.org/{i}")))

    summary = run(_client=fake_client, limit=2, _allm_fn=_stub_llm)
    assert summary["processed"] == 2
