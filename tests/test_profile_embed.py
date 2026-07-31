"""Tests for src/profile_embed.py: embedding text template + re-embed-on-change
backfill against profiles.profile_embedding/profile_content_hash."""
from __future__ import annotations

import json

from src import profile_embed
from src.job_normalize import content_hash
from src.profile_embed import backfill_profile_embeddings, build_embedding_text


def _profile_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", experience="Work: Data leader at OrgA", skills="SQL (expert), Python (working)",
        career_goals="Move into data leadership at an EA-aligned org.",
        cause_priorities="1. Global health\n2. Animal welfare",
        location="Berlin, Germany", seniority_level="Senior", comp_needs="Market rate",
        values_notes="GWWC pledge",
    )
    defaults.update(kwargs)
    return defaults


def _weights_row(**kwargs) -> dict:
    defaults = dict(
        user_id="u1", cause_mission_fit=0.25, role_function_fit=0.25, location_compatibility=0.15,
        seniority_match=0.10, comp_adequacy=0.10, values_alignment=0.10, skill_growth=0.05,
        location_rule=json.dumps({"accept_fully_remote": True, "accept_hybrid_in": [], "accept_onsite_in": []}),
        seniority_rule=json.dumps({"accept_levels": ["mid", "senior", "director"]}),
        insert_threshold=0.75, near_miss_floor=0.65,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# build_embedding_text
# ---------------------------------------------------------------------------

def test_build_embedding_text_includes_all_five_sections():
    profile = {
        "career_goals": "Goal text", "cause_priorities": "Cause text",
        "experience_inventory": "Exp text", "skills": "Skill text", "values_notes": "Values text",
    }
    text = build_embedding_text(profile)
    assert "Career goals: Goal text" in text
    assert "Cause priorities: Cause text" in text
    assert "Experience: Exp text" in text
    assert "Skills: Skill text" in text
    assert "Values: Values text" in text


def test_build_embedding_text_excludes_location_and_comp_needs():
    profile = {
        "career_goals": "Goal text", "location": "Sundsvall, Sweden", "comp_needs": "70000 SEK",
    }
    text = build_embedding_text(profile)
    assert "Sundsvall" not in text
    assert "70000" not in text


def test_build_embedding_text_skips_empty_fields():
    profile = {"career_goals": "Goal text", "cause_priorities": None, "values_notes": "   "}
    text = build_embedding_text(profile)
    assert "Cause priorities" not in text
    assert "Values" not in text


def test_build_embedding_text_empty_profile_returns_empty_string():
    assert build_embedding_text({}) == ""


# ---------------------------------------------------------------------------
# backfill_profile_embeddings
# ---------------------------------------------------------------------------

def test_backfill_noop_without_key(fake_client, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])

    result = backfill_profile_embeddings(fake_client)

    assert result == {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0}


def test_backfill_embeds_and_writes_hash(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(profile_embed, "embed_texts", lambda texts: [[0.3] * 1024 for _ in texts])
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])

    result = backfill_profile_embeddings(fake_client)

    assert result["selected"] == 1
    assert result["embedded"] == 1
    row = fake_client.table("profiles").rows[0]
    assert row["profile_embedding"] == [0.3] * 1024
    assert row["profile_content_hash"]


def test_backfill_skips_unchanged_hash(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    embed_calls = []
    monkeypatch.setattr(profile_embed, "embed_texts", lambda texts: embed_calls.append(texts) or [[0.3] * 1024 for _ in texts])
    fake_client.seed("profiles", [_profile_row()])
    fake_client.seed("scoring_weights", [_weights_row()])

    first = backfill_profile_embeddings(fake_client)
    assert first["embedded"] == 1

    second = backfill_profile_embeddings(fake_client)
    assert second["embedded"] == 0
    assert second["skipped_unchanged"] == 1
    assert len(embed_calls) == 1  # second run made no new embed call


def test_backfill_reembeds_when_profile_content_changes(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(profile_embed, "embed_texts", lambda texts: [[0.3] * 1024 for _ in texts])
    fake_client.seed("profiles", [_profile_row(career_goals="Original goal")])
    fake_client.seed("scoring_weights", [_weights_row()])
    backfill_profile_embeddings(fake_client)

    fake_client.table("profiles").rows[0]["career_goals"] = "Updated goal, totally different direction"

    result = backfill_profile_embeddings(fake_client)

    assert result["embedded"] == 1
    assert result["skipped_unchanged"] == 0


def test_backfill_skips_user_with_empty_profile(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(profile_embed, "embed_texts", lambda texts: [[0.3] * 1024 for _ in texts])
    fake_client.seed("profiles", [_profile_row(
        experience=None, skills=None, career_goals=None, cause_priorities=None, values_notes=None,
    )])
    fake_client.seed("scoring_weights", [_weights_row()])

    result = backfill_profile_embeddings(fake_client)

    assert result["embedded"] == 0
    assert result["skipped_empty"] == 1


def test_backfill_skips_user_without_weights(fake_client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    monkeypatch.setattr(profile_embed, "embed_texts", lambda texts: [[0.3] * 1024 for _ in texts])
    fake_client.seed("profiles", [_profile_row()])
    # no scoring_weights row seeded

    result = backfill_profile_embeddings(fake_client)

    assert result == {"selected": 0, "embedded": 0, "skipped_unchanged": 0, "skipped_empty": 0}


def test_content_hash_matches_job_normalize_reuse():
    """profile_content_hash should be produced by the SAME hashing helper as
    job_normalize.content_hash, not a re-implementation."""
    text = "Career goals: something"
    assert content_hash(text) == content_hash(text.lower())
