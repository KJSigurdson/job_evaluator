"""Tests for src/match_v2.py: gate_v2 survivor filtering, then wiring to the
match_v2_top_jobs() RPC for the pgvector cosine rerank (faked — the real cosine
math runs in Postgres and can't be meaningfully re-implemented in Python here;
these tests verify the calling contract and result-mapping instead)."""
from __future__ import annotations

from src.match_v2 import TOP_N, fetch_matched_job_ids, gate_survivors, gate_survivors_with_rejections, match_user, vector_rerank


def _job(job_id="job-1", location_field="Remote, Global", mode="remote",
         remote_scope="global", seniority=None) -> dict:
    return {
        "id": job_id, "title": f"Title {job_id}", "org": "Org",
        "location_field": location_field,
        "enrichment": {
            "seniority": seniority or ["senior"],
            "work_arrangement": {"mode": mode, "remote_scope": remote_scope,
                                  "eligible_regions": [], "restriction_type": None,
                                  "timezone_constraint": None},
        },
    }


# ---------------------------------------------------------------------------
# gate_survivors
# ---------------------------------------------------------------------------

def test_gate_survivors_filters_by_gate_v2(profile):
    jobs = [
        _job("pass-1", location_field="Remote, Global", mode="remote", remote_scope="global"),
        _job("fail-1", location_field="Nairobi, Kenya", mode="onsite"),
    ]
    result = gate_survivors(jobs, profile)
    assert [j["id"] for j in result] == ["pass-1"]


def test_gate_survivors_empty_jobs_returns_empty(profile):
    assert gate_survivors([], profile) == []


# ---------------------------------------------------------------------------
# gate_survivors_with_rejections
# ---------------------------------------------------------------------------

def test_gate_survivors_with_rejections_tallies_by_rule(profile):
    jobs = [
        _job("pass-1", location_field="Remote, Global", mode="remote", remote_scope="global"),
        _job("fail-location", location_field="Nairobi, Kenya", mode="onsite"),
        _job("fail-seniority", location_field="Remote, Global", mode="remote", remote_scope="global", seniority=["intern"]),
    ]
    survivors, rejections = gate_survivors_with_rejections(jobs, profile)
    assert [j["id"] for j in survivors] == ["pass-1"]
    assert rejections == {"location": 1, "seniority": 1}


def test_gate_survivors_with_rejections_empty_jobs(profile):
    assert gate_survivors_with_rejections([], profile) == ([], {})


# ---------------------------------------------------------------------------
# vector_rerank
# ---------------------------------------------------------------------------

def test_vector_rerank_empty_candidates_skips_rpc_call(fake_client):
    result = vector_rerank(fake_client, profile_embedding="[0.1,0.2]", candidate_job_ids=[])
    assert result == []
    assert fake_client.rpc_calls == []


def test_vector_rerank_calls_rpc_with_expected_params(fake_client):
    fake_client.rpc_responses["match_v2_top_jobs"] = [{"id": "job-1", "similarity": 0.9}]

    vector_rerank(fake_client, profile_embedding="[0.1,0.2]", candidate_job_ids=["job-1", "job-2"], top_n=25)

    [(fn_name, params)] = fake_client.rpc_calls
    assert fn_name == "match_v2_top_jobs"
    assert params == {
        "query_embedding": "[0.1,0.2]", "candidate_job_ids": ["job-1", "job-2"], "match_limit": 25,
    }


def test_vector_rerank_returns_rpc_response_data(fake_client):
    fake_client.rpc_responses["match_v2_top_jobs"] = [
        {"id": "job-1", "similarity": 0.95}, {"id": "job-2", "similarity": 0.80},
    ]

    result = vector_rerank(fake_client, profile_embedding="[0.1,0.2]", candidate_job_ids=["job-1", "job-2"])

    assert result == [{"id": "job-1", "similarity": 0.95}, {"id": "job-2", "similarity": 0.80}]


# ---------------------------------------------------------------------------
# match_user — full path
# ---------------------------------------------------------------------------

def test_match_user_gates_then_reranks_survivors_only(fake_client, profile):
    jobs = [
        _job("pass-1", location_field="Remote, Global", mode="remote", remote_scope="global"),
        _job("pass-2", location_field="Sundsvall, on-site", mode="onsite"),
        _job("fail-1", location_field="Nairobi, Kenya", mode="onsite"),
    ]
    fake_client.rpc_responses["match_v2_top_jobs"] = [
        {"id": "pass-2", "similarity": 0.88}, {"id": "pass-1", "similarity": 0.72},
    ]

    result = match_user(fake_client, user_id="u1", profile=profile, profile_embedding="[0.1]", jobs=jobs)

    assert result["gate_survivor_count"] == 2
    [(_, rpc_params)] = fake_client.rpc_calls
    assert set(rpc_params["candidate_job_ids"]) == {"pass-1", "pass-2"}  # fail-1 excluded from the RPC call
    assert [t["id"] for t in result["top"]] == ["pass-2", "pass-1"]  # RPC ordering preserved
    assert result["top"][0]["similarity"] == 0.88
    assert result["top"][0]["title"] == "Title pass-2"  # job fields merged back in


def test_match_user_no_survivors_skips_rpc_call(fake_client, profile):
    jobs = [_job("fail-1", location_field="Nairobi, Kenya", mode="onsite")]

    result = match_user(fake_client, user_id="u1", profile=profile, profile_embedding="[0.1]", jobs=jobs)

    assert result == {"gate_survivor_count": 0, "top": [], "gate_rejection_counts": {"location": 1}}
    assert fake_client.rpc_calls == []


def test_match_user_respects_top_n_param(fake_client, profile):
    jobs = [_job("pass-1"), _job("pass-2")]
    fake_client.rpc_responses["match_v2_top_jobs"] = [{"id": "pass-1", "similarity": 0.9}]

    match_user(fake_client, user_id="u1", profile=profile, profile_embedding="[0.1]", jobs=jobs, top_n=1)

    [(_, rpc_params)] = fake_client.rpc_calls
    assert rpc_params["match_limit"] == 1


def test_top_n_default_is_40():
    assert TOP_N == 40


def test_match_user_defaults_to_top_n_40(fake_client, profile):
    jobs = [_job("pass-1")]
    fake_client.rpc_responses["match_v2_top_jobs"] = []

    match_user(fake_client, user_id="u1", profile=profile, profile_embedding="[0.1]", jobs=jobs)

    [(_, rpc_params)] = fake_client.rpc_calls
    assert rpc_params["match_limit"] == 40


# ---------------------------------------------------------------------------
# fetch_matched_job_ids
# ---------------------------------------------------------------------------

def test_fetch_matched_job_ids_returns_users_matched_jobs(fake_client):
    fake_client.seed("matches_v2", [
        {"user_id": "u1", "job_id": "job-1"},
        {"user_id": "u1", "job_id": "job-2"},
        {"user_id": "u2", "job_id": "job-3"},
    ])

    result = fetch_matched_job_ids(fake_client, "u1")

    assert result == {"job-1", "job-2"}


def test_fetch_matched_job_ids_empty_when_no_matches(fake_client):
    assert fetch_matched_job_ids(fake_client, "u1") == set()


def test_fetch_matched_job_ids_scoped_to_user(fake_client):
    fake_client.seed("matches_v2", [{"user_id": "u2", "job_id": "job-1"}])

    assert fetch_matched_job_ids(fake_client, "u1") == set()
