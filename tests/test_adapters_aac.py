"""Tests for src/adapters/aac.py: WP REST pagination/taxonomy fetch (network
mocked via a fake httpx.Client, same pattern as test_sources_probablygood.py)
and the pure parse step (no network) — field mapping, the "No description
found" placeholder, unknown-taxonomy-term logging, and the canonical-URL
fallback rule (generic/bare company_website pages, and a batch-wide
different-title collision on a shared URL)."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.adapters import aac
from src.adapters.aac import (
    AACAdapter,
    SOURCE_BOARD,
    _apply_url,
    _fetch_all_pages,
    _fetch_taxonomies,
    _looks_like_generic_apply_url,
    parse_listings,
)
from src.adapters.base import FetchResult, JobPosting

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "aac"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES_DIR / name).read_text())


def _taxonomy_terms() -> dict[str, dict[int, str]]:
    raw = json.loads((_FIXTURES_DIR / "taxonomy_terms.json").read_text())
    return {tax: {int(tid): t["name"] for tid, t in terms.items()} for tax, terms in raw.items()}


def _job(**overrides) -> dict:
    defaults = dict(
        id=1,
        link="https://animaladvocacycareers.org/job/some-role/",
        title={"rendered": "Some Role"},
        content={"rendered": "<p>Do the <b>thing</b>.</p>"},
        meta={"_company_name": "Some Org", "_company_website": "https://example.org/jobs/123"},
        job_listing_location=[],
        job_remote=[],
        job_function=[],
        job_filter=[],
        job_type=[],
        organisation_type=[],
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Field mapping — real fixtures
# ---------------------------------------------------------------------------

def test_parse_listings_peta_fixture_maps_fields():
    job = _load_fixture("remote_fulltime_peta.json")
    terms = _taxonomy_terms()

    [posting] = parse_listings([job], terms)

    assert posting.org == "PETA"
    assert posting.title == "Social Media Content Creator"
    assert posting.url == "https://www.peta.org/about-peta/work-at-peta/search-jobs-peta"
    assert posting.source_board == SOURCE_BOARD
    assert "<" not in posting.description
    assert len(posting.description) > 100
    assert posting.location_field == "United States, Remote"


def test_parse_listings_unescapes_html_entities_in_title():
    job = _load_fixture("onsite_specialist.json")
    [posting] = parse_listings([job], _taxonomy_terms())
    assert "&" not in posting.title or "&amp;" not in posting.title
    assert posting.title == "Specialist, Data And Development Operations"


def test_parse_listings_onsite_fixture_location_field():
    job = _load_fixture("onsite_specialist.json")
    [posting] = parse_listings([job], _taxonomy_terms())
    assert posting.location_field == "United States, On-site"


def test_parse_listings_worldwide_fixture_location_field():
    job = _load_fixture("remote_worldwide.json")
    [posting] = parse_listings([job], _taxonomy_terms())
    assert posting.location_field == "Worldwide, Remote"


def test_parse_listings_multi_location_fixture_joins_all_countries():
    job = _load_fixture("multi_location_remote.json")
    [posting] = parse_listings([job], _taxonomy_terms())
    assert posting.location_field == "China, United States, Remote"


def test_parse_listings_internship_fixture_preserves_full_description():
    job = _load_fixture("internship_remote.json")
    [posting] = parse_listings([job], _taxonomy_terms())
    assert posting.org == "Animal Legal Defense Fund"
    assert posting.url == "https://recruiting.paylocity.com/Recruiting/Jobs/Details/4479370"


# ---------------------------------------------------------------------------
# "No description found" placeholder, or any empty stripped description ->
# the listing is SKIPPED entirely (no JobPosting at all), not just emptied.
# ---------------------------------------------------------------------------

def test_parse_listings_no_description_placeholder_is_skipped():
    job = _job(content={"rendered": "No description found"})
    assert parse_listings([job], {}) == []


def test_parse_listings_no_description_placeholder_case_insensitive():
    job = _job(content={"rendered": "<p>NO DESCRIPTION FOUND</p>"})
    assert parse_listings([job], {}) == []


def test_parse_listings_genuinely_empty_content_is_skipped():
    job = _job(content={"rendered": ""})
    assert parse_listings([job], {}) == []


def test_parse_listings_real_short_description_not_treated_as_placeholder():
    job = _job(content={"rendered": "Short but real."})
    [posting] = parse_listings([job], {})
    assert posting.description == "Short but real."


def test_parse_listings_logs_skipped_empty_description_count(caplog):
    jobs = [
        _job(id=1, content={"rendered": "No description found"}),
        _job(id=2, content={"rendered": ""}),
        _job(id=3, content={"rendered": "Real description text."}),
    ]
    with caplog.at_level("WARNING"):
        postings = parse_listings(jobs, {})

    assert len(postings) == 1
    assert any("skipped 2 listing" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Required-field skip / malformed-row tolerance
# ---------------------------------------------------------------------------

def test_parse_listings_skips_job_with_no_title():
    job = _job(title={"rendered": ""})
    assert parse_listings([job], {}) == []


def test_parse_listings_skips_malformed_job_without_crashing_whole_batch():
    good = _job(id=1)
    malformed = {"id": 2, "title": {"rendered": "Broken"}}  # no "link" key
    postings = parse_listings([malformed, good], {})
    assert len(postings) == 1
    assert postings[0].org == "Some Org"


# ---------------------------------------------------------------------------
# Unknown taxonomy term id -> logged, omitted from location_field
# ---------------------------------------------------------------------------

def test_parse_listings_unknown_term_id_omitted_and_logged(caplog):
    job = _job(job_listing_location=[174, 99999], job_remote=[89])
    terms = {"job_listing_location": {174: "United States"}, "job_remote": {89: "Remote"}}

    with caplog.at_level("WARNING"):
        [posting] = parse_listings([job], terms)

    assert posting.location_field == "United States, Remote"
    assert any("99999" in r.message or "unknown taxonomy term" in r.message.lower() for r in caplog.records)


def test_parse_listings_unknown_term_in_non_location_taxonomy_still_logged(caplog):
    job = _job(job_type=[99999])
    with caplog.at_level("WARNING"):
        parse_listings([job], {})
    assert any("job_type" in r.message for r in caplog.records)


def test_parse_listings_no_location_terms_is_none():
    job = _job(job_listing_location=[], job_remote=[])
    [posting] = parse_listings([job], {})
    assert posting.location_field is None


# ---------------------------------------------------------------------------
# _looks_like_generic_apply_url — the three confirmed untrustworthy patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://greenheartrecruitment.com/",
    "https://greenheartrecruitment.com",
    "https://www.svme.org/",
    "https://www.animaladvocacyafrica.org/careers",
    "https://www.animaladvocacyafrica.org/careers/",
    "https://altprotein.jobs/?s=br9j68ha",
])
def test_looks_like_generic_apply_url_true_for_confirmed_cases(url):
    assert _looks_like_generic_apply_url(url) is True


@pytest.mark.parametrize("url", [
    "https://recruiting.paylocity.com/Recruiting/Jobs/Details/4479370",
    "https://job-boards.greenhouse.io/farmsanctuary/jobs/8040479003",
    "https://ciwf.livevacancies.co.uk/#/job/details/261",
])
def test_looks_like_generic_apply_url_false_for_job_specific_urls(url):
    assert _looks_like_generic_apply_url(url) is False


# ---------------------------------------------------------------------------
# _apply_url — single-job candidate selection (before batch-collision check)
# ---------------------------------------------------------------------------

def test_apply_url_uses_company_website_by_default():
    job = _job(meta={"_company_website": "https://example.org/jobs/123"})
    assert _apply_url(job) == "https://example.org/jobs/123"


def test_apply_url_falls_back_to_link_when_company_website_empty():
    job = _job(meta={"_company_website": ""}, link="https://animaladvocacycareers.org/job/x/")
    assert _apply_url(job) == "https://animaladvocacycareers.org/job/x"


def test_apply_url_falls_back_to_link_when_company_website_is_bare():
    job = _job(meta={"_company_website": "https://greenheartrecruitment.com/"}, link="https://animaladvocacycareers.org/job/x/")
    assert _apply_url(job) == "https://animaladvocacycareers.org/job/x"


def test_apply_url_preserves_hash_routed_fragment():
    job = _job(meta={"_company_website": "https://ciwf.livevacancies.co.uk/#/job/details/261"})
    assert _apply_url(job) == "https://ciwf.livevacancies.co.uk/#/job/details/261"


# ---------------------------------------------------------------------------
# parse_listings batch-level collision handling
# ---------------------------------------------------------------------------

def test_parse_listings_same_url_same_title_keeps_shared_url():
    """Rule 3: a legitimate duplicate (two AAC listing ids, same role, same
    apply link) is left alone at the adapter level — src.ingest.py's
    existing canonical_url dedup collapses it to one `jobs` row downstream
    (confirmed in the Phase 2 prep report), so the adapter doesn't need to
    deduplicate itself, only avoid a FALSE positive fallback."""
    shared_url = "https://example.org/apply"
    a = _job(id=1, title={"rendered": "Volunteer Treasurer"}, meta={"_company_website": shared_url})
    b = _job(id=2, title={"rendered": "Volunteer Treasurer"}, meta={"_company_website": shared_url})

    postings = parse_listings([a, b], {})

    assert len(postings) == 2
    assert postings[0].url == postings[1].url == shared_url


def test_parse_listings_same_url_different_title_falls_back_to_link_for_both():
    """Rule 3's collision case: a shared company_website can't distinguish
    two DIFFERENT roles — both must fall back to their own AAC link instead
    of silently merging into one `jobs` row under the generic URL."""
    shared_url = "https://example.org/careers-generic"
    a = _job(id=1, title={"rendered": "Role A"}, link="https://animaladvocacycareers.org/job/role-a/",
             meta={"_company_website": shared_url})
    b = _job(id=2, title={"rendered": "Role B"}, link="https://animaladvocacycareers.org/job/role-b/",
             meta={"_company_website": shared_url})

    postings = parse_listings([a, b], {})

    by_title = {p.title: p.url for p in postings}
    assert by_title["Role A"] == "https://animaladvocacycareers.org/job/role-a"
    assert by_title["Role B"] == "https://animaladvocacycareers.org/job/role-b"
    assert by_title["Role A"] != by_title["Role B"]


def test_parse_listings_ciwf_four_distinct_jobs_four_distinct_urls_no_fallback_needed():
    """The fragment-preservation fix (src.job_normalize.normalize_job_url)
    already makes these 4 distinct without needing the different-title
    fallback at all — confirming the two safety nets are independent."""
    jobs = [
        _job(id=i, title={"rendered": f"CIWF Role {i}"},
             meta={"_company_website": f"https://ciwf.livevacancies.co.uk/#/job/details/{261-i}"})
        for i in range(4)
    ]

    postings = parse_listings(jobs, {})

    assert len({p.url for p in postings}) == 4


def test_parse_listings_three_way_collision_different_titles_all_fall_back():
    shared_url = "https://example.org/apply"
    jobs = [
        _job(id=i, title={"rendered": f"Role {i}"}, link=f"https://animaladvocacycareers.org/job/role-{i}/",
             meta={"_company_website": shared_url})
        for i in range(3)
    ]
    postings = parse_listings(jobs, {})
    assert len({p.url for p in postings}) == 3


# ---------------------------------------------------------------------------
# AACAdapter.fetch() — network mocked at _fetch_all_pages/_fetch_taxonomies
# ---------------------------------------------------------------------------

def test_fetch_wraps_pages_into_fetch_result(monkeypatch):
    monkeypatch.setattr(aac, "_fetch_all_pages", lambda: ([_job()], True, 1))
    monkeypatch.setattr(aac, "_fetch_taxonomies", lambda: {})

    result = AACAdapter().fetch()

    assert isinstance(result, FetchResult)
    assert len(result.postings) == 1
    assert isinstance(result.postings[0], JobPosting)
    assert result.fetched == 1
    assert result.nb_hits == 1
    assert result.complete is True


def test_fetch_propagates_incomplete_flag(monkeypatch):
    monkeypatch.setattr(aac, "_fetch_all_pages", lambda: ([], False, 132))
    monkeypatch.setattr(aac, "_fetch_taxonomies", lambda: {})

    result = AACAdapter().fetch()

    assert result.complete is False
    assert result.fetched == 0
    assert result.nb_hits == 132


# ---------------------------------------------------------------------------
# _fetch_all_pages / _fetch_taxonomies — network mocked via a fake
# httpx.Client (same pattern as test_sources_probablygood.py's
# _FakeHttpxClient).
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, data, headers=None, status_ok=True):
        self._data = data
        self.headers = headers or {}
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._data


class _FakeHttpxClient:
    def __init__(self, get_fn):
        self._get_fn = get_fn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, *, params=None):
        return self._get_fn(url, params=params)


def test_fetch_all_pages_single_page(monkeypatch):
    def fake_get(url, *, params):
        return _FakeResponse([_job(), _job(id=2)], headers={"X-WP-Total": "2", "X-WP-TotalPages": "1"})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    rows, complete, total = _fetch_all_pages()

    assert len(rows) == 2
    assert complete is True
    assert total == 2


def test_fetch_all_pages_follows_pagination_to_total_pages(monkeypatch):
    calls = []

    def fake_get(url, *, params):
        calls.append(params["page"])
        if params["page"] == 1:
            return _FakeResponse([_job(id=1)], headers={"X-WP-Total": "2", "X-WP-TotalPages": "2"})
        return _FakeResponse([_job(id=2)], headers={})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    rows, complete, total = _fetch_all_pages()

    assert calls == [1, 2]
    assert len(rows) == 2
    assert complete is True
    assert total == 2


def test_fetch_all_pages_partial_failure_returns_what_succeeded(monkeypatch):
    def fake_get(url, *, params):
        if params["page"] == 1:
            return _FakeResponse([_job(id=1)], headers={"X-WP-Total": "2", "X-WP-TotalPages": "2"})
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    rows, complete, total = _fetch_all_pages()

    assert len(rows) == 1  # page 1's data survives
    assert complete is False
    assert total == 2


def test_fetch_all_pages_count_mismatch_marks_incomplete(monkeypatch):
    def fake_get(url, *, params):
        return _FakeResponse([_job()], headers={"X-WP-Total": "5", "X-WP-TotalPages": "1"})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    rows, complete, total = _fetch_all_pages()

    assert complete is False
    assert total == 5
    assert len(rows) == 1


def test_fetch_taxonomies_builds_id_to_name_map(monkeypatch):
    def fake_get(url, *, params):
        if url.endswith("job_remote"):
            return _FakeResponse([{"id": 89, "name": "Remote"}], headers={"X-WP-TotalPages": "1"})
        return _FakeResponse([], headers={"X-WP-TotalPages": "1"})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    taxonomies = _fetch_taxonomies()

    assert taxonomies["job_remote"] == {89: "Remote"}
    assert set(taxonomies.keys()) == {
        "job_listing_location", "job_remote", "job_function", "job_filter", "job_type", "organisation_type",
    }


def test_fetch_taxonomies_failed_endpoint_logs_and_contributes_empty_map(monkeypatch, caplog):
    def fake_get(url, *, params):
        if url.endswith("job_filter"):
            raise httpx.ConnectError("boom")
        return _FakeResponse([], headers={"X-WP-TotalPages": "1"})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    with caplog.at_level("ERROR"):
        taxonomies = _fetch_taxonomies()

    assert taxonomies["job_filter"] == {}
    assert any("job_filter" in r.message for r in caplog.records)


def test_fetch_taxonomies_multi_page_warns_incomplete(monkeypatch, caplog):
    def fake_get(url, *, params):
        return _FakeResponse([{"id": 1, "name": "x"}], headers={"X-WP-TotalPages": "2"})

    monkeypatch.setattr(aac.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_get))

    with caplog.at_level("WARNING"):
        _fetch_taxonomies()

    assert any("INCOMPLETE" in r.message for r in caplog.records)
