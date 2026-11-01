"""Unit tests for ProbablyGood source parse_hits() and helpers — no network calls."""
from __future__ import annotations

from datetime import date

import pytest

from src.sources import probablygood
from src.sources.probablygood import (
    _extract_comp,
    _extract_deadline,
    _strip_html,
    _tag_names,
    fetch_raw_hits_bucketed,
    fetch_search_key,
    parse_hits,
)


# ---------------------------------------------------------------------------
# Fixture helper — real field names from the ProbablyGood Algolia index
# ---------------------------------------------------------------------------

def hit(**overrides) -> dict:
    base = {
        "title": "Head of Data",
        "org": {"name": "GiveDirectly", "id": "givedirectly"},
        "url_external": "https://givedirectly.org/jobs/head-of-data",
        "locations": [{"name": "Remote", "id": "remote"}],
        "tags_experience": [{"name": "Senior", "id": "senior"}],
        "tags_area": [{"name": "Global health and development", "id": "global-health"}],
        "tags_skill": [{"name": "Data analysis", "id": "data-analysis"}],
        "tags_workload": [{"name": "Full-time", "id": "full-time"}],
        "salary_text": "$120,000",
        "closes_at_unix": 1893456000,  # 2030-01-01
        "description": "<p>Lead our global data team.</p>",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# parse_hits — basic field mapping
# ---------------------------------------------------------------------------

def test_title_mapped():
    [p] = parse_hits([hit()])
    assert p.title == "Head of Data"


def test_org_from_nested_name():
    [p] = parse_hits([hit()])
    assert p.org == "GiveDirectly"


def test_source_is_probably_good():
    [p] = parse_hits([hit()])
    assert p.source == "probably_good"


def test_url_from_url_external():
    [p] = parse_hits([hit()])
    assert p.url == "https://givedirectly.org/jobs/head-of-data"


def test_location_from_locations_list():
    [p] = parse_hits([hit(locations=[{"name": "Remote"}])])
    assert p.location == "Remote"


def test_location_multiple_values_joined():
    [p] = parse_hits([hit(locations=[{"name": "London"}, {"name": "Remote"}])])
    assert p.location == "London, Remote"


def test_location_none_when_empty():
    [p] = parse_hits([hit(locations=[])])
    assert p.location is None


def test_seniority_from_tags_experience():
    [p] = parse_hits([hit(tags_experience=[{"name": "Senior"}, {"name": "Mid-level"}])])
    assert p.seniority == "Senior, Mid-level"


def test_seniority_none_when_empty():
    [p] = parse_hits([hit(tags_experience=[])])
    assert p.seniority is None


def test_cause_area_from_tags_area():
    [p] = parse_hits([hit(tags_area=[{"name": "Global health and development"}])])
    assert p.cause_area == "Global health and development"


def test_cause_area_none_when_empty():
    [p] = parse_hits([hit(tags_area=[])])
    assert p.cause_area is None


def test_comp_from_salary_text():
    [p] = parse_hits([hit(salary_text="$90,000–$110,000")])
    assert p.comp == "$90,000–$110,000"


def test_comp_none_when_salary_text_absent():
    h = hit()
    del h["salary_text"]
    [p] = parse_hits([h])
    assert p.comp is None


def test_comp_none_when_salary_text_empty():
    [p] = parse_hits([hit(salary_text="")])
    assert p.comp is None


def test_deadline_from_closes_at_unix():
    [p] = parse_hits([hit(closes_at_unix=1893456000)])
    assert p.deadline == date(2030, 1, 1)


def test_deadline_none_when_closes_at_unix_zero():
    [p] = parse_hits([hit(closes_at_unix=0)])
    assert p.deadline is None


def test_deadline_none_when_closes_at_unix_absent():
    h = hit()
    del h["closes_at_unix"]
    [p] = parse_hits([h])
    assert p.deadline is None


# ---------------------------------------------------------------------------
# parse_hits — raw_text content
# ---------------------------------------------------------------------------

def test_raw_text_contains_title():
    [p] = parse_hits([hit()])
    assert "Head of Data" in p.raw_text


def test_raw_text_contains_org():
    [p] = parse_hits([hit()])
    assert "GiveDirectly" in p.raw_text


def test_raw_text_contains_location():
    [p] = parse_hits([hit(locations=[{"name": "Remote"}])])
    assert "Remote" in p.raw_text


def test_raw_text_contains_tags_area():
    [p] = parse_hits([hit(tags_area=[{"name": "Animal welfare"}])])
    assert "Animal welfare" in p.raw_text


def test_raw_text_contains_tags_skill():
    [p] = parse_hits([hit(tags_skill=[{"name": "Python"}])])
    assert "Python" in p.raw_text


def test_raw_text_contains_tags_workload():
    [p] = parse_hits([hit(tags_workload=[{"name": "Part-time"}])])
    assert "Part-time" in p.raw_text


def test_raw_text_contains_stripped_description():
    [p] = parse_hits([hit(description="<p>Lead <b>data</b> efforts.</p>")])
    assert "Lead" in p.raw_text
    assert "data" in p.raw_text
    assert "<p>" not in p.raw_text


# ---------------------------------------------------------------------------
# parse_hits — skipping rules
# ---------------------------------------------------------------------------

def test_skips_hit_with_empty_title():
    assert parse_hits([hit(title="")]) == []


def test_skips_hit_with_missing_title():
    h = hit()
    del h["title"]
    assert parse_hits([h]) == []


def test_skips_hit_with_empty_url():
    assert parse_hits([hit(url_external="")]) == []


def test_skips_hit_with_missing_url():
    h = hit()
    del h["url_external"]
    assert parse_hits([h]) == []


def test_multiple_hits_all_returned():
    h2 = hit(title="M&E Lead", url_external="https://example.org/mne")
    assert len(parse_hits([hit(), h2])) == 2


# ---------------------------------------------------------------------------
# parse_hits — graceful handling of absent/malformed nested fields
# ---------------------------------------------------------------------------

def test_org_empty_when_org_key_absent():
    h = hit()
    del h["org"]
    [p] = parse_hits([h])
    assert p.org == ""


def test_org_empty_when_org_is_none():
    [p] = parse_hits([hit(org=None)])
    assert p.org == ""


def test_org_empty_when_org_has_no_name():
    [p] = parse_hits([hit(org={"id": "xyz"})])
    assert p.org == ""


def test_no_crash_on_empty_tag_lists():
    [p] = parse_hits([hit(
        locations=[], tags_experience=[], tags_area=[],
        tags_skill=[], tags_workload=[],
    )])
    assert p.location is None
    assert p.seniority is None


# ---------------------------------------------------------------------------
# _tag_names
# ---------------------------------------------------------------------------

def test_tag_names_extracts_names():
    assert _tag_names([{"name": "Remote"}, {"name": "London"}]) == ["Remote", "London"]


def test_tag_names_skips_missing_name():
    assert _tag_names([{"id": "xyz"}]) == []


def test_tag_names_skips_empty_name():
    assert _tag_names([{"name": ""}, {"name": "Remote"}]) == ["Remote"]


def test_tag_names_returns_empty_for_none():
    assert _tag_names(None) == []


def test_tag_names_returns_empty_for_empty_list():
    assert _tag_names([]) == []


# ---------------------------------------------------------------------------
# _extract_comp
# ---------------------------------------------------------------------------

def test_extract_comp_returns_salary_text():
    assert _extract_comp({"salary_text": "£60,000"}) == "£60,000"


def test_extract_comp_none_when_absent():
    assert _extract_comp({}) is None


def test_extract_comp_none_when_empty():
    assert _extract_comp({"salary_text": "  "}) is None


# ---------------------------------------------------------------------------
# _extract_deadline
# ---------------------------------------------------------------------------

def test_extract_deadline_from_float_timestamp():
    assert _extract_deadline({"closes_at_unix": 1893456000.0}) == date(2030, 1, 1)


def test_extract_deadline_none_when_zero():
    assert _extract_deadline({"closes_at_unix": 0}) is None


def test_extract_deadline_none_when_negative():
    assert _extract_deadline({"closes_at_unix": -1}) is None


def test_extract_deadline_none_when_absent():
    assert _extract_deadline({}) is None


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    assert "<p>" not in _strip_html("<p>Hello <b>world</b></p>")


def test_strip_html_preserves_text():
    assert "Hello" in _strip_html("<p>Hello</p>")


def test_strip_html_empty_string():
    assert _strip_html("") == ""


# ---------------------------------------------------------------------------
# fetch_search_key() — GraphQL key fetch, no real network (httpx.Client
# monkeypatched as a context manager).
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict, text: str | None = None):
        self._payload = payload
        self.text = text if text is not None else str(payload)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpxClient:
    def __init__(self, post_fn):
        self._post_fn = post_fn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, url, *, headers, json):
        return self._post_fn(url, headers=headers, json=json)


def test_fetch_search_key_query_includes_jobs_object_id_field():
    """The exact bug this fixes: the backend started returning data:null for
    the old query (missing this field) starting mid-September — confirmed
    live that adding it back (and nothing else — no header/cookie changes
    were needed) restores a 200 with real data."""
    assert "jobs_object_id_field" in probablygood._GRAPHQL_BODY["query"]


def test_fetch_search_key_raises_on_data_null(monkeypatch):
    def fake_post(url, *, headers, json):
        return _FakeResponse({"data": None, "errors": [{"message": "An unknown error occurred."}]})

    monkeypatch.setattr(probablygood.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_post))

    with pytest.raises(RuntimeError, match="data:null"):
        fetch_search_key()


def test_fetch_search_key_returns_app_id_api_key_index(monkeypatch):
    def fake_post(url, *, headers, json):
        return _FakeResponse({"data": {"algolia_search_key": {
            "app_id": "APPID", "api_key": "KEY", "index_name_jobs": "jobs_prod",
        }}})

    monkeypatch.setattr(probablygood.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_post))

    app_id, api_key, index = fetch_search_key()

    assert (app_id, api_key, index) == ("APPID", "KEY", "jobs_prod")


# ---------------------------------------------------------------------------
# fetch_raw_hits_bucketed() — facet-bucketed fetch, no network. Key fetch via
# httpx.Client (above); the bucketed Algolia queries via module-level
# httpx.post, same convention as src/sources/eightyk.py's equivalent tests.
# ---------------------------------------------------------------------------

def _bucket_hit(object_id: str) -> dict:
    return {"objectID": object_id, "title": f"Job {object_id}"}


def _patch_key_fetch(monkeypatch):
    def fake_post(url, *, headers, json):
        return _FakeResponse({"data": {"algolia_search_key": {
            "app_id": "APPID", "api_key": "KEY", "index_name_jobs": "jobs_prod",
        }}})
    monkeypatch.setattr(probablygood.httpx, "Client", lambda timeout=None: _FakeHttpxClient(fake_post))


class _FakeHttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_raw_hits_bucketed_unions_and_dedupes_across_buckets(monkeypatch):
    _patch_key_fetch(monkeypatch)

    def fake_algolia_post(url, *, headers, json, timeout=30):
        if "facets" in json:
            return _FakeHttpResponse({"nbHits": 3, "facets": {"tags_area.name": {"A": 2, "B": 2}}})
        value = json["facetFilters"][0][0].split(":", 1)[1]
        if value == "A":
            return _FakeHttpResponse({"hits": [_bucket_hit("job-1"), _bucket_hit("job-2")], "nbPages": 1})
        return _FakeHttpResponse({"hits": [_bucket_hit("job-1"), _bucket_hit("job-3")], "nbPages": 1})

    monkeypatch.setattr(probablygood.httpx, "post", fake_algolia_post)

    result = fetch_raw_hits_bucketed()

    assert result.nb_hits == 3
    assert result.fetched == 3
    assert result.complete is True
    assert {h["objectID"] for h in result.hits} == {"job-1", "job-2", "job-3"}


def test_fetch_raw_hits_bucketed_incomplete_when_union_undercounts_nb_hits(monkeypatch):
    _patch_key_fetch(monkeypatch)

    def fake_algolia_post(url, *, headers, json, timeout=30):
        if "facets" in json:
            return _FakeHttpResponse({"nbHits": 5, "facets": {"tags_area.name": {"A": 2}}})
        return _FakeHttpResponse({"hits": [_bucket_hit("job-1"), _bucket_hit("job-2")], "nbPages": 1})

    monkeypatch.setattr(probablygood.httpx, "post", fake_algolia_post)

    result = fetch_raw_hits_bucketed()

    assert result.fetched == 2
    assert result.nb_hits == 5
    assert result.complete is False


def test_fetch_raw_hits_bucketed_marks_incomplete_when_a_bucket_itself_hits_the_page_cap(monkeypatch):
    _patch_key_fetch(monkeypatch)

    def fake_algolia_post(url, *, headers, json, timeout=30):
        if "facets" in json:
            return _FakeHttpResponse({"nbHits": 1000, "facets": {"tags_area.name": {"A": 1000}}})
        return _FakeHttpResponse({"hits": [_bucket_hit(f"job-{i}") for i in range(1000)], "nbPages": 1})

    monkeypatch.setattr(probablygood.httpx, "post", fake_algolia_post)

    result = fetch_raw_hits_bucketed()

    assert result.fetched == 1000
    assert result.complete is False


def test_fetch_raw_hits_bucketed_falls_back_when_facet_has_no_values(monkeypatch):
    _patch_key_fetch(monkeypatch)

    def fake_algolia_post(url, *, headers, json, timeout=30):
        if "facets" in json:
            return _FakeHttpResponse({"nbHits": 2, "facets": {"tags_area.name": {}}})
        return _FakeHttpResponse({"hits": [_bucket_hit("job-1"), _bucket_hit("job-2")], "nbPages": 1})

    monkeypatch.setattr(probablygood.httpx, "post", fake_algolia_post)

    result = fetch_raw_hits_bucketed()

    assert result.fetched == 2
    assert result.nb_hits == 2
    assert result.complete is True
    assert result.bucket_sizes == {}
