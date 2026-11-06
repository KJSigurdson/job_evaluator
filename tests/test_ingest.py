"""Tests for src/ingest.py: dedup/content-change detection and batched writes
against `jobs`/`job_aliases`/`run_metrics`."""
from __future__ import annotations

import pytest

from src.adapters.base import JobPosting
from src.ingest import IngestCounts, ingest
from src.job_normalize import content_hash, normalize_job_url


def _simulate_touch_jobs_last_seen(client):
    """A realistic-enough fake of the real SQL function: bumps last_seen on
    every seeded `jobs` row whose canonical_url is in the RPC's urls param,
    returns the count actually touched. FakeSupabaseClient can't execute
    real SQL, so without this, a test asserting last_seen actually changed
    would see nothing happen (the default handler in
    _default_touch_jobs_last_seen_rpc already installs this for the
    `fake_client` fixture; tests that build their own FakeSupabaseClient()
    directly must call this themselves — see its call sites below)."""
    def _touch(params):
        urls = set(params["urls"])
        touched = 0
        for row in client.table("jobs").rows:
            if row["canonical_url"] in urls:
                row["last_seen"] = params["seen_at"]
                touched += 1
        return touched
    client.rpc_responses["touch_jobs_last_seen"] = _touch


@pytest.fixture(autouse=True)
def _default_touch_jobs_last_seen_rpc(fake_client):
    """touch_jobs_last_seen is called for every ingest() run with at least
    one 'unchanged'/cross-board/org-title-match touch. Installed by default
    on every test's `fake_client` so a test that doesn't care about this
    specific write still doesn't have to configure it (FakeSupabaseClient
    raises on an unconfigured rpc_responses entry) — tests that DO care can
    just read `fake_client.table("jobs").rows` afterward, same as any other
    write in this file."""
    _simulate_touch_jobs_last_seen(fake_client)


def _posting(**overrides) -> JobPosting:
    defaults = dict(
        url="https://jobs.80000hours.org/head-of-data",
        org="GiveDirectly",
        title="Head of Data",
        description="We are looking for a Head of Data to build our BI function.",
        location_field="Remote, Global",
        source_board="80000_hours",
    )
    defaults.update(overrides)
    return JobPosting(**defaults)


# ---------------------------------------------------------------------------
# New jobs
# ---------------------------------------------------------------------------

def test_ingest_new_posting_inserts_a_job(fake_client):
    counts = ingest([_posting()], client=fake_client)

    assert counts == IngestCounts(ingested=1, new=1)
    rows = fake_client.table("jobs").rows
    assert len(rows) == 1
    assert rows[0]["canonical_url"] == normalize_job_url(_posting().url)
    assert rows[0]["status"] == "pending_enrich"
    assert rows[0]["content_hash"] == content_hash(_posting().description)
    assert rows[0]["first_seen"] == rows[0]["last_seen"]
    assert rows[0]["location_field"] == _posting().location_field


def test_ingest_new_posting_batches_writes_in_one_insert_call(fake_client):
    postings = [_posting(url=f"https://jobs.80000hours.org/role-{i}", title=f"Role {i}") for i in range(10)]
    ingest(postings, client=fake_client)

    assert len(fake_client.table("jobs").insert_calls) == 1
    assert len(fake_client.table("jobs").insert_calls[0]) == 10


def test_ingest_writes_run_metrics_row(fake_client):
    ingest([_posting()], client=fake_client)

    rows = fake_client.table("run_metrics").rows
    assert len(rows) == 1
    assert rows[0]["jobs_ingested"] == 1
    assert rows[0]["per_board"]["80000_hours"]["new"] == 1


def test_ingest_run_metrics_row_stamped_with_run_id_stage_trigger(fake_client):
    ingest([_posting()], client=fake_client, run_id="run-abc", trigger="scheduled")

    [row] = fake_client.table("run_metrics").rows
    assert row["run_id"] == "run-abc"
    assert row["stage"] == "ingest"
    assert row["trigger"] == "scheduled"


def test_ingest_run_metrics_defaults_run_id_and_trigger_to_none(fake_client):
    ingest([_posting()], client=fake_client)

    [row] = fake_client.table("run_metrics").rows
    assert row["run_id"] is None
    assert row["trigger"] is None
    assert row["stage"] == "ingest"


def test_ingest_survives_run_metrics_insert_failure(fake_client, caplog):
    """insert_run_metrics() never raises — a schema-drift failure writing
    the observability row must not undo or block on the jobs/job_aliases
    writes already committed above it."""
    fake_client.table("run_metrics").raise_on_execute["insert"] = Exception("PGRST204: boom")

    with caplog.at_level("ERROR"):
        counts = ingest([_posting()], client=fake_client)  # must not raise

    assert counts == IngestCounts(ingested=1, new=1)
    assert fake_client.table("jobs").rows  # the real write survived
    assert fake_client.table("run_metrics").rows == []
    assert any(r.levelname == "ERROR" for r in caplog.records)


# ---------------------------------------------------------------------------
# Same URL, same content — last_seen touch only
# ---------------------------------------------------------------------------

def test_ingest_unchanged_content_only_touches_last_seen(fake_client):
    p = _posting()
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "enriched",
        "enrichment": {"summary": "existing enrichment"}, "first_seen": "2026-01-01T00:00:00+00:00",
        "last_seen": "2026-01-01T00:00:00+00:00",
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, unchanged=1)
    row = fake_client.table("jobs").rows[0]
    # Untouched — the whole point of the unchanged path.
    assert row["status"] == "enriched"
    assert row["enrichment"] == {"summary": "existing enrichment"}
    assert row["content_hash"] == content_hash(p.description)
    # Touched.
    assert row["last_seen"] != "2026-01-01T00:00:00+00:00"
    assert fake_client.table("jobs").insert_calls == []


def test_ingest_unchanged_touch_rpc_call_carries_only_urls_and_seen_at(fake_client):
    """The NULL-fill regression this design originally existed to avoid (a
    full-row upsert whose payload could, by construction mistake, grow to
    reference org/title/status/enrichment/embedding/enriched_at) is now
    structurally impossible: touch_jobs_last_seen's RPC params are exactly
    {urls, seen_at} — there is no payload shape for those columns to ever
    leak into. Also confirms this path no longer goes through .upsert() at
    all (see 2026-11-06's touch_jobs_last_seen RPC change)."""
    p = _posting()
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
    }])

    ingest([p], client=fake_client)

    [(fn_name, params)] = fake_client.rpc_calls
    assert fn_name == "touch_jobs_last_seen"
    assert set(params.keys()) == {"urls", "seen_at"}
    assert params["urls"] == [canonical]
    assert fake_client.table("jobs").upsert_calls == []


# ---------------------------------------------------------------------------
# _write_unchanged_last_seen — chunking + per-chunk logging
# ---------------------------------------------------------------------------

def test_write_unchanged_last_seen_chunks_at_1000_urls(fake_client):
    from src.ingest import _TOUCH_CHUNK_SIZE, _write_unchanged_last_seen

    assert _TOUCH_CHUNK_SIZE == 1000
    urls = [f"https://example.org/job-{i}" for i in range(2500)]
    fake_client.rpc_responses["touch_jobs_last_seen"] = lambda params: len(params["urls"])

    _write_unchanged_last_seen(fake_client, urls, "2026-11-06T00:00:00+00:00")

    assert len(fake_client.rpc_calls) == 3  # 1000 + 1000 + 500
    chunk_sizes = [len(params["urls"]) for _fn, params in fake_client.rpc_calls]
    assert chunk_sizes == [1000, 1000, 500]
    # every chunk is disjoint and their union is exactly the input, in order
    assert [url for _fn, params in fake_client.rpc_calls for url in params["urls"]] == urls
    for fn_name, params in fake_client.rpc_calls:
        assert fn_name == "touch_jobs_last_seen"
        assert params["seen_at"] == "2026-11-06T00:00:00+00:00"


def test_write_unchanged_last_seen_single_chunk_under_limit(fake_client):
    from src.ingest import _write_unchanged_last_seen

    fake_client.rpc_responses["touch_jobs_last_seen"] = lambda params: len(params["urls"])
    urls = ["https://example.org/job-1", "https://example.org/job-2"]

    _write_unchanged_last_seen(fake_client, urls, "2026-11-06T00:00:00+00:00")

    assert len(fake_client.rpc_calls) == 1
    [(fn_name, params)] = fake_client.rpc_calls
    assert params["urls"] == urls


def test_write_unchanged_last_seen_logs_touched_count_and_elapsed_ms(fake_client, caplog):
    from src.ingest import _write_unchanged_last_seen

    fake_client.rpc_responses["touch_jobs_last_seen"] = lambda params: len(params["urls"])

    with caplog.at_level("INFO"):
        _write_unchanged_last_seen(fake_client, ["https://example.org/job-1"], "2026-11-06T00:00:00+00:00")

    [record] = [r for r in caplog.records if "touch_jobs_last_seen" in r.message]
    assert "touched=1" in record.message
    assert "chunk_size=1" in record.message
    assert "elapsed_ms=" in record.message


def test_write_unchanged_last_seen_logs_rpc_return_as_is_not_chunk_size(fake_client, caplog):
    """If the RPC reports fewer rows touched than URLs requested (e.g. a URL
    deleted between this run's read and write), the log must show the REAL
    number, never silently substitute len(chunk) and claim full success."""
    from src.ingest import _write_unchanged_last_seen

    fake_client.rpc_responses["touch_jobs_last_seen"] = lambda params: len(params["urls"]) - 1

    with caplog.at_level("INFO"):
        _write_unchanged_last_seen(fake_client, ["https://example.org/job-1", "https://example.org/job-2"], "now")

    [record] = [r for r in caplog.records if "touch_jobs_last_seen" in r.message]
    assert "touched=1" in record.message
    assert "chunk_size=2" in record.message


# ---------------------------------------------------------------------------
# Same URL, content changed — re-enrich
# ---------------------------------------------------------------------------

def test_ingest_changed_content_with_no_enrichment_yet_resets_status(fake_client):
    """A job that's never been successfully enriched has nothing worth
    preserving — real content change still flips it to pending_enrich, same
    as before this phase's reenrich_requested_at change."""
    p = _posting(description="Updated description text.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Old description text."),
        "raw_description": "Old description text.", "status": "pending_enrich",
        "enrichment": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, updated=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "pending_enrich"
    assert row["raw_description"] == "Updated description text."
    assert row["content_hash"] == content_hash("Updated description text.")
    assert row.get("reenrich_requested_at") is None


def test_ingest_changed_content_with_existing_enrichment_stays_matchable(fake_client):
    """A job that's already been successfully enriched once keeps
    status='enriched' through a real content change instead of dropping out
    of the matching pool for up to _RE_ENRICH_COOLDOWN_DAYS — it just gets
    reenrich_requested_at stamped so enrich_jobs.py knows a re-enrich is
    owed (see src/enrich_jobs.py's selection query)."""
    p = _posting(description="Updated description text.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Old description text."),
        "raw_description": "Old description text.", "status": "enriched",
        "enrichment": {"summary": "stale enrichment"}, "reenrich_requested_at": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, updated=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["raw_description"] == "Updated description text."
    assert row["content_hash"] == content_hash("Updated description text.")
    assert row["reenrich_requested_at"] is not None
    # enrichment itself isn't cleared by ingest — the enrich worker overwrites it
    # once it re-processes this job; the upsert payload never references that key.
    assert row["enrichment"] == {"summary": "stale enrichment"}


def test_ingest_changed_content_does_not_rebump_existing_reenrich_request(fake_client):
    """A second real content change before enrich_jobs.py ever services the
    first request must not push the cooldown clock out further by
    overwriting the original request timestamp."""
    p = _posting(description="Second updated description.")
    canonical = normalize_job_url(p.url)
    original_request_time = "2026-10-01T00:00:00+00:00"
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("First updated description."),
        "raw_description": "First updated description.", "status": "enriched",
        "enrichment": {"summary": "stale enrichment"}, "reenrich_requested_at": original_request_time,
    }])

    ingest([p], client=fake_client)

    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["reenrich_requested_at"] == original_request_time


def test_ingest_changed_upsert_payload_never_references_enrichment_fields(fake_client):
    p = _posting(description="New text")
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": normalize_job_url(p.url), "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Old text"),
    }])

    ingest([p], client=fake_client)

    [payload] = fake_client.table("jobs").upsert_calls
    assert set(payload[0].keys()) == {
        "canonical_url", "source_board", "raw_description", "location_field", "content_hash", "last_seen", "status",
    }


# ---------------------------------------------------------------------------
# No URL match, org/title match — alias, not a new job
# ---------------------------------------------------------------------------

def test_ingest_org_title_match_creates_alias_not_new_job(fake_client):
    p = _posting(url="https://otherboard.example.org/repost-of-head-of-data")
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://jobs.80000hours.org/head-of-data",
        "source_board": "80000_hours", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": "irrelevant",
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, aliased=1)
    assert fake_client.table("jobs").insert_calls == []
    aliases = fake_client.table("job_aliases").rows
    assert len(aliases) == 1
    assert aliases[0]["job_id"] == "job-1"
    assert aliases[0]["alias_url"] == normalize_job_url(p.url)
    assert aliases[0]["source_board"] == "80000_hours"


def test_ingest_org_title_match_bumps_canonical_jobs_last_seen(fake_client):
    """A job only ever re-discovered via org/title alias (e.g. re-posted under
    a new URL every run) must still have its OWN row's last_seen touched —
    otherwise a closure step keyed on last_seen would wrongly treat it as
    gone stale even though it's clearly still live (see this module's
    post-loop alias_matched_job_ids bump)."""
    p = _posting(url="https://otherboard.example.org/repost-of-head-of-data")
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://jobs.80000hours.org/head-of-data",
        "source_board": "80000_hours", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": "irrelevant", "last_seen": "2026-01-01T00:00:00+00:00",
    }])

    ingest([p], client=fake_client)

    [row] = [r for r in fake_client.table("jobs").rows if r["id"] == "job-1"]
    assert row["last_seen"] != "2026-01-01T00:00:00+00:00"


def test_ingest_org_title_match_is_case_and_whitespace_insensitive(fake_client):
    p = _posting(org="  givedirectly  ", title="HEAD OF DATA", url="https://other.example.org/x")
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://jobs.80000hours.org/head-of-data",
        "source_board": "80000_hours", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": "irrelevant",
    }])

    counts = ingest([p], client=fake_client)
    assert counts.aliased == 1


def test_ingest_two_new_postings_same_run_alias_to_each_other():
    """Two re-posts of the same role, BOTH new to this run: the second must alias
    to the first (client-generated id), not create a second jobs row."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    first = _posting(url="https://jobs.80000hours.org/head-of-data-v1")
    second = _posting(url="https://jobs.80000hours.org/head-of-data-v2")

    counts = ingest([first, second], client=client)

    assert counts.new == 1
    assert counts.aliased == 1
    assert len(client.table("jobs").rows) == 1
    [alias] = client.table("job_aliases").rows
    assert alias["job_id"] == client.table("jobs").rows[0]["id"]


def test_ingest_same_canonical_url_twice_in_one_batch_does_not_double_insert():
    """Regression: a source board can list the exact same canonical_url twice in
    one fetch (observed for real against 80,000 Hours) — not a re-post under a
    different URL, the literal same URL twice. The second occurrence must be
    treated as unchanged/updated against the first, never as a second "new" insert
    (which would crash on jobs.canonical_url's unique constraint in production)."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    _simulate_touch_jobs_last_seen(client)
    p = _posting(url="https://jobs.80000hours.org/duplicated-listing")

    counts = ingest([p, p], client=client)

    assert counts.new == 1
    assert counts.unchanged == 1  # identical content both times
    assert len(client.table("jobs").rows) == 1
    assert len(client.table("jobs").insert_calls) == 1
    assert len(client.table("jobs").insert_calls[0]) == 1  # exactly one row inserted, not two


def test_ingest_same_canonical_url_twice_different_content_updates_not_duplicate_insert():
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    first = _posting(url="https://jobs.80000hours.org/duplicated-listing", description="version one")
    second = _posting(url="https://jobs.80000hours.org/duplicated-listing", description="version two")

    counts = ingest([first, second], client=client)

    assert counts.new == 1
    assert counts.updated == 1
    assert len(client.table("jobs").rows) == 1
    assert client.table("jobs").rows[0]["content_hash"] == content_hash("version two")


def test_ingest_same_existing_canonical_url_three_times_differing_content_no_duplicate_upsert_key():
    """Regression for production crash: 'ON CONFLICT DO UPDATE command cannot affect
    row a second time'. An EXISTING job's canonical_url appearing 3+ times in one
    batch, with content alternating, must collapse to exactly one upsert entry per
    conflict-target key (last occurrence wins) — never two entries with the same
    canonical_url in one bulk upsert call."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    p = _posting(url="https://jobs.80000hours.org/existing-listing")
    canonical = normalize_job_url(p.url)
    client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("original text"),
    }])
    postings = [
        p.__class__(**{**p.__dict__, "description": "version one"}),
        p.__class__(**{**p.__dict__, "description": "version two"}),
        p.__class__(**{**p.__dict__, "description": "version one"}),  # back to a hash already seen mid-batch
    ]

    counts = ingest(postings, client=client)

    assert counts.ingested == 3
    assert counts.updated + counts.unchanged == 3
    assert len(client.table("jobs").upsert_calls) == 1  # exactly one upsert round trip
    [payload] = client.table("jobs").upsert_calls
    assert len(payload) == 1  # collapsed to one row despite 3 occurrences
    assert client.table("jobs").rows[0]["content_hash"] == content_hash("version one")


def test_ingest_same_new_canonical_url_three_times_differing_content_no_duplicate_upsert_key():
    """Same regression, but for a job that's brand-new THIS run: occurrence 1 -> insert,
    occurrences 2 and 3 must collapse into a single changed_updates upsert entry, not
    two, even though by_url only gets refreshed via the 'new job' path initially."""
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    p = _posting(url="https://jobs.80000hours.org/brand-new-listing")
    postings = [
        p.__class__(**{**p.__dict__, "description": "version one"}),
        p.__class__(**{**p.__dict__, "description": "version two"}),
        p.__class__(**{**p.__dict__, "description": "version three"}),
    ]

    counts = ingest(postings, client=client)

    assert counts.new == 1
    assert counts.updated == 2
    assert len(client.table("jobs").insert_calls) == 1
    assert len(client.table("jobs").upsert_calls) == 1
    [payload] = client.table("jobs").upsert_calls
    assert len(payload) == 1  # collapsed, not two duplicate-keyed entries
    assert client.table("jobs").rows[0]["content_hash"] == content_hash("version three")


def test_ingest_same_alias_canonical_url_twice_in_one_batch_no_duplicate_upsert_key():
    from tests.conftest import FakeSupabaseClient
    client = FakeSupabaseClient()
    _simulate_touch_jobs_last_seen(client)
    client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://jobs.80000hours.org/head-of-data",
        "source_board": "80000_hours", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": "irrelevant",
    }])
    p = _posting(url="https://otherboard.example.org/repost-of-head-of-data")

    counts = ingest([p, p], client=client)

    assert counts.aliased == 2  # both occurrences individually counted
    assert len(client.table("job_aliases").upsert_calls) == 1
    [payload] = client.table("job_aliases").upsert_calls
    assert len(payload) == 1  # but collapsed to one row — same alias_url twice would crash
    assert len(client.table("job_aliases").rows) == 1


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Same URL, different source board — cross-board rediscovery, not a content
# update (see ingest.py's module docstring: the real root cause of the
# ~300/run "updated" churn confirmed against production on 2026-08-03).
# ---------------------------------------------------------------------------

def test_ingest_cross_board_same_url_does_not_flag_updated_or_touch_text(fake_client):
    p = _posting(source_board="iap", description="Org | Title | Location | Comp | Cause")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("80k's fuller prose description."),
        "raw_description": "80k's fuller prose description.", "status": "enriched",
        "enrichment": {"summary": "existing enrichment"},
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, cross_board=1)
    row = fake_client.table("jobs").rows[0]
    # Untouched — whichever board's text is already stored wins, no re-enrich.
    assert row["status"] == "enriched"
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "80k's fuller prose description."
    assert row["content_hash"] == content_hash("80k's fuller prose description.")
    assert fake_client.table("jobs").insert_calls == []


def test_ingest_cross_board_touch_never_changes_source_board_or_content_hash(fake_client):
    """A cross-board rediscovery only ever touches last_seen now (via the
    touch_jobs_last_seen RPC) — the stored row's source_board/content_hash
    stay exactly as they were (the ORIGINAL owner's values), never
    overwritten with the incoming posting's board/hash. Structurally
    guaranteed by the new design: the RPC has no payload shape for those
    columns to ever carry (see 2026-11-06's touch_jobs_last_seen change)."""
    p = _posting(source_board="probably_good")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": "existing-hash",
        "raw_description": "existing text",
    }])

    ingest([p], client=fake_client)

    [(fn_name, params)] = fake_client.rpc_calls
    assert fn_name == "touch_jobs_last_seen"
    assert params["urls"] == [canonical]
    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "80000_hours"  # the ORIGINAL owner, not the incoming board
    assert row["content_hash"] == "existing-hash"
    assert fake_client.table("jobs").upsert_calls == []


def test_ingest_third_board_in_same_batch_still_compares_against_original_owner(fake_client):
    """A cross-board hit must NOT overwrite the in-memory cache — a later
    occurrence of the same URL from yet another board in the same batch should
    still be judged a cross-board hit too, not "changed" against the first
    cross-board occurrence's (irrelevant) content."""
    canonical = normalize_job_url("https://jobs.80000hours.org/head-of-data")
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": "GiveDirectly", "title": "Head of Data", "content_hash": content_hash("original text"),
        "raw_description": "original text",
    }])
    p1 = _posting(url=canonical, source_board="probably_good", description="pg's version")
    p2 = _posting(url=canonical, source_board="iap", description="iap's version")

    counts = ingest([p1, p2], client=fake_client)

    assert counts.cross_board == 2
    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "original text"


# ---------------------------------------------------------------------------
# Same URL, different board, STORED board is restricted (iap) — permitted
# board always wins (src/ingest.py's _RESTRICTED_SOURCE_BOARDS).
# ---------------------------------------------------------------------------

def test_ingest_permitted_board_overrides_restricted_board_same_url_no_enrichment(fake_client):
    p = _posting(source_board="80000_hours", description="80k's fuller prose description.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "iap",
        "org": p.org, "title": p.title, "content_hash": content_hash("Org | Title | Location"),
        "raw_description": "Org | Title | Location", "status": "pending_enrich", "enrichment": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, board_upgraded=1)
    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "80k's fuller prose description."
    assert row["status"] == "pending_enrich"


def test_ingest_permitted_board_overrides_restricted_board_same_url_preserves_enrichment(fake_client):
    p = _posting(source_board="80000_hours", description="80k's fuller prose description.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "iap",
        "org": p.org, "title": p.title, "content_hash": content_hash("Org | Title | Location"),
        "raw_description": "Org | Title | Location", "status": "enriched",
        "enrichment": {"summary": "stale, from IAP's terse text"}, "reenrich_requested_at": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, board_upgraded=1)
    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "80k's fuller prose description."
    assert row["status"] == "enriched"
    assert row["reenrich_requested_at"] is not None


def test_ingest_restricted_board_never_overrides_permitted_board_same_url(fake_client):
    """The reverse of the above must never happen — iap seeing a URL already
    attributed to 80k/ProbablyGood is a normal (unchanged) cross-board hit,
    not a downgrade."""
    p = _posting(source_board="iap", description="Org | Title | Location")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("80k's fuller prose description."),
        "raw_description": "80k's fuller prose description.", "status": "enriched",
        "enrichment": {"summary": "good enrichment"},
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, cross_board=1)
    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "80k's fuller prose description."


# ---------------------------------------------------------------------------
# Org/title match, matched job's CANONICAL row is restricted-board-sourced —
# permitted board wins here too, but the URL itself also changes.
# ---------------------------------------------------------------------------

def test_ingest_permitted_board_org_title_match_upgrades_restricted_canonical_job(fake_client):
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://airtable.com/iap-eoi-form",
        "source_board": "iap", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": content_hash("Org | Title | Location"),
        "raw_description": "Org | Title | Location", "status": "enriched",
        "enrichment": {"summary": "stale, from IAP's terse text"}, "reenrich_requested_at": None,
    }])
    p = _posting(
        url="https://jobs.80000hours.org/head-of-data", org="GiveDirectly", title="Head of Data",
        source_board="80000_hours", description="80k's fuller prose description.",
    )

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, board_upgraded=1)
    [row] = fake_client.table("jobs").rows
    assert row["id"] == "job-1"  # same job, re-pointed in place
    assert row["canonical_url"] == normalize_job_url(p.url)
    assert row["source_board"] == "80000_hours"
    assert row["raw_description"] == "80k's fuller prose description."
    assert row["status"] == "enriched"
    assert row["reenrich_requested_at"] is not None
    # Old IAP URL demoted to an alias, not left dangling or deleted.
    [alias] = fake_client.table("job_aliases").rows
    assert alias["alias_url"] == "https://airtable.com/iap-eoi-form"
    assert alias["job_id"] == "job-1"
    assert alias["source_board"] == "iap"


def test_ingest_restricted_board_org_title_match_against_permitted_canonical_stays_a_normal_alias(fake_client):
    """The reverse must never swap a permitted board's canonical job out for
    an iap-sourced URL — iap stays a plain alias, same as before this fix."""
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": "https://jobs.80000hours.org/head-of-data",
        "source_board": "80000_hours", "org": "GiveDirectly", "title": "Head of Data",
        "content_hash": content_hash("80k's fuller prose description."),
        "raw_description": "80k's fuller prose description.", "status": "enriched",
        "enrichment": {"summary": "good enrichment"},
    }])
    p = _posting(
        url="https://airtable.com/iap-eoi-form", org="GiveDirectly", title="Head of Data",
        source_board="iap", description="Org | Title | Location",
    )

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, aliased=1)
    [row] = fake_client.table("jobs").rows
    assert row["canonical_url"] == "https://jobs.80000hours.org/head-of-data"
    assert row["source_board"] == "80000_hours"
    [alias] = fake_client.table("job_aliases").rows
    assert alias["alias_url"] == normalize_job_url(p.url)
    assert alias["job_id"] == "job-1"


# ---------------------------------------------------------------------------
# Same URL, same board, hash differs but text is near-identical — cosmetic,
# no re-enrich spend (RE_ENRICH_SIMILARITY_THRESHOLD guardrail).
# ---------------------------------------------------------------------------

def test_ingest_minor_change_refreshes_text_without_resetting_status(fake_client):
    old_text = "A" * 500
    new_text = old_text + " Minor trailing note."  # small addition -> similarity above threshold
    p = _posting(description=new_text)
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(old_text),
        "raw_description": old_text, "status": "enriched",
        "enrichment": {"summary": "existing enrichment"},
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, minor_update=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"  # not reset to pending_enrich
    assert row["enrichment"] == {"summary": "existing enrichment"}
    assert row["raw_description"] == new_text
    assert row["content_hash"] == content_hash(new_text)


def test_ingest_substantially_different_text_still_flags_updated(fake_client):
    """Guardrail sanity check: a real content change (well below the similarity
    threshold) must still take the normal 'updated' -> pending_enrich path."""
    p = _posting(description="A completely different role description entirely.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Original text about a data role."),
        "raw_description": "Original text about a data role.", "status": "enriched",
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, updated=1)
    assert fake_client.table("jobs").rows[0]["status"] == "pending_enrich"


def test_ingest_dry_run_writes_nothing(fake_client):
    counts = ingest([_posting()], client=fake_client, dry_run=True)

    assert counts == IngestCounts(ingested=1, new=1)
    assert fake_client.table("jobs").rows == []
    assert fake_client.table("job_aliases").rows == []
    assert fake_client.table("run_metrics").rows == []


def test_ingest_dry_run_computes_full_counts_across_all_paths(fake_client):
    fake_client.seed("jobs", [
        {"id": "unchanged-job", "canonical_url": normalize_job_url("https://jobs.80000hours.org/unchanged"),
         "source_board": "80000_hours", "org": "OrgA", "title": "Role A",
         "content_hash": content_hash("same text")},
        {"id": "changed-job", "canonical_url": normalize_job_url("https://jobs.80000hours.org/changed"),
         "source_board": "80000_hours", "org": "OrgB", "title": "Role B",
         "content_hash": content_hash("old text")},
        {"id": "alias-target", "canonical_url": "https://jobs.80000hours.org/original-role",
         "source_board": "80000_hours", "org": "OrgC", "title": "Role C",
         "content_hash": "irrelevant"},
    ])
    postings = [
        _posting(url="https://jobs.80000hours.org/unchanged", org="OrgA", title="Role A", description="same text"),
        _posting(url="https://jobs.80000hours.org/changed", org="OrgB", title="Role B", description="new text"),
        _posting(url="https://jobs.80000hours.org/repost-role-c", org="OrgC", title="Role C"),
        _posting(url="https://jobs.80000hours.org/brand-new", org="OrgD", title="Role D"),
    ]

    counts = ingest(postings, client=fake_client, dry_run=True)

    assert counts == IngestCounts(ingested=4, new=1, updated=1, unchanged=1, aliased=1)
    assert len(fake_client.table("jobs").rows) == 3  # still just the 3 seeded rows — no writes happened
    assert fake_client.table("job_aliases").rows == []


# ---------------------------------------------------------------------------
# Reopening a closed job (see src/closure.py)
# ---------------------------------------------------------------------------

def test_ingest_reopens_closed_job_with_enrichment_to_enriched(fake_client):
    p = _posting()
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "closed",
        "enrichment": {"summary": "still good"}, "reenrich_requested_at": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, reopened=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    # Content unchanged from when it closed — nothing new to re-enrich.
    assert row.get("reenrich_requested_at") is None


def test_ingest_reopens_closed_job_with_enrichment_and_changed_content_flags_reenrich(fake_client):
    p = _posting(description="Updated description since closing.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Old description."),
        "raw_description": "Old description.", "status": "closed",
        "enrichment": {"summary": "stale"}, "reenrich_requested_at": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, reopened=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "enriched"
    assert row["reenrich_requested_at"] is not None


def test_ingest_reopen_preserves_original_source_board_when_not_restricted(fake_client):
    """Regression for a real incident: AAC rediscovering a closed
    probably_good job relabeled it 'aac' outright. Neither board is
    restricted, so the ORIGINAL owner must be preserved — same precedence
    as the cross-board touch branch for a still-open job."""
    p = _posting(source_board="aac")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "probably_good",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "closed",
        "enrichment": {"summary": "still good"}, "reenrich_requested_at": None,
    }])

    ingest([p], client=fake_client)

    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "probably_good"


def test_ingest_reopen_incoming_permitted_board_overrides_restricted_original(fake_client):
    """The one case where the incoming board DOES win: the original owner
    was restricted (iap) and the incoming board isn't — same "permitted
    beats restricted" rule as a live cross-board touch, applied on reopen."""
    p = _posting(source_board="aac")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "iap",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "closed",
        "enrichment": {"summary": "still good"}, "reenrich_requested_at": None,
    }])

    ingest([p], client=fake_client)

    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "aac"


def test_ingest_reopen_same_restricted_board_keeps_it(fake_client):
    """Both sides restricted (today's only restricted board reopening its
    own closed job) — nothing to upgrade, original (==incoming) stands."""
    p = _posting(source_board="iap")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "iap",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "closed",
        "enrichment": {"summary": "still good"}, "reenrich_requested_at": None,
    }])

    ingest([p], client=fake_client)

    row = fake_client.table("jobs").rows[0]
    assert row["source_board"] == "iap"


def test_ingest_reopens_closed_job_with_no_enrichment_to_pending_enrich(fake_client):
    """A job closed before it was ever successfully enriched has nothing to
    preserve — reopens straight to pending_enrich, same as a never-enriched
    job's real-content-change branch."""
    p = _posting()
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "closed", "enrichment": None,
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, reopened=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "pending_enrich"


def test_ingest_open_job_unaffected_by_reopen_branch(fake_client):
    """Sanity check: a normal, still-open job must never be counted as
    reopened — the branch is gated strictly on status == 'closed'."""
    p = _posting()
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
        "raw_description": p.description, "status": "enriched", "enrichment": {"summary": "x"},
    }])

    counts = ingest([p], client=fake_client)

    assert counts.reopened == 0
    assert counts.unchanged == 1
