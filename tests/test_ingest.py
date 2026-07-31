"""Tests for src/ingest.py: dedup/content-change detection and batched writes
against `jobs`/`job_aliases`/`run_metrics`."""
from __future__ import annotations

from src.adapters.base import JobPosting
from src.ingest import IngestCounts, ingest
from src.job_normalize import content_hash, normalize_job_url


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


def test_ingest_unchanged_upsert_payload_has_uniform_minimal_keys(fake_client):
    """The NULL-fill regression this design exists to avoid: the unchanged-path
    upsert payload must never reference org/title/status/enrichment/embedding/
    enriched_at, so PostgREST's generated UPDATE clause can't touch any of them.
    canonical_url/source_board/content_hash ARE required even though they're not
    "changing" — Postgres validates NOT NULL on an upsert's INSERT branch
    regardless of which branch the ON CONFLICT actually takes, so a NOT-NULL,
    no-default column omitted here fails even a same-content, existing-row update
    (confirmed the hard way against production)."""
    p = _posting()
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": normalize_job_url(p.url), "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash(p.description),
    }])

    ingest([p], client=fake_client)

    [payload] = fake_client.table("jobs").upsert_calls
    assert set(payload[0].keys()) == {"canonical_url", "source_board", "content_hash", "last_seen"}


# ---------------------------------------------------------------------------
# Same URL, content changed — re-enrich
# ---------------------------------------------------------------------------

def test_ingest_changed_content_updates_and_resets_status(fake_client):
    p = _posting(description="Updated description text.")
    canonical = normalize_job_url(p.url)
    fake_client.seed("jobs", [{
        "id": "job-1", "canonical_url": canonical, "source_board": "80000_hours",
        "org": p.org, "title": p.title, "content_hash": content_hash("Old description text."),
        "raw_description": "Old description text.", "status": "enriched",
        "enrichment": {"summary": "stale enrichment"},
    }])

    counts = ingest([p], client=fake_client)

    assert counts == IngestCounts(ingested=1, updated=1)
    row = fake_client.table("jobs").rows[0]
    assert row["status"] == "pending_enrich"
    assert row["raw_description"] == "Updated description text."
    assert row["content_hash"] == content_hash("Updated description text.")
    # enrichment itself isn't cleared by ingest — the enrich worker overwrites it
    # once it re-processes this job; the upsert payload never references that key.
    assert row["enrichment"] == {"summary": "stale enrichment"}


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
