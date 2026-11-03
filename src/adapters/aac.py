"""Animal Advocacy Careers (AAC) adapter for the True North v2 ingest pipeline.

No v1 equivalent — unlike the 80k/ProbablyGood/IAP adapters, which each wrap
an existing v1 src/sources/*.py fetch function unchanged, AAC is a brand-new
source. This adapter owns its own network fetch end to end (WordPress REST,
not Algolia).

Source: WordPress + WP Job Manager, exposing listings via the standard WP
REST API at /wp-json/wp/v2/job-listings (confirmed public, no auth needed,
paginated via per_page/page + X-WP-Total/X-WP-TotalPages headers — see the
Phase 2 recon report). Unauthenticated requests only ever see status=publish
listings, so filled/expired postings are already excluded server-side —
nothing extra to filter here.

Field mapping (full audit in the Phase 2 recon report):
  - org: meta._company_name
  - title: title.rendered, HTML-entity-unescaped (WP encodes e.g. "&#8211;")
  - description: content.rendered, HTML-tag-stripped — the full job posting
    text is already in the listing payload; no per-job detail-page fetch
    needed. "No description found" (AAC's own placeholder for speculative/
    general-interest postings) is treated as empty, not real content.
  - location_field: job_listing_location + job_remote taxonomy term NAMES,
    joined — two separate taxonomies rather than one free-text tag list.
  - url: see _apply_url() — deliberately NOT always meta._company_website.
    Confirmed for real: several current postings' company_website is a bare
    domain, a generic /careers page, or a WordPress search-results page —
    none of which safely identifies ONE specific posting — and Compassion in
    World Farming's careers site posts multiple DISTINCT jobs that share a
    bare apply URL differing only by a hash-routed fragment.

All 6 taxonomies (job_listing_location/job_remote feed location_field; the
other 4 don't feed any JobPosting field but are still fetched and checked
for unknown term ids every run, per the Phase 2 spec — a term AAC adds
site-side should surface loudly, not render as a silent gap) are re-fetched
every run, not cached — WP Job Manager lets site admins add a new term (a
new country, say) at any time.
"""
from __future__ import annotations

import html
import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx

from src.adapters.base import FetchResult, JobPosting, SourceAdapter
from src.job_normalize import normalize_job_url

log = logging.getLogger(__name__)

SOURCE_BOARD = "aac"

_BASE_URL = "https://animaladvocacycareers.org/wp-json/wp/v2"
_PER_PAGE = 100
_TIMEOUT = 30

# Every taxonomy WP Job Manager attaches to a job_listing post (confirmed
# public, paginated identically to the main collection, in the Phase 2
# recon). Only job_listing_location/job_remote feed a JobPosting field today.
_TAXONOMIES = (
    "job_listing_location", "job_remote", "job_function",
    "job_filter", "job_type", "organisation_type",
)
_LOCATION_TAXONOMIES = ("job_listing_location", "job_remote")

# AAC's own literal placeholder for a posting with no real description —
# confirmed for real against 2 current listings (speculative-application,
# volunteer-general-application).
_NO_DESCRIPTION_PLACEHOLDER = "no description found"

# meta._company_website values confirmed, for real, to NOT safely identify
# one specific posting: bare-domain pages (greenheartrecruitment.com/,
# svme.org/), a literal /careers landing page
# (animaladvocacyafrica.org/careers), and a WordPress search-results page
# (altprotein.jobs/?s=...). Falling back to AAC's own `link` (always unique
# per post) avoids trusting a generic company page to distinguish postings
# that have nothing else in common.
_GENERIC_PATHS = frozenset({"", "/", "/careers", "/jobs"})


class AACAdapter(SourceAdapter):
    source_board = SOURCE_BOARD

    def fetch(self) -> FetchResult:
        rows, complete, x_wp_total = _fetch_all_pages()
        taxonomies = _fetch_taxonomies()
        postings = parse_listings(rows, taxonomies)
        return FetchResult(postings=postings, fetched=len(rows), nb_hits=x_wp_total, complete=complete)


def _fetch_all_pages() -> tuple[list[dict], bool, int | None]:
    """GET /job-listings, paginated at _PER_PAGE to X-WP-TotalPages. Returns
    (every raw listing dict across every page fetched, complete, x_wp_total).

    complete=True iff every expected page was fetched successfully AND the
    total row count collected matches X-WP-Total — not just "no exception".
    A page request that fails partway through still returns whatever pages
    already succeeded (never raises, never discards good data already
    fetched), but that partial result is reported complete=False so
    src.closure.py never treats a listing missing from this run as evidence
    the job closed.
    """
    url = f"{_BASE_URL}/job-listings"
    rows: list[dict] = []
    x_wp_total: int | None = None
    total_pages = 1
    page = 1

    with httpx.Client(timeout=_TIMEOUT) as client:
        while page <= total_pages:
            try:
                resp = client.get(url, params={"per_page": _PER_PAGE, "page": page})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.error("AAC: page %d/%d fetch failed, stopping with partial results: %s", page, total_pages, exc)
                return rows, False, x_wp_total

            if page == 1:
                try:
                    x_wp_total = int(resp.headers.get("X-WP-Total", "0"))
                    total_pages = max(1, int(resp.headers.get("X-WP-TotalPages", "1")))
                except ValueError:
                    log.warning("AAC: non-numeric X-WP-Total/X-WP-TotalPages header — assuming single page")

            rows.extend(resp.json())
            page += 1

    complete = x_wp_total is None or len(rows) == x_wp_total
    if not complete:
        log.warning("AAC: fetched %d row(s) across all pages but X-WP-Total reported %s", len(rows), x_wp_total)
    return rows, complete, x_wp_total


def _fetch_taxonomies() -> dict[str, dict[int, str]]:
    """{taxonomy_name: {term_id: term_name}} for every taxonomy in
    _TAXONOMIES. A taxonomy endpoint that fails to fetch (or reports more
    than one page — none do today, see the Phase 2 recon, but term counts
    can grow) logs loudly and contributes a possibly-incomplete map, so any
    job referencing a term id outside it logs as "unknown" via
    parse_listings() rather than silently resolving to nothing. Never
    raises, matching this pipeline's per-source graceful-failure philosophy.
    """
    result: dict[str, dict[int, str]] = {}
    with httpx.Client(timeout=_TIMEOUT) as client:
        for taxonomy in _TAXONOMIES:
            terms: dict[int, str] = {}
            try:
                resp = client.get(f"{_BASE_URL}/{taxonomy}", params={"per_page": 100})
                resp.raise_for_status()
                total_pages = int(resp.headers.get("X-WP-TotalPages", "1") or "1")
                if total_pages > 1:
                    log.warning(
                        "AAC: taxonomy %r reports %d pages at per_page=100 — term map is INCOMPLETE "
                        "(this adapter only fetches page 1)", taxonomy, total_pages,
                    )
                for term in resp.json():
                    terms[term["id"]] = term["name"]
            except (httpx.HTTPError, ValueError) as exc:
                log.error("AAC: failed to fetch taxonomy %r — its term names will log as unknown: %s", taxonomy, exc)
            result[taxonomy] = terms
    return result


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def _resolve_terms(
    term_ids: list[int], terms: dict[int, str], taxonomy: str, unknown_out: dict[str, set[int]],
) -> list[str]:
    names = []
    for term_id in term_ids:
        name = terms.get(term_id)
        if name is None:
            unknown_out.setdefault(taxonomy, set()).add(term_id)
            continue
        names.append(name)
    return names


def _looks_like_generic_apply_url(url: str) -> bool:
    """True for a company_website value that can't safely stand in as a
    canonical_url for ONE specific posting — see module docstring.

    A hash-routed fragment (starts with "/") is checked FIRST and makes this
    always False, regardless of how bare the path itself looks: that's
    exactly the Compassion in World Farming shape
    ("https://ciwf.livevacancies.co.uk/#/job/details/261") — path is just
    "/", but the fragment carries the actual per-posting identity, so it's
    the opposite of generic. normalize_job_url() preserves that fragment
    for the same reason.
    """
    parsed = urlparse(url)
    if parsed.fragment.startswith("/"):
        return False
    path = parsed.path.rstrip("/").lower()
    if path in _GENERIC_PATHS:
        return True
    if "s" in parse_qs(parsed.query):  # a WordPress-style search-results page
        return True
    return False


def _apply_url(job: dict) -> str:
    """The canonical_url CANDIDATE for *job*, before parse_listings()'s
    batch-level different-title-collision check: meta._company_website
    normalized, unless it's empty or _looks_like_generic_apply_url(), in
    which case AAC's own (always-unique-per-post) `link` is used instead.
    """
    meta = job.get("meta") or {}
    company_website = (meta.get("_company_website") or "").strip()
    if not company_website or _looks_like_generic_apply_url(company_website):
        return normalize_job_url(job["link"])
    return normalize_job_url(company_website)


def parse_listings(jobs: list[dict], taxonomies: dict[str, dict[int, str]]) -> list[JobPosting]:
    """Convert raw WP REST job_listing dicts to JobPosting objects.

    Two passes: first compute each job's candidate apply URL (_apply_url()),
    then — batch-wide — any URL shared by postings with DIFFERENT titles is
    untrustworthy for ALL of them (the shared company_website can't actually
    distinguish those postings, so trusting it would wrongly merge distinct
    roles into one `jobs` row): every posting in that group falls back to its
    own AAC `link` instead. A URL shared by postings with the SAME title is
    left alone — that's a legitimate duplicate (the same role posted twice
    under separate AAC listing ids), and src.ingest.py's existing
    canonical_url dedup already collapses it to one row before any upsert,
    same as it would for any other source board (confirmed in the Phase 2
    prep report — no ingest.py change needed for this).

    A listing missing a title, or whose `link`/`meta`/`content` is malformed,
    is skipped and logged rather than crashing the whole fetch. A listing
    whose stripped description is empty (including AAC's own "No description
    found" placeholder) is also skipped — not just description-stripped to
    "" — since there's no real text for Tier 1/Tier 2 to score or for the
    enrichment LLM to classify posting_type from; see
    _NO_DESCRIPTION_PLACEHOLDER. The count skipped for this reason is logged
    once at the end, not per-listing.
    """
    candidates: list[tuple[dict, str, str]] = []
    for job in jobs:
        try:
            title = html.unescape((job.get("title") or {}).get("rendered") or "").strip()
            if not title:
                continue
            url = _apply_url(job)
        except (KeyError, TypeError) as exc:
            log.error("AAC: skipping malformed listing (id=%r): %s", job.get("id"), exc)
            continue
        candidates.append((job, title, url))

    url_titles: dict[str, set[str]] = {}
    for _job, title, url in candidates:
        url_titles.setdefault(url, set()).add(title)

    unknown_terms: dict[str, set[int]] = {}
    postings: list[JobPosting] = []
    skipped_empty_description = 0
    for job, title, url in candidates:
        description = _strip_html((job.get("content") or {}).get("rendered") or "")
        if description.strip().lower() == _NO_DESCRIPTION_PLACEHOLDER:
            description = ""
        if not description:
            # No real text for Tier 1/Tier 2 to score against, or for the
            # enrichment LLM to classify posting_type from — confirmed for
            # real: AAC's own "No description found" placeholder shows up on
            # a handful of speculative/general-interest listings (also now
            # covered by gate_v2's posting_type exclusion, but an empty
            # description is a content-quality problem independent of
            # posting_type, so it's filtered here too, before this listing
            # ever becomes a JobPosting).
            skipped_empty_description += 1
            continue

        if len(url_titles[url]) > 1:
            url = normalize_job_url(job["link"])

        meta = job.get("meta") or {}
        org = (meta.get("_company_name") or "").strip()

        location_terms: list[str] = []
        for taxonomy in _LOCATION_TAXONOMIES:
            location_terms += _resolve_terms(
                job.get(taxonomy) or [], taxonomies.get(taxonomy, {}), taxonomy, unknown_terms,
            )
        location_field = ", ".join(location_terms) or None

        # Not used by any JobPosting field — resolved anyway purely so an
        # unknown term id in one of these surfaces below, per the Phase 2
        # spec's explicit instruction to catch taxonomy drift loudly.
        for taxonomy in ("job_function", "job_filter", "job_type", "organisation_type"):
            _resolve_terms(job.get(taxonomy) or [], taxonomies.get(taxonomy, {}), taxonomy, unknown_terms)

        postings.append(JobPosting(
            url=url, org=org, title=title, description=description,
            location_field=location_field, source_board=SOURCE_BOARD,
        ))

    if unknown_terms:
        log.warning(
            "AAC: unknown taxonomy term id(s) encountered (site added a term this adapter doesn't know about "
            "yet — location_field or validation may be incomplete for affected listings): %s",
            {tax: sorted(ids) for tax, ids in unknown_terms.items()},
        )
    if skipped_empty_description:
        log.warning(
            "AAC: skipped %d listing(s) with an empty description (including AAC's own "
            "'No description found' placeholder) — no real text to score or classify",
            skipped_empty_description,
        )

    return postings
