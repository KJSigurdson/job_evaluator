"""Tests for notify.py: send_match_digest must never raise, regardless of failure mode."""
from __future__ import annotations

import httpx
import pytest

from src import notify


_MATCHES = [{"title": "Head of Data", "org": "GiveDirectly", "location": "Remote",
             "fit_score": 0.85, "why_fits": "Strong fit.", "url": "https://example.org/job/1"}]


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)


def _set_config(monkeypatch, *, url="https://example.org/digest", secret="shh"):
    monkeypatch.setenv("MATCH_DIGEST_URL", url)
    monkeypatch.setenv("PIPELINE_SHARED_SECRET", secret)


# ---------------------------------------------------------------------------
# Happy path — correct body + header
# ---------------------------------------------------------------------------

def test_send_match_digest_posts_correct_body_and_header(monkeypatch):
    _set_config(monkeypatch, url="https://example.org/digest", secret="topsecret")
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse(200)

    monkeypatch.setattr(notify.httpx, "post", fake_post)

    notify.send_match_digest("u1", _MATCHES)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://example.org/digest"
    assert call["json"] == {"user_id": "u1", "matches": _MATCHES}
    assert call["headers"] == {"x-pipeline-secret": "topsecret"}


# ---------------------------------------------------------------------------
# Failure modes — never raise
# ---------------------------------------------------------------------------

def test_send_match_digest_swallows_httpx_exception(monkeypatch):
    _set_config(monkeypatch)

    def raising_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(notify.httpx, "post", raising_post)

    notify.send_match_digest("u1", _MATCHES)  # must not raise


def test_send_match_digest_swallows_non_2xx_response(monkeypatch):
    _set_config(monkeypatch)
    monkeypatch.setattr(notify.httpx, "post", lambda *a, **k: _FakeResponse(500))

    notify.send_match_digest("u1", _MATCHES)  # must not raise


def test_send_match_digest_swallows_arbitrary_exception(monkeypatch):
    _set_config(monkeypatch)

    def raising_post(*args, **kwargs):
        raise ValueError("something unexpected")

    monkeypatch.setattr(notify.httpx, "post", raising_post)

    notify.send_match_digest("u1", _MATCHES)  # must not raise


# ---------------------------------------------------------------------------
# Missing config — skip silently, no request attempted
# ---------------------------------------------------------------------------

def test_send_match_digest_skips_when_url_missing(monkeypatch):
    monkeypatch.delenv("MATCH_DIGEST_URL", raising=False)
    monkeypatch.setenv("PIPELINE_SHARED_SECRET", "shh")
    called = []
    monkeypatch.setattr(notify.httpx, "post", lambda *a, **k: called.append(1))

    notify.send_match_digest("u1", _MATCHES)

    assert called == []


def test_send_match_digest_skips_when_secret_missing(monkeypatch):
    monkeypatch.setenv("MATCH_DIGEST_URL", "https://example.org/digest")
    monkeypatch.delenv("PIPELINE_SHARED_SECRET", raising=False)
    called = []
    monkeypatch.setattr(notify.httpx, "post", lambda *a, **k: called.append(1))

    notify.send_match_digest("u1", _MATCHES)

    assert called == []


def test_send_match_digest_skips_when_both_missing(monkeypatch):
    monkeypatch.delenv("MATCH_DIGEST_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SHARED_SECRET", raising=False)
    called = []
    monkeypatch.setattr(notify.httpx, "post", lambda *a, **k: called.append(1))

    notify.send_match_digest("u1", _MATCHES)

    assert called == []
