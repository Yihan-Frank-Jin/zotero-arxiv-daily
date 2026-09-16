"""Retriever-specific fixtures."""

import feedparser
import pytest


@pytest.fixture()
def mock_feedparser(monkeypatch):
    """Serve a real Atom fixture; reject every non-RSS HTTP request."""
    from pathlib import Path
    from types import SimpleNamespace
    from contextlib import nullcontext
    import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever

    payload = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    parsed = feedparser.parse(payload)
    raw_parse = feedparser.parse

    def get(url, **kwargs):
        assert url.startswith("https://rss.arxiv.org/atom/"), url
        return nullcontext(SimpleNamespace(content=payload))

    monkeypatch.setattr(arxiv_retriever, "_get_response", get)
    monkeypatch.setattr(feedparser, "parse", lambda data: parsed if data == payload else raw_parse(data))
    return parsed


@pytest.fixture()
def mock_biorxiv_api(monkeypatch):
    """Patch requests.get to return the canned bioRxiv API response."""
    import requests
    from types import SimpleNamespace

    from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE

    original_get = requests.get

    def _patched(url, **kwargs):
        if "api.biorxiv.org" in url:
            resp = SimpleNamespace()
            resp.status_code = 200
            resp.json = lambda: SAMPLE_BIORXIV_API_RESPONSE
            resp.raise_for_status = lambda: None
            return resp
        return original_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", _patched)
