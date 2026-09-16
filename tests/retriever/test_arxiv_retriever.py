"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import pytest
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("Full text must not be downloaded before ranking")

    for kind in ("tar", "html", "pdf"):
        monkeypatch.setattr(arxiv_retriever, f"extract_text_from_{kind}", unexpected_download)
    papers = ArxivRetriever(config).retrieve_papers()
    new_entries = [e for e in mock_feedparser.entries if e.arxiv_announce_type == "new"]
    assert len(papers) == len(new_entries)
    assert {p.title for p in papers} == {e.title for e in new_entries}
    first = papers[0]
    assert first.abstract == new_entries[0].summary.split("Abstract: ", 1)[1].strip()
    assert first.authors == [a.strip() for a in new_entries[0].author.split(",")]
    assert first.authors
    assert first.pdf_url == first.url.replace("/abs/", "/pdf/")
    assert all(p.full_text is None for p in papers)


def test_cross_lists_deduplication_and_debug(config, mock_feedparser):
    config.source.arxiv.include_cross_list = True
    mock_feedparser.entries.append(mock_feedparser.entries[0])
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    expected = {e.id for e in mock_feedparser.entries if e.arxiv_announce_type in {"new", "cross"}}
    assert len(papers) == len(expected)
    assert len({p.url for p in papers}) == len(papers)
    config.executor.debug = True
    assert len(retriever.retrieve_papers()) == min(10, len(expected))


def test_invalid_feed_is_not_treated_as_no_papers(config, mock_feedparser):
    mock_feedparser["bozo"] = True
    with pytest.raises(ValueError, match="invalid Atom feed"):
        ArxivRetriever(config).retrieve_papers()


def test_valid_empty_feed(config, mock_feedparser):
    mock_feedparser.entries.clear()
    assert ArxivRetriever(config).retrieve_papers() == []


def test_missing_abstract_is_reported(config, mock_feedparser):
    entry = next(e for e in mock_feedparser.entries if e.arxiv_announce_type == "new")
    entry["summary"] = ""
    with pytest.raises(ValueError, match="Missing title or abstract"):
        ArxivRetriever(config).retrieve_papers()


def test_full_text_failure_keeps_metadata(config, mock_feedparser, monkeypatch):
    paper = ArxivRetriever(config).retrieve_papers()[0]
    original = (paper.title, paper.abstract, paper.authors[:], paper.url)
    calls = []
    def fail(p):
        calls.append(p.url)
        raise TimeoutError("download stalled")
    for kind in ("tar", "html", "pdf"):
        monkeypatch.setattr(arxiv_retriever, f"extract_text_from_{kind}", fail)
    ArxivRetriever(config).enrich_paper(paper)
    assert len(calls) == 3
    assert paper.full_text is None
    assert (paper.title, paper.abstract, paper.authors, paper.url) == original


def test_full_text_fallback_stops_after_success(config, mock_feedparser, monkeypatch):
    paper = ArxivRetriever(config).retrieve_papers()[0]
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda p: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda p: "HTML full text")
    pdf_calls = []
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda p: pdf_calls.append(p))
    ArxivRetriever(config).enrich_paper(paper)
    assert paper.full_text == "HTML full text"
    assert pdf_calls == []


def _response(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response.url = "https://rss.arxiv.org/atom/astro-ph.GA"
    response._content = b""
    response._content_consumed = True
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return response


def test_429_retries_are_bounded_and_honor_retry_after(monkeypatch):
    responses = [_response(429, "45"), _response(429, "60"), _response(429)]
    calls, waits = [], []
    def get(url, **kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]
    monkeypatch.setattr(arxiv_retriever.requests, "get", get)
    monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
    with pytest.raises(requests.HTTPError):
        arxiv_retriever._get_response("https://rss.arxiv.org/atom/astro-ph.GA")
    assert len(calls) == 3
    assert waits == [3, 45, 3, 60, 3]
    assert all(c["timeout"] == arxiv_retriever.DOWNLOAD_TIMEOUT for c in calls)


@pytest.mark.parametrize("retry_after", ["3600", "Wed, 01 Jan 2098 00:00:00 GMT"])
def test_long_server_cooldown_is_not_retried_early(monkeypatch, retry_after):
    waits = []
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        return _response(429, retry_after)
    monkeypatch.setattr(arxiv_retriever.requests, "get", get)
    monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
    with pytest.raises(requests.HTTPError):
        arxiv_retriever._get_response("https://rss.arxiv.org/atom/astro-ph.GA")
    assert calls == [1]
    assert waits == [3]


def test_http_recovery_and_permanent_failure(monkeypatch):
    responses = iter([_response(503), _response(200), _response(404)])
    waits = []
    monkeypatch.setattr(arxiv_retriever.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
    with arxiv_retriever._get_response("https://rss.arxiv.org/atom/astro-ph.GA") as response:
        assert response.status_code == 200
    assert waits == [3, 30, 3]
    with pytest.raises(requests.HTTPError):
        arxiv_retriever._get_response("https://rss.arxiv.org/atom/astro-ph.GA")


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
