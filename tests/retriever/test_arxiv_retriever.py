"""Tests for ArxivRetriever."""

import copy
import time
from types import SimpleNamespace

import feedparser
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
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


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


# The daily feed already has enough metadata to survive an unavailable API.


@pytest.mark.parametrize("failure", [
    arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 2, 429),
    arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 2, 503),
    arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 2, 502),
    requests.Timeout("arXiv API timed out"),
])
@pytest.mark.parametrize("include_cross_list", [False, True])
def test_api_failure_uses_rss_metadata(config, mock_feedparser, monkeypatch, failure, include_cross_list):
    calls = []
    config.source.arxiv.include_cross_list = include_cross_list

    def unavailable(self, search):
        calls.append(search.id_list)
        raise failure

    monkeypatch.setattr(arxiv_retriever.arxiv.Client, "results", unavailable)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    allowed = {"new", "cross"} if include_cross_list else {"new"}
    entries = [e for e in mock_feedparser.entries if e.arxiv_announce_type in allowed]
    assert len(calls) == 1
    assert len(papers) == len(entries)
    for paper, entry in zip(papers, entries):
        pid = entry.id.removeprefix("oai:arXiv.org:")
        assert paper.title == entry.title
        assert paper.abstract == entry.summary.split("Abstract: ", 1)[1]
        assert paper.authors == entry.author.split(", ")
        assert paper.url == f"https://arxiv.org/abs/{pid}"
        assert paper.pdf_url == f"https://arxiv.org/pdf/{pid}"


def test_failure_after_successful_batch_retains_all_papers(config, mock_feedparser, monkeypatch):
    entry = next(e for e in mock_feedparser.entries if e.arxiv_announce_type == "new")
    entries = []
    for index in range(45):
        item = copy.deepcopy(entry)
        item["id"] = f"oai:arXiv.org:2609.{index:05d}v1"
        entries.append(item)
    mock_feedparser["entries"] = entries
    calls = []

    def partial_success(self, search):
        calls.append(search.id_list)
        if len(calls) == 1:
            return iter(arxiv_retriever._result_from_rss(e) for e in entries[:20])
        raise arxiv_retriever.arxiv.HTTPError("api", 2, 503)

    monkeypatch.setattr(arxiv_retriever.arxiv.Client, "results", partial_success)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    papers = ArxivRetriever(config)._retrieve_raw_papers()
    assert len(calls) == 2  # The third batch must not hit the failing API.
    assert [p.get_short_id() for p in papers] == [e.id.removeprefix("oai:arXiv.org:") for e in entries]


def test_nontransient_api_error_is_not_hidden(config, mock_feedparser, monkeypatch):
    def bad_request(self, search):
        raise arxiv_retriever.arxiv.HTTPError("api", 0, 400)

    monkeypatch.setattr(arxiv_retriever.arxiv.Client, "results", bad_request)
    with pytest.raises(arxiv_retriever.arxiv.HTTPError):
        ArxivRetriever(config)._retrieve_raw_papers()


@pytest.mark.parametrize("field", ["title", "summary", "author"])
def test_incomplete_rss_is_not_silently_accepted(mock_feedparser, field):
    entry = copy.deepcopy(mock_feedparser.entries[0])
    entry[field] = ""
    with pytest.raises(ValueError, match="Incomplete RSS metadata"):
        arxiv_retriever._result_from_rss(entry)


def test_rss_preserves_math_authors_and_download_urls(mock_feedparser):
    entry = copy.deepcopy(mock_feedparser.entries[0])
    entry["summary"] = "arXiv:2508.13426v1 Announce Type: cross\nAbstract: We find $a < b$ & $c > d$."
    entry["author"] = "Alice (University, Department), Bob"
    paper = arxiv_retriever._result_from_rss(entry)
    assert paper.summary == "We find $a < b$ & $c > d$."
    assert [a.name for a in paper.authors] == ["Alice (University, Department)", "Bob"]
    assert paper.source_url() == "https://arxiv.org/src/2508.13426v1"
    assert paper.pdf_url == "https://arxiv.org/pdf/2508.13426v1"


def test_invalid_feed_fails_instead_of_reporting_no_papers(config, mock_feedparser):
    mock_feedparser["status"] = 503
    mock_feedparser["entries"] = []
    with pytest.raises(ValueError, match="Failed to read arXiv RSS"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_empty_valid_feed_needs_no_api(config, mock_feedparser, monkeypatch):
    mock_feedparser["entries"] = []
    def unexpected_request(self, search):
        pytest.fail("An empty feed must not call the metadata API")
    monkeypatch.setattr(arxiv_retriever.arxiv.Client, "results", unexpected_request)
    assert ArxivRetriever(config)._retrieve_raw_papers() == []


def test_fallback_preserves_debug_limit_and_deduplicates(config, mock_feedparser, monkeypatch):
    entry = next(e for e in mock_feedparser.entries if e.arxiv_announce_type == "new")
    entries = []
    for index in range(15):
        item = copy.deepcopy(entry)
        item["id"] = f"oai:arXiv.org:2609.{index:05d}v1"
        entries.extend([item, copy.deepcopy(item)])
    mock_feedparser["entries"] = entries
    config.executor.debug = True
    def unavailable(self, search):
        raise requests.ConnectionError("API unavailable")
    monkeypatch.setattr(arxiv_retriever.arxiv.Client, "results", unavailable)
    papers = ArxivRetriever(config)._retrieve_raw_papers()
    assert len(papers) == 10
    assert len({p.entry_id for p in papers}) == 10
