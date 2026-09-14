from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _result_from_rss(entry: feedparser.FeedParserDict) -> ArxivResult:
    """Adapt an arXiv Atom announcement without another metadata API request.

    Atom's dates describe announcements, not original submission dates, so do
    not assign them to Result.published/updated. This pipeline does not use them.
    See https://info.arxiv.org/help/atom_specifications.html.
    """
    paper_id = entry.get("id", "").removeprefix("oai:arXiv.org:")
    if not re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-zA-Z.-]+/[0-9]{7})(?:v[0-9]+)?", paper_id):
        raise ValueError(f"Invalid arXiv RSS paper ID: {paper_id!r}")
    title = entry.get("title", "").strip()
    # The Atom summary is plain text: preserve mathematics and literal '<'.
    summary = re.sub(
        r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*",
        "", entry.get("summary", "").strip(), count=1,
    ).strip()
    # feedparser maps dc:creator to author, with a comma-separated author list.
    authors = [
        ArxivResult.Author(name.strip())
        for name in re.split(r",\s*(?![^()]*\))", entry.get("author", ""))
        if name.strip()
    ]
    if not title or not summary or not authors:
        raise ValueError(f"Incomplete RSS metadata for {paper_id}; cannot safely recover API failure.")
    categories = [tag["term"] for tag in entry.get("tags", []) if tag.get("term")]
    entry_url = f"https://arxiv.org/abs/{paper_id}"
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    return ArxivResult(
        entry_id=entry_url,
        title=title,
        authors=authors,
        summary=summary,
        categories=categories,
        primary_category=categories[0] if categories else "",
        links=[
            ArxivResult.Link(entry_url, rel="alternate", content_type="text/html"),
            ArxivResult.Link(pdf_url, title="pdf", rel="related", content_type="application/pdf"),
        ],
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Keep retries bounded: RSS already contains the metadata needed below.
        client = arxiv.Client(num_retries=2, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if feed.get("status", 200) >= 400 or feed.get("bozo") or not feed.feed.get("title"):
            raise ValueError(f"Failed to read arXiv RSS feed for {query}; refusing an empty fallback.")
        if 'Feed error for query' in feed.feed.title:
            raise ValueError(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        # Keep each paper once when it appears in several subscribed categories.
        entries_by_id = {
            entry.id.removeprefix("oai:arXiv.org:"): entry
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        }
        paper_ids = list(entries_by_id)
        if self.config.executor.debug:
            paper_ids = paper_ids[:10]

        raw_papers = []
        with tqdm(total=len(paper_ids)) as bar:
            for i in range(0, len(paper_ids), 20):
                search = arxiv.Search(id_list=paper_ids[i:i + 20])
                try:
                    batch = list(client.results(search))
                except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError, requests.RequestException) as exc:
                    if isinstance(exc, arxiv.HTTPError) and not (
                        exc.status == 429 or 500 <= exc.status < 600
                    ):
                        raise
                    # Do not keep hitting an unavailable API for later batches.
                    # Use this run's already-fetched feed, never stale cached data.
                    logger.warning(
                        f"arXiv API unavailable ({exc}); using RSS metadata for "
                        f"the remaining {len(paper_ids) - i} papers."
                    )
                    batch = [
                        _result_from_rss(entries_by_id[paper_id])
                        for paper_id in paper_ids[i:]
                    ]
                    raw_papers.extend(batch)
                    bar.update(len(batch))
                    break
                raw_papers.extend(batch)
                bar.update(len(batch))
                if i + 20 < len(paper_ids):
                    sleep(3)
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
