from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
HTML_EXTRACT_TIMEOUT = 90


def _get_response(url: str, *, stream: bool = False) -> requests.Response:
    """Use one bounded retry layer, honoring server cooldowns without bypassing them."""
    for attempt in range(3):
        # Also pace requests made in separate extraction worker processes.
        sleep(3)
        try:
            response = requests.get(
                url, stream=stream, timeout=DOWNLOAD_TIMEOUT,
                headers={"User-Agent": "zotero-arxiv-daily/1.0 (daily research digest)"},
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
            sleep(30 * (attempt + 1))
            continue
        if response.status_code not in {429, 500, 502, 503, 504}:
            try:
                response.raise_for_status()
            except requests.HTTPError:
                response.close()
                raise
            return response

        retry_after = response.headers.get("Retry-After")
        wait = 30 * (attempt + 1)
        if retry_after:
            try:
                wait = max(wait, float(retry_after))
            except ValueError:
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    wait = max(wait, (retry_date - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        logger.warning(
            f"arXiv HTTP {response.status_code}: {url}; "
            f"attempt {attempt + 1}/3, Retry-After={retry_after!r}"
        )
        if attempt == 2 or wait > 60:
            # Stop instead of retrying earlier than a long server cooldown.
            try:
                response.raise_for_status()
            finally:
                response.close()
        response.close()
        sleep(wait)
    raise RuntimeError("arXiv request attempts exhausted")


def _download_file(url: str, path: str) -> None:
    with _get_response(url, stream=True) as response:
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

    with _get_response(html_url) as response:
        downloaded = response.content
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


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[feedparser.FeedParserDict]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        with _get_response(f"https://rss.arxiv.org/atom/{query}") as response:
            feed = feedparser.parse(response.content)
        if feed.get("bozo") or feed.get("version") != "atom10":
            raise ValueError("arXiv returned an invalid Atom feed; refusing to treat it as no papers")
        if 'Feed error for query' in feed.feed.get("title", ""):
            raise ValueError(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        seen = set()
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed_announce_types:
                continue
            paper_id = entry.get("id", "").removeprefix("oai:arXiv.org:")
            if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?", paper_id):
                raise ValueError(f"Invalid arXiv ID in Atom feed: {paper_id!r}")
            canonical_id = re.sub(r"v\d+$", "", paper_id)
            if canonical_id not in seen:
                seen.add(canonical_id)
                raw_papers.append(entry)
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        logger.info(f"Retrieved metadata for {len(raw_papers)} arXiv papers directly from Atom")
        return raw_papers

    def retrieve_papers(self) -> list[Paper]:
        # Metadata conversion is local and requires no per-paper network delay.
        return [self.convert_to_paper(entry) for entry in self._retrieve_raw_papers()]

    def convert_to_paper(self, raw_paper: feedparser.FeedParserDict) -> Paper:
        paper_id = raw_paper.id.removeprefix("oai:arXiv.org:")
        abstract = re.sub(
            r"^\s*arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*",
            "", raw_paper.get("summary", ""), count=1,
        ).strip()
        title = raw_paper.get("title", "").strip()
        if not title or not abstract:
            raise ValueError(f"Missing title or abstract in arXiv feed for {paper_id}")
        # arXiv Atom encodes dc:creator as a comma-separated list. Feedparser
        # exposes it as author (and as one authors entry containing the list).
        authors = [name.strip() for name in raw_paper.get("author", "").split(",") if name.strip()]
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        )

    def enrich_paper(self, paper: Paper) -> None:
        if paper.full_text:
            return
        for extractor in (extract_text_from_tar, extract_text_from_html, extract_text_from_pdf):
            try:
                paper.full_text = extractor(paper)
            except Exception as exc:
                logger.warning(f"Full text extraction failed for {paper.title}: {exc}")
            if paper.full_text:
                return
        logger.warning(f"Using abstract only for {paper.title}; full text unavailable")


def extract_text_from_html(paper: Paper) -> str | None:
    return _run_with_hard_timeout(
        _extract_text_from_html_worker,
        (paper.url.replace("/abs/", "/html/"),),
        timeout=HTML_EXTRACT_TIMEOUT,
        operation="HTML extraction",
        paper_title=paper.title,
    )


def extract_text_from_pdf(paper: Paper) -> str | None:
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


def extract_text_from_tar(paper: Paper) -> str | None:
    source_url = paper.url.replace("/abs/", "/src/")
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.url, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
