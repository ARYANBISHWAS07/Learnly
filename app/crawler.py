import asyncio
import json
import os
import re
from collections import deque
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
from crawl4ai import BrowserConfig
from app import llm_service, chunk_store

DEFAULT_MAX_PAGES = 10000
CRAWL_MAX_PAGES_LIMIT = 5
ROBOTS_TIMEOUT_SECONDS = 8


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


CRAWL_RUN_TIMEOUT_SECONDS = _int_env("CRAWL_RUN_TIMEOUT_SECONDS", 300)
CRAWL_PAGE_TIMEOUT_MS = _int_env("CRAWL_PAGE_TIMEOUT_MS", 45000)
CRAWL_LOG_MAX_ENTRIES = _int_env("CRAWL_LOG_MAX_ENTRIES", 5000)
CRAWLER_STARTUP_TIMEOUT_SECONDS = _int_env("CRAWLER_STARTUP_TIMEOUT_SECONDS", 30)
CRAWLER_SHUTDOWN_TIMEOUT_SECONDS = _int_env("CRAWLER_SHUTDOWN_TIMEOUT_SECONDS", 10)
CRAWL_ENGINE = os.getenv("CRAWL_ENGINE", "http").strip().lower()

_HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTI_SPACES_RE = re.compile(r"[ \t]+")


class _HTMLToTextParser(HTMLParser):
    """Lightweight HTML to text parser without external dependencies."""

    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Union[str, None]]]) -> None:
        lower_tag = tag.lower()
        if lower_tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if lower_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lower_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


class _LinkExtractor(HTMLParser):
    """Extract link href attributes from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Union[str, None]]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {name.lower(): value for name, value in attrs if name}
        href = attr_map.get("href")
        if href:
            self.links.append(href)


def _ensure_absolute_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        raise ValueError("URL is required")

    parsed = urlparse(candidate)
    if not parsed.scheme:
        candidate = f"https://{candidate}"
        parsed = urlparse(candidate)

    if not parsed.netloc and parsed.path:
        candidate = f"https://{parsed.path}"

    return candidate


def _allowed_domains_from_url(parsed_url) -> List[str]:
    netloc = (parsed_url.netloc or "").lower().strip()
    host = (parsed_url.hostname or "").lower().strip()
    if not netloc and not host:
        return []

    domains = set()
    if netloc:
        domains.add(netloc)
    if host:
        domains.add(host)
    if host.startswith("www."):
        domains.add(host[4:])
    return sorted(domains)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: List[str] = []

    for line in text.split("\n"):
        normalized_line = _MULTI_SPACES_RE.sub(" ", line).strip()
        cleaned_lines.append(normalized_line)

    normalized = "\n".join(cleaned_lines)
    normalized = _MULTI_BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def _looks_like_html(content: str) -> bool:
    return bool(_HTML_TAG_RE.search(content[:2500]))


def _html_to_text(content: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(content)
    parser.close()
    return _normalize_text(unescape(parser.get_text()))


def _extract_title_from_html(content: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", content or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _normalize_text(unescape(match.group(1)))


def _is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    host = (parsed.hostname or "").lower()
    for allowed in allowed_domains:
        allowed_value = allowed.lower()
        if netloc == allowed_value or host == allowed_value:
            return True
        if host.endswith(f".{allowed_value}"):
            return True
    return False


def _extract_links_from_html(base_url: str, html: str, allowed_domains: List[str]) -> List[str]:
    extractor = _LinkExtractor()
    extractor.feed(html or "")
    extractor.close()

    links: List[str] = []
    seen = set()
    for raw_href in extractor.links:
        href = raw_href.strip()
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""))
        if not _is_allowed_domain(normalized, allowed_domains):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


def _markdown_to_text(content: str) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown fences while preserving block content.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n", "", text)
    text = text.replace("```", "")

    text = _MARKDOWN_IMAGE_RE.sub(lambda m: m.group(1), text)
    text = _MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Strip common markdown decorations.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~]", "", text)

    return _normalize_text(unescape(text))


def _flatten_json_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (_flatten_json_content(item) for item in value) if part)
    if isinstance(value, dict):
        return "\n".join(part for part in (_flatten_json_content(item) for item in value.values()) if part)
    return str(value)


def _coerce_to_plain_text(raw_content: Any) -> str:
    if raw_content is None:
        return ""

    content = str(raw_content)
    if not content.strip():
        return ""

    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed_json = json.loads(stripped)
            json_text = _normalize_text(_flatten_json_content(parsed_json))
            if json_text:
                return json_text
        except (json.JSONDecodeError, TypeError):
            pass

    if _looks_like_html(content):
        return _html_to_text(content)
    return _markdown_to_text(content)


def _extract_text_from_crawl_result(page: Any) -> str:
    candidates: List[Any] = []

    if getattr(page, "extracted_content", None):
        candidates.append(page.extracted_content)

    markdown = getattr(page, "markdown", None)
    if markdown:
        if hasattr(markdown, "fit_markdown") and markdown.fit_markdown:
            candidates.append(markdown.fit_markdown)
        if hasattr(markdown, "raw_markdown") and markdown.raw_markdown:
            candidates.append(markdown.raw_markdown)
        candidates.append(str(markdown))

    if getattr(page, "cleaned_html", None):
        candidates.append(page.cleaned_html)
    if getattr(page, "html", None):
        candidates.append(page.html)

    best_text = ""
    for candidate in candidates:
        text = _coerce_to_plain_text(candidate)
        if not text:
            continue
        if len(text) > len(best_text):
            best_text = text
        if len(text) >= 200:
            return text

    return best_text


def _build_robots_url(parsed_url) -> str:
    scheme = parsed_url.scheme or "https"
    return urlunparse((scheme, parsed_url.netloc, "/robots.txt", "", "", ""))


def _fetch_text_url(url: str, user_agent: str, timeout: int = ROBOTS_TIMEOUT_SECONDS) -> tuple[Optional[str], Optional[int], Optional[str]]:
    request = Request(
        url=url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/plain,text/*,*/*;q=0.1",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", response.getcode())
            charset = response.headers.get_content_charset() or "utf-8"
            text = response.read().decode(charset, errors="replace")
            return text, status_code, None
    except HTTPError as exc:
        body = None
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = None
        return body, exc.code, str(exc)
    except URLError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)


async def _normalize_crawl_results(crawl_results: Any) -> List[Any]:
    if crawl_results is None:
        return []
    if isinstance(crawl_results, list):
        return crawl_results
    if hasattr(crawl_results, "__aiter__"):
        items: List[Any] = []
        async for result in crawl_results:
            items.append(result)
        return items
    if hasattr(crawl_results, "__iter__") and not isinstance(crawl_results, (str, bytes, dict)):
        return list(crawl_results)
    return [crawl_results]


def _run_crawl_sync_internal(
    start_url: str,
    max_pages: int = 0,
    max_depth: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    include_report: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    normalized_start_url = _ensure_absolute_url(start_url)
    parsed_url = urlparse(normalized_start_url)
    domain = parsed_url.hostname or parsed_url.netloc
    if not domain:
        raise ValueError(f"Unable to determine domain from URL: {start_url}")

    allowed_domains = _allowed_domains_from_url(parsed_url)
    if not allowed_domains:
        raise ValueError(f"Unable to build allowed domains from URL: {start_url}")

    browser_config = BrowserConfig(headless=True, verbose=False)
    effective_max_pages = max_pages if max_pages > 0 else DEFAULT_MAX_PAGES
    effective_max_pages = min(effective_max_pages, CRAWL_MAX_PAGES_LIMIT)
    page_timeout_seconds = max(8, CRAWL_PAGE_TIMEOUT_MS // 1000)

    results: List[Dict[str, Any]] = []
    seen_urls = set()
    robots_url = _build_robots_url(parsed_url)
    robots_parser = RobotFileParser()
    robots_collected = False
    crawled_routes: List[str] = []
    not_crawled_routes: List[Dict[str, str]] = []
    not_crawled_seen: set[tuple[str, str]] = set()
    crawl_logs: List[Dict[str, str]] = []
    crawl_logs_truncated = False
    blocked_by_robots = 0
    failed_pages = 0

    def add_not_crawled(url: str, reason: str, detail: str = "") -> None:
        if not url:
            return
        key = (url, reason)
        if key in not_crawled_seen:
            return
        payload: Dict[str, str] = {"url": url, "reason": reason}
        if detail:
            payload["detail"] = detail[:500]
        not_crawled_routes.append(payload)
        not_crawled_seen.add(key)

    def add_log(level: str, event: str, url: str = "", message: str = "", reason: str = "") -> None:
        nonlocal crawl_logs_truncated
        if len(crawl_logs) >= CRAWL_LOG_MAX_ENTRIES:
            crawl_logs_truncated = True
            return

        entry: Dict[str, str] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
        }
        if url:
            entry["url"] = url
        if reason:
            entry["reason"] = reason
        if message:
            entry["message"] = message[:800]
        crawl_logs.append(entry)

    if progress_callback:
        progress_callback(0, effective_max_pages, f"Fetching robots.txt for {domain}...")
    add_log("info", "robots_fetch_start", robots_url, f"Fetching robots.txt for {domain}")

    robots_text, robots_status, robots_error = _fetch_text_url(
        robots_url,
        browser_config.user_agent,
        ROBOTS_TIMEOUT_SECONDS,
    )

    if robots_status == 200 and robots_text:
        robots_parser.parse(robots_text.splitlines())
        robots_content = _normalize_text(robots_text)
        if robots_content:
            results.append({
                "url": robots_url,
                "title": "robots.txt",
                "content": robots_content,
            })
            seen_urls.add(robots_url)
            robots_collected = True
            crawled_routes.append(robots_url)
            add_log("info", "robots_fetched", robots_url, f"robots.txt fetched successfully (HTTP {robots_status})")
    else:
        if robots_error:
            add_not_crawled(robots_url, "robots_txt_unavailable", robots_error)
            add_log("warning", "robots_fetch_failed", robots_url, robots_error, "robots_txt_unavailable")
        elif robots_status is not None:
            add_not_crawled(robots_url, f"http_{robots_status}", "robots.txt not available")
            add_log("warning", "robots_fetch_failed", robots_url, f"robots.txt returned HTTP {robots_status}", f"http_{robots_status}")
        else:
            add_not_crawled(robots_url, "robots_txt_unavailable", "robots.txt not available")
            add_log("warning", "robots_fetch_failed", robots_url, "robots.txt unavailable", "robots_txt_unavailable")

    if progress_callback:
        if robots_collected:
            progress_callback(0, effective_max_pages, f"robots.txt fetched (HTTP {robots_status}). Starting page crawl...")
        elif robots_error:
            progress_callback(0, effective_max_pages, f"robots.txt unavailable ({robots_error}). Continuing page crawl...")
        else:
            progress_callback(0, effective_max_pages, "robots.txt unavailable. Continuing page crawl...")

    if robots_error and "name resolution" in robots_error.lower():
        failed_pages += 1
        add_not_crawled(normalized_start_url, "dns_resolution_failed", robots_error)
        add_log("error", "crawl_skipped", normalized_start_url, robots_error, "dns_resolution_failed")
    else:
        add_log(
            "info",
            "crawl_start",
            normalized_start_url,
            (
                f"Starting HTTP crawl with max_pages={effective_max_pages}, "
                f"max_depth={max_depth}, timeout={page_timeout_seconds}s"
            ),
        )
        queue = deque([(normalized_start_url, 0)])
        queued_urls = {normalized_start_url}
        visited_or_attempted = set()

        while queue and (len(crawled_routes) - (1 if robots_collected else 0)) < effective_max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited_or_attempted:
                continue
            visited_or_attempted.add(current_url)

            if progress_callback:
                progress_callback(
                    min(len(visited_or_attempted), effective_max_pages),
                    effective_max_pages,
                    f"Crawling URL: {current_url[:80]}",
                )

            try:
                if robots_text and not robots_parser.can_fetch(browser_config.user_agent, current_url):
                    blocked_by_robots += 1
                    add_not_crawled(current_url, "blocked_by_robots_txt", "Blocked by robots.txt rules")
                    add_log("warning", "page_skipped", current_url, "Blocked by robots.txt rules", "blocked_by_robots_txt")
                    continue
            except Exception:
                pass

            html_content, status_code, error_message = _fetch_text_url(
                current_url,
                browser_config.user_agent,
                page_timeout_seconds,
            )
            page_url = current_url

            if error_message:
                failed_pages += 1
                add_not_crawled(page_url, "fetch_error", error_message)
                add_log("warning", "page_failed", page_url, error_message, "fetch_error")
                continue

            if status_code is None:
                failed_pages += 1
                add_not_crawled(page_url, "request_failed", "Request failed without HTTP status")
                add_log("warning", "page_failed", page_url, "Request failed without HTTP status", "request_failed")
                continue

            if status_code >= 400:
                failed_pages += 1
                reason = f"http_{status_code}"
                add_not_crawled(page_url, reason, f"Request returned HTTP {status_code}")
                add_log("warning", "page_failed", page_url, f"Request returned HTTP {status_code}", reason)
                continue

            if page_url in seen_urls:
                add_not_crawled(page_url, "duplicate_url", "Duplicate URL skipped")
                add_log("info", "page_skipped", page_url, "Duplicate URL skipped", "duplicate_url")
                continue

            text_content = _coerce_to_plain_text(html_content or "")
            if not text_content:
                add_not_crawled(page_url, "empty_content", "No text content extracted")
                add_log("warning", "page_skipped", page_url, "No text content extracted", "empty_content")
                continue

            title = _extract_title_from_html(html_content or "")
            if not title:
                parsed_page_url = urlparse(page_url)
                title = parsed_page_url.path.strip("/") or page_url

            seen_urls.add(page_url)
            crawled_routes.append(page_url)
            results.append({
                "url": page_url,
                "title": title,
                "content": text_content,
            })
            add_log("info", "page_crawled", page_url, f"Crawled successfully ({len(text_content)} chars)")
            print(f"Crawled: {page_url} ({len(text_content)} chars)")

            if depth >= max_depth:
                continue

            for discovered_url in _extract_links_from_html(page_url, html_content or "", allowed_domains):
                if discovered_url in queued_urls or discovered_url in seen_urls:
                    continue
                queued_urls.add(discovered_url)
                queue.append((discovered_url, depth + 1))

    if progress_callback:
        pages_without_robots = len(results) - (1 if robots_collected else 0)
        summary_message = f"Crawl complete! Collected {pages_without_robots} pages"
        if robots_collected:
            summary_message += " + robots.txt"
        if blocked_by_robots:
            summary_message += f". Skipped {blocked_by_robots} URL(s) blocked by robots.txt"
        elif failed_pages:
            summary_message += f". {failed_pages} crawl result(s) failed"
        if not_crawled_routes:
            summary_message += f". Not crawled routes: {len(not_crawled_routes)}"
        progress_callback(len(results), len(results), summary_message + ".")

    crawl_summary = {
        "start_url": normalized_start_url,
        "domain": domain,
        "pages_collected_total": len(results),
        "pages_collected_without_robots": len(results) - (1 if robots_collected else 0),
        "robots_txt_collected": robots_collected,
        "routes_crawled_count": len(crawled_routes),
        "routes_not_crawled_count": len(not_crawled_routes),
        "failed_routes_count": failed_pages,
        "blocked_by_robots_count": blocked_by_robots,
        "crawl_logs_count": len(crawl_logs),
        "crawl_logs_truncated": crawl_logs_truncated,
        "crawl_timeout_seconds": CRAWL_RUN_TIMEOUT_SECONDS,
    }

    print(f"Total pages collected: {len(results)}")
    if include_report:
        return {
            "pages": results,
            "crawled_routes": crawled_routes,
            "not_crawled_routes": not_crawled_routes,
            "crawl_logs": crawl_logs,
            "summary": crawl_summary,
        }
    return results


async def run_crawl(
    start_url: str,
    max_pages: int = 0,
    max_depth: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    include_report: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Crawl a website using Crawl4AI and return plain text content.
    Includes robots.txt as part of the collected output and enforces robots rules.

    Args:
        start_url: The starting URL to crawl
        max_pages: Maximum number of pages to crawl (0 = unlimited)
        max_depth: Maximum depth of crawling from start URL
        progress_callback: Optional callback for progress (current, total, message)

    Returns:
        By default, a list of dictionaries containing url, title, and text content.
        If include_report=True, returns a dictionary with pages and route-level crawl report.
    """
    return await asyncio.to_thread(
        _run_crawl_sync_internal,
        start_url,
        max_pages,
        max_depth,
        progress_callback=progress_callback,
        include_report=include_report,
    )


def crawl_sync(
    url: str,
    max_pages: int = 0,
    max_depth: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    include_report: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Synchronous wrapper for run_crawl with optional progress callback."""
    return _run_crawl_sync_internal(
        url,
        max_pages,
        max_depth,
        progress_callback=progress_callback,
        include_report=include_report,
    )
#to handle the creation of the questionnaire and this is the place where vector_store will meet llm_service CURRENTLY SYNCHRONOUS BUT CAN BE MADE ASYNCHRONOUS IF NEEDED IN THE FUTURE
def handle_chunk(chunk: str, metadata: Dict[str, Any], collection_id: str):
    """
    Handle a chunk of text and its metadata by generating a questionnaire.
    """
    # A failed questionnaire (LLM error, rate limit, DB error) must not abort the whole scrape job.
    try:
        questionnaire = llm_service.generate_questionnaire_from_chunk(chunk)
        chunk_store.save_chunk_questionnaire(
            collection_id=collection_id,
            questionnaire=questionnaire,
            metadata=metadata,
            chunk_text=chunk,
        )
    except Exception as exc:
        print(f"Questionnaire generation failed for {metadata.get('url', '')}#{metadata.get('chunk_index')}: {exc}")
