"""Production-ready financial news scraping utilities.

The module exposes a single public entry point:

    get_news(ticker: str) -> List[Dict]

Each returned item follows this schema:

    {
        "text": "one independent news item",
        "date": "YYYY-MM-DD",
        "source": "ET | Twitter | API",
        "ticker": "TCS"
    }

The implementation queries NewsAPI, Economic Times, and Twitter/X in
priority order. All successful results are combined, cleaned, filtered to
the last 7 days, and deduplicated.
"""

from __future__ import annotations

import html
import importlib
import logging
import os
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, TypedDict

import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning

from config import NEWS_API_KEY, get_active_services

try:
    from backend.preprocessing.ticker_mapping import normalize_query
except Exception:  # pragma: no cover - optional import in runtime-constrained envs
    normalize_query = None

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 7
MIN_TEXT_LENGTH = 20
MAX_NEWSAPI_RESULTS = 100
MAX_ET_RESULTS = 50
MAX_TWITTER_RESULTS = 50
MAX_GOOGLE_RESULTS = 50
MAX_REDDIT_RESULTS = 50
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
ECONOMIC_TIMES_TOPIC_URL = "https://economictimes.indiatimes.com/topic/{ticker}"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
REDDIT_SEARCH_URL = "https://www.reddit.com/r/IndianStockMarket/new.json"
REQUEST_TIMEOUT = 15


class NewsItem(TypedDict):
    text: str
    date: str
    source: str
    ticker: str


@dataclass(frozen=True)
class ScrapeContext:
    ticker: str
    lookback_days: int = LOOKBACK_DAYS

    @property
    def normalized_ticker(self) -> str:
        return self.ticker.upper().strip()

    @property
    def cutoff_date(self) -> date:
        return datetime.now(timezone.utc).date() - timedelta(days=self.lookback_days)


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
    )
    return session


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""

    text = html.unescape(str(value))
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_date(value: object) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, date):
        return value
    else:
        raw = _clean_text(str(value))
        if not raw:
            return None

        candidate = raw.replace("Z", "+00:00")
        try:
            dt_value = datetime.fromisoformat(candidate)
        except ValueError:
            dt_value = None

        if dt_value is None:
            known_formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d %b %Y",
                "%d %B %Y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%a, %d %b %Y %H:%M:%S %Z",
            ]
            for fmt in known_formats:
                try:
                    dt_value = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue

        if dt_value is None:
            month_pattern = re.search(
                r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
                raw,
            )
            if month_pattern:
                candidate = month_pattern.group(1)
                for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
                    try:
                        dt_value = datetime.strptime(candidate, fmt)
                        break
                    except ValueError:
                        continue

        if dt_value is None:
            return None

    if dt_value.tzinfo is not None:
        dt_value = dt_value.astimezone(timezone.utc)

    return dt_value.date()


def _format_date(value: object) -> Optional[str]:
    parsed = _parse_date(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _is_recent(value: str, cutoff: date) -> bool:
    parsed = _parse_date(value)
    if parsed is None:
        return False
    return cutoff <= parsed <= datetime.now(timezone.utc).date()


def _build_item(text: str, date_value: str, source: str, ticker: str) -> NewsItem:
    return {
        "text": text,
        "date": date_value,
        "source": source,
        "ticker": ticker,
    }


def _normalize_for_dedup(text: str) -> str:
    normalized = _clean_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s$%.-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _deduplicate_items(items: List[NewsItem]) -> List[NewsItem]:
    unique_items: List[NewsItem] = []
    seen_norms: List[str] = []

    for item in items:
        normalized = _normalize_for_dedup(item["text"])
        if not normalized:
            continue

        if any(
            normalized == seen
            or SequenceMatcher(None, normalized, seen).ratio() >= 0.92
            for seen in seen_norms
        ):
            continue

        seen_norms.append(normalized)
        unique_items.append(item)

    return unique_items


def _finalize_items(items: List[NewsItem], cutoff: date) -> List[NewsItem]:
    cleaned: List[NewsItem] = []

    for item in items:
        text = _clean_text(item.get("text", ""))
        if len(text) < MIN_TEXT_LENGTH:
            continue

        date_value = _format_date(item.get("date"))
        if date_value is None:
            continue

        if not _is_recent(date_value, cutoff):
            continue

        cleaned.append(
            {
                "text": text,
                "date": date_value,
                "source": item["source"],
                "ticker": item["ticker"].upper().strip(),
            }
        )

    cleaned = _deduplicate_items(cleaned)
    cleaned.sort(key=lambda row: (row["date"], _source_rank(row["source"])), reverse=True)
    return cleaned


def _source_rank(source: str) -> int:
    ranking = {"API": 5, "Google": 4, "ET": 3, "Reddit": 2, "Twitter": 1}
    return ranking.get(source, 0)


def _ticker_base(ticker: str) -> str:
    return str(ticker).split(".")[0].upper().strip()


def _resolve_query_terms(ticker: str) -> Dict[str, str]:
    normalized_ticker = str(ticker or "").upper().strip()
    base_ticker = _ticker_base(normalized_ticker)
    company_name = base_ticker
    if normalize_query is not None:
        try:
            resolved = _clean_text(normalize_query(normalized_ticker))
            if resolved:
                company_name = resolved
        except Exception:
            company_name = base_ticker

    return {
        "ticker": normalized_ticker,
        "base_ticker": base_ticker,
        "company_name": company_name,
    }


def fetch_news_api(ticker: str) -> List[Dict]:
    """Fetch financial news from NewsAPI.

    Returns an empty list when NEWS_API_KEY is missing, the API fails,
    or no recent articles are found.
    """

    query_terms = _resolve_query_terms(ticker)
    api_key = os.getenv("NEWS_API_KEY") or NEWS_API_KEY
    if not api_key:
        logger.warning("NEWS_API_KEY not set; skipping NewsAPI")
        return []

    context = ScrapeContext(ticker=ticker)
    company_name = query_terms["company_name"]
    base_ticker = query_terms["base_ticker"]
    logger.info("Using query '%s' for ticker '%s'", company_name, ticker)
    query = f'"{company_name}" OR "{company_name} stock" OR "{base_ticker} stock" OR "{base_ticker} news"'
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": MAX_NEWSAPI_RESULTS,
        "from": context.cutoff_date.isoformat(),
        "apiKey": api_key,
    }

    session = _create_session()

    try:
        response = session.get(NEWSAPI_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            logger.error("NewsAPI returned unexpected status for %s: %s", ticker, payload.get("status"))
            return []

        articles = payload.get("articles", []) or []
        items: List[NewsItem] = []

        for article in articles:
            title = _clean_text(article.get("title"))
            description = _clean_text(article.get("description"))
            combined_text = _clean_text(" ".join(part for part in [title, description] if part))
            article_date = _format_date(article.get("publishedAt"))

            if len(combined_text) < MIN_TEXT_LENGTH or article_date is None:
                continue

            items.append(_build_item(combined_text, article_date, "API", context.normalized_ticker))

        finalized = _finalize_items(items, context.cutoff_date)
        logger.info("NewsAPI returned %d items for %s", len(finalized), context.normalized_ticker)
        return finalized

    except Exception as exc:
        logger.exception("NewsAPI fetch failed for %s: %s", ticker, exc)
        return []


def _extract_et_candidate_text(container) -> tuple[str, str, Optional[str]]:
    title_text = ""
    snippet_text = ""
    date_text: Optional[str] = None

    title_candidates = []
    title_candidates.extend(container.select("h1 a, h2 a, h3 a, a[title]"))
    title_candidates.extend(container.select("h1, h2, h3"))

    for candidate in title_candidates:
        candidate_text = _clean_text(candidate.get("title") or candidate.get_text(" ", strip=True))
        if len(candidate_text) >= MIN_TEXT_LENGTH:
            title_text = candidate_text
            break

    if not title_text:
        direct_links = container.find_all("a", href=True)
        for candidate in direct_links:
            candidate_text = _clean_text(candidate.get_text(" ", strip=True))
            if len(candidate_text) >= MIN_TEXT_LENGTH:
                title_text = candidate_text
                break

    snippet_candidates = container.select("p, span.summary, span.desc, div.summary, div.desc")
    for candidate in snippet_candidates:
        candidate_text = _clean_text(candidate.get_text(" ", strip=True))
        if len(candidate_text) >= 20:
            snippet_text = candidate_text
            break

    time_candidates = container.select("time, span.date, span.time, div.date, div.time")
    for candidate in time_candidates:
        candidate_text = _clean_text(candidate.get("datetime") or candidate.get_text(" ", strip=True))
        if candidate_text:
            date_text = candidate_text
            break

    if not date_text:
        text_blob = _clean_text(container.get_text(" ", strip=True))
        match = re.search(
            r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})",
            text_blob,
        )
        if match:
            date_text = match.group(1)

    return title_text, snippet_text, date_text


def fetch_et_news(ticker: str) -> List[Dict]:
    """Fetch recent news from Economic Times topic pages."""

    context = ScrapeContext(ticker=ticker)
    query_terms = _resolve_query_terms(ticker)
    logger.info("Using query '%s' for ticker '%s'", query_terms["company_name"], ticker)
    et_slug = re.sub(r"\s+", "-", query_terms["company_name"].lower())
    if not et_slug:
        et_slug = query_terms["base_ticker"].lower()
    url = ECONOMIC_TIMES_TOPIC_URL.format(ticker=et_slug)
    session = _create_session()

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        candidate_selectors = [
            "article",
            "div.eachStory",
            "div.eachstory",
            "li",
        ]

        items: List[NewsItem] = []
        seen_blocks = []

        for selector in candidate_selectors:
            for container in soup.select(selector):
                signature = _clean_text(container.get_text(" ", strip=True))
                if not signature or signature in seen_blocks:
                    continue
                seen_blocks.append(signature)

                title_text, snippet_text, date_text = _extract_et_candidate_text(container)
                combined_text = _clean_text(" ".join(part for part in [title_text, snippet_text] if part))
                article_date = _format_date(date_text)

                if len(combined_text) < MIN_TEXT_LENGTH or article_date is None:
                    continue

                items.append(_build_item(combined_text, article_date, "ET", context.normalized_ticker))

        finalized = _finalize_items(items[:MAX_ET_RESULTS], context.cutoff_date)
        logger.info("Economic Times returned %d items for %s", len(finalized), context.normalized_ticker)
        return finalized

    except Exception as exc:
        logger.exception("Economic Times fetch failed for %s: %s", ticker, exc)
        return []


def fetch_twitter_news(ticker: str) -> List[Dict]:
    """Fetch recent tweets using snscrape.

    snscrape is treated as an optional dependency. If it is unavailable,
    the function logs the issue and returns an empty list.
    """

    context = ScrapeContext(ticker=ticker)
    query_terms = _resolve_query_terms(ticker)
    logger.info("Using query '%s' for ticker '%s'", query_terms["company_name"], ticker)

    try:
        sntwitter = importlib.import_module("snscrape.modules.twitter")
    except Exception as exc:
        logger.warning("snscrape is unavailable; skipping Twitter fallback for %s: %s", ticker, exc)
        return []

    query = (
        f"{query_terms['company_name']} stock OR "
        f"{query_terms['company_name']} news OR "
        f"{query_terms['base_ticker']} stock"
    )

    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        items: List[NewsItem] = []

        for index, tweet in enumerate(scraper.get_items()):
            if index >= MAX_TWITTER_RESULTS:
                break

            tweet_text = _clean_text(getattr(tweet, "content", ""))
            tweet_date = getattr(tweet, "date", None)
            formatted_date = _format_date(tweet_date)

            if len(tweet_text) < MIN_TEXT_LENGTH or formatted_date is None:
                continue

            items.append(_build_item(tweet_text, formatted_date, "Twitter", context.normalized_ticker))

        finalized = _finalize_items(items, context.cutoff_date)
        logger.info("Twitter returned %d items for %s", len(finalized), context.normalized_ticker)
        return finalized

    except Exception as exc:
        logger.exception("Twitter scrape failed for %s: %s", ticker, exc)
        return []


def fetch_google_news(ticker: str) -> List[Dict]:
    """Fetch recent headlines from Google News RSS as an additional source."""

    context = ScrapeContext(ticker=ticker)
    query_terms = _resolve_query_terms(ticker)
    logger.info("Using query '%s' for ticker '%s'", query_terms["company_name"], ticker)
    session = _create_session()

    query = f"{query_terms['company_name']} stock OR {query_terms['base_ticker']} stock"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}

    try:
        response = session.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        try:
            soup = BeautifulSoup(response.content, "xml")
        except Exception:
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(response.content, "html.parser")
        items: List[NewsItem] = []

        for entry in soup.find_all("item")[:MAX_GOOGLE_RESULTS]:
            title = _clean_text(entry.title.get_text(" ", strip=True) if entry.title else "")
            description = _clean_text(entry.description.get_text(" ", strip=True) if entry.description else "")
            text = _clean_text(" ".join(part for part in [title, description] if part))
            published = _format_date(entry.pubDate.get_text(strip=True) if entry.pubDate else None)

            if len(text) < MIN_TEXT_LENGTH or published is None:
                continue

            items.append(_build_item(text, published, "Google", context.normalized_ticker))

        finalized = _finalize_items(items, context.cutoff_date)
        logger.info("Google News returned %d items for %s", len(finalized), context.normalized_ticker)
        return finalized
    except Exception as exc:
        logger.exception("Google News fetch failed for %s: %s", ticker, exc)
        return []


def fetch_reddit_news(ticker: str) -> List[Dict]:
    """Fetch recent subreddit posts as sentiment-bearing text snippets."""

    context = ScrapeContext(ticker=ticker)
    query_terms = _resolve_query_terms(ticker)
    logger.info("Using query '%s' for ticker '%s'", query_terms["company_name"], ticker)
    session = _create_session()

    headers = {"User-Agent": "FinIntelEngine/1.0"}
    params = {"limit": MAX_REDDIT_RESULTS}

    try:
        response = session.get(REDDIT_SEARCH_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        children = (((payload or {}).get("data") or {}).get("children") or [])

        items: List[NewsItem] = []
        name_pattern = query_terms["company_name"].lower()
        ticker_pattern = query_terms["base_ticker"].lower()

        for child in children:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            title = _clean_text(data.get("title", ""))
            body = _clean_text(data.get("selftext", ""))
            text = _clean_text(" ".join(part for part in [title, body] if part))
            if len(text) < MIN_TEXT_LENGTH:
                continue

            text_lower = text.lower()
            if name_pattern not in text_lower and ticker_pattern not in text_lower:
                continue

            created_utc = data.get("created_utc")
            date_value = None
            try:
                date_value = datetime.utcfromtimestamp(float(created_utc)).date().isoformat()
            except Exception:
                date_value = None

            if not date_value:
                continue

            items.append(_build_item(text, date_value, "Reddit", context.normalized_ticker))

        finalized = _finalize_items(items, context.cutoff_date)
        logger.info("Reddit returned %d items for %s", len(finalized), context.normalized_ticker)
        return finalized
    except Exception as exc:
        logger.exception("Reddit fetch failed for %s: %s", ticker, exc)
        return []


def get_news(ticker: str) -> List[Dict]:
    """Return recent financial news items for a ticker.

    Priority order:
    1. NewsAPI when NEWS_API_KEY is available
    2. Economic Times topic page
    3. Twitter/X via snscrape

    Successful sources are combined and deduplicated. The function always
    returns a list, even if no source succeeds.
    """

    context = ScrapeContext(ticker=ticker)

    services = get_active_services()
    logger.info(
        "Service status for %s: news_api=%s",
        context.normalized_ticker,
        services.get("news_api", False),
    )

    sources = [
        ("API", fetch_news_api),
        ("Google", fetch_google_news),
        ("ET", fetch_et_news),
        ("Reddit", fetch_reddit_news),
        ("Twitter", fetch_twitter_news),
    ]

    combined_items: List[NewsItem] = []

    for source_name, fetcher in sources:
        try:
            items = fetcher(context.normalized_ticker)
        except Exception as exc:
            logger.exception("Source %s failed for %s: %s", source_name, context.normalized_ticker, exc)
            items = []

        if items:
            logger.info("Collected %d %s items for %s", len(items), source_name, context.normalized_ticker)
            combined_items.extend(items)
        else:
            logger.info("%s returned no recent items for %s", source_name, context.normalized_ticker)

    if not combined_items:
        return []

    finalized = _finalize_items(combined_items, context.cutoff_date)
    logger.info("Returning %d combined news items for %s", len(finalized), context.normalized_ticker)
    return finalized


__all__ = [
    "NewsItem",
    "fetch_news_api",
    "fetch_google_news",
    "fetch_et_news",
    "fetch_reddit_news",
    "fetch_twitter_news",
    "get_news",
]
