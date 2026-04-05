"""Ticker and company name mapping helpers.

This module provides dynamic and static lookup helpers for converting
between company names and Yahoo Finance tickers. It is designed to be
safe for use in scraping and sentiment pipelines:

- `get_company_name(ticker)` returns a company name for a ticker
- `get_ticker(company_name)` returns a yfinance-compatible ticker
- `get_search_query(ticker)` returns a news-search friendly query

Lookups are cached in-memory to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import os
import re
import importlib
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# In-memory caches to avoid duplicate network calls.
ticker_to_name_cache: Dict[str, str] = {}
name_to_ticker_cache: Dict[str, str] = {}

# Static fallback mappings.
STATIC_TICKER_MAP = {
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "RELIANCE": "Reliance Industries",
    "HDFCBANK": "HDFC Bank",
}

STATIC_NAME_TO_TICKER = {
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
}

_YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
_REQUEST_TIMEOUT = 10


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_key(value: str) -> str:
    return _clean_text(value).upper()


def _normalize_company_key(value: str) -> str:
    return _clean_text(value).casefold()


def _strip_yfinance_suffix(ticker: str) -> str:
    cleaned = _clean_text(ticker)
    cleaned = cleaned.replace("^", "")
    cleaned = cleaned.split(".")[0] if cleaned.endswith(('.NS', '.BO', '.AX', '.TO', '.L')) else cleaned
    return cleaned.upper()


def _best_static_company_match(company_name: str) -> Optional[str]:
    target = _normalize_company_key(company_name)
    best_match: Tuple[float, Optional[str]] = (0.0, None)

    for static_name, ticker in STATIC_NAME_TO_TICKER.items():
        similarity = SequenceMatcher(None, target, static_name.casefold()).ratio()
        if similarity > best_match[0]:
            best_match = (similarity, ticker)

    if best_match[0] >= 0.75:
        return best_match[1]
    return None


def _best_static_ticker_match(company_name: str) -> Optional[str]:
    target = _normalize_company_key(company_name)
    best_match: Tuple[float, Optional[str]] = (0.0, None)

    for ticker, static_name in STATIC_TICKER_MAP.items():
        similarity = SequenceMatcher(None, target, static_name.casefold()).ratio()
        if similarity > best_match[0]:
            best_match = (similarity, ticker)

    if best_match[0] >= 0.75:
        return best_match[1]
    return None


def _fetch_yahoo_search(company_name: str) -> List[Dict[str, object]]:
    params = {"q": company_name, "quotesCount": 10, "newsCount": 0}
    response = requests.get(_YAHOO_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    quotes = payload.get("quotes", []) or []
    return [quote for quote in quotes if isinstance(quote, dict)]


def _get_yfinance_module():
    try:
        return importlib.import_module("yfinance")
    except Exception as exc:
        logger.info("yfinance is unavailable; falling back to static mappings: %s", exc)
        return None


def _format_search_ticker(ticker: str) -> str:
    cleaned = _clean_text(ticker).upper()
    if "." in cleaned:
        return cleaned
    if cleaned in STATIC_TICKER_MAP:
        return f"{cleaned}.NS"
    return cleaned


def get_company_name(ticker: str) -> str:
    """Return a company name for a Yahoo Finance ticker.

    The function first tries yfinance, then static mappings, and finally
    falls back to a cleaned version of the ticker.
    """

    cleaned_ticker = _clean_text(ticker)
    if not cleaned_ticker:
        return ""

    cache_key = cleaned_ticker.upper()
    if cache_key in ticker_to_name_cache:
        return ticker_to_name_cache[cache_key]

    lookup_ticker = cleaned_ticker
    base_ticker = _strip_yfinance_suffix(cache_key)
    if cache_key in STATIC_TICKER_MAP:
        lookup_ticker = cache_key
    elif base_ticker in STATIC_TICKER_MAP:
        lookup_ticker = base_ticker

    try:
        yf_module = _get_yfinance_module()
        if yf_module is not None:
            ticker_obj = yf_module.Ticker(lookup_ticker)
            info = getattr(ticker_obj, "info", {}) or {}
            company_name = _clean_text(info.get("longName") or info.get("shortName") or info.get("name"))
            if company_name:
                ticker_to_name_cache[cache_key] = company_name
                logger.info("Resolved company name via yfinance: %s -> %s", cleaned_ticker, company_name)
                return company_name
    except Exception as exc:
        logger.warning("yfinance lookup failed for %s: %s", cleaned_ticker, exc)

    static_company = STATIC_TICKER_MAP.get(cache_key) or STATIC_TICKER_MAP.get(base_ticker)
    if static_company:
        ticker_to_name_cache[cache_key] = static_company
        logger.info("Resolved company name via static mapping: %s -> %s", cleaned_ticker, static_company)
        return static_company

    cleaned_fallback = re.sub(r"\.[A-Z]+$", "", cache_key).replace("^", "")
    logger.info("Falling back to cleaned ticker for company name: %s -> %s", cleaned_ticker, cleaned_fallback)
    ticker_to_name_cache[cache_key] = cleaned_fallback
    return cleaned_fallback


def get_ticker(company_name: str) -> str:
    """Return a yfinance-compatible ticker for a company name.

    The function first tries the Yahoo Finance search endpoint, then a
    static mapping, and finally a fuzzy match over the static mapping.
    """

    cleaned_company = _clean_text(company_name)
    if not cleaned_company:
        return ""

    cache_key = cleaned_company.casefold()
    if cache_key in name_to_ticker_cache:
        return name_to_ticker_cache[cache_key]

    try:
        search_results = _fetch_yahoo_search(cleaned_company)
        best_candidate: Optional[str] = None
        best_score = 0.0

        for result in search_results:
            symbol = _clean_text(result.get("symbol"))
            longname = _clean_text(result.get("longname") or result.get("shortname") or result.get("name"))
            if not symbol:
                continue

            if longname:
                similarity = SequenceMatcher(None, cleaned_company.casefold(), longname.casefold()).ratio()
            else:
                similarity = SequenceMatcher(None, cleaned_company.casefold(), symbol.casefold()).ratio()

            if similarity > best_score:
                best_candidate = symbol
                best_score = similarity

        if best_candidate and best_score >= 0.55:
            formatted = _format_search_ticker(best_candidate)
            name_to_ticker_cache[cache_key] = formatted
            logger.info("Resolved ticker via Yahoo search: %s -> %s", cleaned_company, formatted)
            return formatted
    except Exception as exc:
        logger.warning("Yahoo search lookup failed for %s: %s", cleaned_company, exc)

    static_exact = STATIC_NAME_TO_TICKER.get(cleaned_company)
    if static_exact:
        name_to_ticker_cache[cache_key] = static_exact
        logger.info("Resolved ticker via static mapping: %s -> %s", cleaned_company, static_exact)
        return static_exact

    fuzzy_static = _best_static_company_match(cleaned_company)
    if fuzzy_static:
        name_to_ticker_cache[cache_key] = fuzzy_static
        logger.info("Resolved ticker via fuzzy static mapping: %s -> %s", cleaned_company, fuzzy_static)
        return fuzzy_static

    logger.info("Falling back to cleaned company name for ticker: %s", cleaned_company)
    fallback = _normalize_key(cleaned_company).replace(" ", "")
    name_to_ticker_cache[cache_key] = fallback
    return fallback


def get_search_query(ticker: str) -> str:
    """Return a news search query for a ticker.

    Prefers the mapped company name, otherwise falls back to ticker-based
    search text.
    """

    cleaned_ticker = _clean_text(ticker)
    if not cleaned_ticker:
        return ""

    company_name = get_company_name(cleaned_ticker)
    if company_name and company_name != _strip_yfinance_suffix(cleaned_ticker):
        return f"{company_name} stock"

    return f"{_strip_yfinance_suffix(cleaned_ticker)} stock"


def normalize_query(ticker: str) -> str:
    """Return a human-readable company query term for scraping/search.

    Examples:
        INFY.NS -> Infosys
        TCS.NS -> Tata Consultancy Services
    """

    cleaned_ticker = _clean_text(ticker)
    if not cleaned_ticker:
        return ""

    company_name = get_company_name(cleaned_ticker)
    if company_name:
        return company_name

    return _strip_yfinance_suffix(cleaned_ticker)


__all__ = [
    "STATIC_TICKER_MAP",
    "STATIC_NAME_TO_TICKER",
    "ticker_to_name_cache",
    "name_to_ticker_cache",
    "get_company_name",
    "get_ticker",
    "get_search_query",
    "normalize_query",
]
