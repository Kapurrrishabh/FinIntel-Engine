"""Fundamental financial data scraper.

This module extracts structured annual financial data from Yahoo Finance via
`yfinance` and returns the fields needed by the fundamental and risk analysis
pipeline.

Public entry point:
    get_financial_data(ticker: str) -> Dict[str, float]

The function always returns a complete dictionary with safe float defaults.
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FINANCIAL_DATA: Dict[str, float] = {
    "rev_t": 0.0,
    "rev_prev": 0.0,
    "net_income": 0.0,
    "equity": 0.0,
    "debt": 0.0,
    "current_assets": 0.0,
    "current_liabilities": 0.0,
    "ebit": 0.0,
    "interest_expense": 0.0,
}

STATEMENT_ALIASES = {
    "revenue": ["Total Revenue", "totalRevenue", "Revenue"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
    "equity": [
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ],
    "debt": [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
        "Short Long Term Debt",
    ],
    "current_assets": ["Total Current Assets"],
    "current_liabilities": ["Total Current Liabilities"],
    "ebit": ["EBIT", "Ebit", "Operating Income"],
    "interest_expense": ["Interest Expense", "InterestExpense"],
}


def _get_yfinance_module():
    try:
        return importlib.import_module("yfinance")
    except Exception as exc:
        logger.warning("yfinance is unavailable; returning default financial data: %s", exc)
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _normalize_row_name(name: object) -> str:
    return str(name or "").strip().casefold()


def safe_get(statement_df, key: str) -> float:
    """Safely extract the latest available value for a row name.

    Returns 0.0 if the dataframe is missing, the row does not exist, or the
    value is invalid.
    """

    if statement_df is None or getattr(statement_df, "empty", True):
        logger.debug("Statement dataframe empty while looking up %s", key)
        return 0.0

    normalized_key = _normalize_row_name(key)

    try:
        row_lookup = {_normalize_row_name(index): index for index in statement_df.index}
    except Exception:
        logger.debug("Unable to build row lookup while looking up %s", key)
        return 0.0

    actual_row = row_lookup.get(normalized_key)
    if actual_row is None:
        logger.debug("Missing financial field: %s", key)
        return 0.0

    row = statement_df.loc[actual_row]
    try:
        if hasattr(row, "iloc"):
            series = row.dropna()
            if getattr(series, "empty", False):
                logger.debug("No usable values for financial field: %s", key)
                return 0.0
            return _safe_float(series.iloc[0], default=0.0)
        return _safe_float(row, default=0.0)
    except Exception:
        logger.debug("Failed extracting financial field: %s", key)
        return 0.0


def _get_first_available(statement_df, keys: Sequence[str]) -> float:
    for key in keys:
        value = safe_get(statement_df, key)
        if value != 0.0:
            return value
    return 0.0


def _get_latest_two_values(statement_df, keys: Sequence[str]) -> Tuple[float, float]:
    if statement_df is None or getattr(statement_df, "empty", True):
        return 0.0, 0.0

    normalized_lookup = {_normalize_row_name(index): index for index in statement_df.index}

    for key in keys:
        actual_row = normalized_lookup.get(_normalize_row_name(key))
        if actual_row is None:
            continue

        try:
            row = statement_df.loc[actual_row]
            values = [value for value in row.tolist() if not _is_invalid_number(value)]
            if not values:
                continue

            latest = _safe_float(values[0], default=0.0)
            previous = _safe_float(values[1], default=0.0) if len(values) > 1 else 0.0
            return latest, previous
        except Exception:
            logger.debug("Failed reading recent values for %s", key)
            continue

    return 0.0, 0.0


def _is_invalid_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(numeric) or math.isinf(numeric)


def _normalize_expense(value: float) -> float:
    if value == 0.0:
        return 0.0
    return abs(value)


def _default_output() -> Dict[str, float]:
    return dict(DEFAULT_FINANCIAL_DATA)


def get_financial_data(ticker: str) -> Dict[str, float]:
    """Fetch structured annual financial data for a ticker.

    The function returns a complete dictionary with the required schema and
    falls back to zero-filled values if Yahoo Finance data cannot be loaded.
    """

    cleaned_ticker = _normalize_ticker(ticker)
    if not cleaned_ticker:
        logger.warning("Empty ticker supplied to get_financial_data")
        return _default_output()

    yf_module = _get_yfinance_module()
    if yf_module is None:
        return _default_output()

    try:
        ticker_obj = yf_module.Ticker(cleaned_ticker)
        financials = getattr(ticker_obj, "financials", None)
        balance_sheet = getattr(ticker_obj, "balance_sheet", None)
        cashflow = getattr(ticker_obj, "cashflow", None)

        rev_t, rev_prev = _get_latest_two_values(financials, STATEMENT_ALIASES["revenue"])
        net_income = _get_first_available(financials, STATEMENT_ALIASES["net_income"])
        equity = _get_first_available(balance_sheet, STATEMENT_ALIASES["equity"])

        debt = _get_first_available(balance_sheet, STATEMENT_ALIASES["debt"])
        if debt == 0.0 and balance_sheet is not None and not getattr(balance_sheet, "empty", True):
            long_term_debt = safe_get(balance_sheet, "Long Term Debt")
            short_term_debt = safe_get(balance_sheet, "Short Long Term Debt")
            debt = long_term_debt + short_term_debt

        current_assets = _get_first_available(balance_sheet, STATEMENT_ALIASES["current_assets"])
        current_liabilities = _get_first_available(balance_sheet, STATEMENT_ALIASES["current_liabilities"])
        ebit = _get_first_available(financials, STATEMENT_ALIASES["ebit"])

        interest_expense = _get_first_available(financials, STATEMENT_ALIASES["interest_expense"])
        if interest_expense == 0.0:
            interest_expense = _get_first_available(cashflow, STATEMENT_ALIASES["interest_expense"])
        interest_expense = _normalize_expense(interest_expense)

        output = {
            "rev_t": _safe_float(rev_t),
            "rev_prev": _safe_float(rev_prev),
            "net_income": _safe_float(net_income),
            "equity": _safe_float(equity),
            "debt": _safe_float(debt),
            "current_assets": _safe_float(current_assets),
            "current_liabilities": _safe_float(current_liabilities),
            "ebit": _safe_float(ebit),
            "interest_expense": _safe_float(interest_expense),
        }

        missing_fields = [key for key, value in output.items() if value == 0.0]
        if missing_fields:
            logger.info("Missing or zero financial fields for %s: %s", cleaned_ticker, ", ".join(missing_fields))

        logger.info("Successfully fetched financial data for %s", cleaned_ticker)
        return output

    except Exception as exc:
        logger.exception("Failed to fetch financial data for %s: %s", cleaned_ticker, exc)
        return _default_output()


__all__ = ["DEFAULT_FINANCIAL_DATA", "get_financial_data", "safe_get"]
