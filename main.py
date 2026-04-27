"""CLI entrypoint for FinIntel analysis."""

from __future__ import annotations

import argparse
import json
from typing import List

from orchestrator import analyze_stock, analyze_stocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinIntel analysis for one or more tickers.")
    parser.add_argument("ticker", nargs="*", help="Ticker symbol(s), e.g. AAPL NVDA TSLA")
    parser.add_argument("--period", default="2y", help="Historical lookback period for market data")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tickers: List[str] = [str(item).strip() for item in args.ticker if str(item).strip()]

    if not tickers:
        print(json.dumps({"status": "error", "error": "Please provide at least one ticker."}, indent=2))
        return 1

    if len(tickers) == 1:
        payload = analyze_stock(tickers[0], period=args.period)
    else:
        payload = analyze_stocks(tickers, period=args.period)

    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())