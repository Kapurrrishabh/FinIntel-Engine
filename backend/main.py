"""
FinIntel FastAPI Backend
========================
Production-grade backend with proper caching, error handling,
input validation, typed responses, and clean architecture.
"""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports — adjust paths to match your project structure
from backend.scraper.news_scraper import get_news
from backend.preprocessing.sentiment_model_scoring import run_sentiment_model
from backend.orchestration.complete_pipeline import run_complete_pipeline, _get_price_data

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CACHE_TTL_STOCK   = 300     # 5 min — stock analysis (ML pipeline is slow)
CACHE_TTL_MARKET  = 60      # 1 min — live ticker data
CACHE_TTL_NEWS    = 600     # 10 min — news items
CACHE_TTL_SEARCH  = 3600    # 1 hour — search results are mostly static

MARKET_TICKERS = [
    "^NSEI", "^BSESN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "AAPL", "NVDA", "MSFT", "TSLA",
]

TICKER_DISPLAY = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
}

# ─────────────────────────────────────────────
# CACHE STORE
# ─────────────────────────────────────────────
class TTLCache:
    """Simple in-memory TTL cache. Replace with Redis in production."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > entry["ttl"]:
            del self._store[key]
            return None
        return entry["data"]

    def set(self, key: str, data: Any, ttl: int) -> None:
        self._store[key] = {"data": data, "ts": time.time(), "ttl": ttl}

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def stats(self) -> dict:
        return {"entries": len(self._store)}


cache = TTLCache()

# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up market cache on startup
    print("🚀 FinIntel API starting — warming up market cache…")
    try:
        _build_market_data()
        print("✅ Market cache warmed up.")
    except Exception as e:
        print(f"⚠️  Market cache warmup failed: {e}")
    yield
    print("🛑 FinIntel API shutting down.")


app = FastAPI(
    title="FinIntel API",
    version="2.0.0",
    description="AI-powered financial intelligence backend.",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Lock down to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────
class PortfolioItem(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)


class PortfolioEntry(PortfolioItem):
    current_price: float
    pnl: float
    pnl_percent: float


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_entries: int
    timestamp: float


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def safe_json(data: Any) -> Any:
    """Recursively sanitise data for JSON serialisation."""
    if isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [safe_json(v) for v in data]
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return 0.0
    if isinstance(data, pd.Timestamp):
        return str(data.date())
    return data


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _display_name(sym: str) -> str:
    return TICKER_DISPLAY.get(sym, sym.replace(".NS", "").replace(".BO", ""))


# ─────────────────────────────────────────────
# INTERNAL: MARKET DATA BUILDER
# ─────────────────────────────────────────────
def _build_market_data() -> dict:
    """Download and structure live market data. Raises on failure."""
    data = yf.download(
        MARKET_TICKERS,
        period="5d",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
    )

    result: dict[str, list] = {"ticker": [], "trending": [], "insights": []}

    for sym in MARKET_TICKERS:
        try:
            df = (data[sym] if isinstance(data.columns, pd.MultiIndex) else data).dropna()
            if df.empty or len(df) < 1:
                continue

            current = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2]) if len(df) >= 2 else current
            change  = round(((current - prev) / prev) * 100, 2) if prev else 0.0
            display = _display_name(sym)
            is_indian = sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^")

            result["ticker"].append({
                "symbol": display,
                "price": f"{'₹' if is_indian else '$'}{current:,.2f}",
                "change": change,
            })

            if sym not in ("^NSEI", "^BSESN"):
                history = [round(float(x), 2) for x in df["Close"].tolist()[-8:]]
                result["trending"].append({
                    "symbol": sym,
                    "name": display,
                    "price": f"{'₹' if is_indian else '$'}{current:,.2f}",
                    "change": f"{'+' if change >= 0 else ''}{change}%",
                    "isUp": change >= 0,
                    "history": history,
                    "category": "indian" if is_indian else "us",
                })
        except Exception as e:
            print(f"⚠️  Skipping {sym}: {e}")
            continue

    # Derive a simple macro insight from NIFTY
    nifty_change = next((t["change"] for t in result["ticker"] if "NIFTY" in t["symbol"]), 0.0)
    bullish = nifty_change >= 0
    result["insights"] = [
        {
            "label": "Market Sentiment",
            "value": "Bullish" if bullish else "Bearish",
            "color": "text-emerald-400" if bullish else "text-rose-400",
            "bg": "bg-emerald-400/10" if bullish else "bg-rose-400/10",
            "conf": min(round(abs(nifty_change) * 40 + 55), 95),
            "icon": "📈" if bullish else "📉",
            "signal": "buy" if bullish else "sell",
            "risk": "medium",
            "trend": "up" if bullish else "down",
            "tooltip": f"Based on NIFTY 50 movement of {nifty_change:+.2f}% today.",
        }
    ]

    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health_check() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "cache_entries": cache.stats()["entries"],
        "timestamp": time.time(),
    }


# ── MARKET ───────────────────────────────────

@app.get("/market")
def get_market_data() -> dict:
    """Live market ticker + trending + macro insights."""
    cached = cache.get("market:global")
    if cached:
        return {**cached, "cached": True}

    try:
        data = _build_market_data()
        result = safe_json(data)
        cache.set("market:global", result, CACHE_TTL_MARKET)
        return {**result, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Market data unavailable: {e}")


# ── STOCK ANALYSIS ───────────────────────────

@app.get("/stock/{ticker}")
def get_stock_analysis(
    ticker: str = Path(..., min_length=1, max_length=20, description="Stock ticker symbol"),
) -> dict:
    """Full ML pipeline: technical + sentiment + fundamental analysis."""
    ticker = normalize_ticker(ticker)
    cache_key = f"stock:{ticker}"

    cached = cache.get(cache_key)
    if cached:
        print(f"⚡ CACHE HIT: {ticker}")
        return {**cached, "cached": True}

    print(f"🔬 CACHE MISS — running ML pipeline for {ticker}…")
    try:
        result    = run_complete_pipeline(ticker)
        price_df  = _get_price_data(ticker)

        prices = [
            {"date": str(idx.date()), "close": round(float(row["Close"]), 4)}
            for idx, row in price_df.tail(120).iterrows()
        ] if price_df is not None and not price_df.empty else []

        fundamentals: dict[str, Any] = {
            "roe": None, "debt_equity": None, "revenue_growth": None,
            "profit_margin": None, "market_cap": None, "pe_ratio": None, "eps": None,
        }

        try:
            metrics = result["fundamental"]["details"]["fundamental"]["metrics"]
            fundamentals.update({
                "roe":            round((metrics.get("return_on_equity") or 0) * 100, 2),
                "debt_equity":    metrics.get("debt_to_equity"),
                "revenue_growth": round((metrics.get("revenue_growth") or 0) * 100, 2),
                "profit_margin":  round((metrics.get("net_profit_margin") or 0) * 100, 2),
            })
        except Exception:
            pass

        try:
            info = yf.Ticker(ticker).info
            fundamentals.update({
                "market_cap": info.get("marketCap"),
                "pe_ratio":   info.get("trailingPE") or info.get("forwardPE"),
                "eps":        info.get("trailingEps") or info.get("forwardEps"),
                "sector":     info.get("sector"),
                "industry":   info.get("industry"),
                "name":       info.get("longName") or info.get("shortName"),
                "website":    info.get("website"),
            })
        except Exception:
            pass

        tech   = result.get("technical", {})
        sent   = result.get("sentiment", {})
        signal = tech.get("signal", "neutral")

        explanation = (
            f"Decision: {result.get('decision', 'N/A')}\n"
            f"Technical Signal: {signal}\n"
            f"Sentiment Score: {sent.get('sentiment_score', 0):.2f}\n"
            f"Final Score: {result.get('final_score', 0.5):.2f}"
        )

        response = safe_json({
            "symbol":           ticker,
            "name":             fundamentals.get("name", ticker),
            "prices":           prices,
            "technical_score":  round(tech.get("technical_score", 0.5), 4),
            "technical_signal": signal,
            "sentiment_score":  round(sent.get("sentiment_score", 0.5), 4),
            "fundamentals":     fundamentals,
            "final_score":      round(result.get("final_score", 0.5), 4),
            "decision":         result.get("decision", "HOLD"),
            "explanation":      explanation.strip(),
            "generated_at":     time.time(),
        })

        cache.set(cache_key, response, CACHE_TTL_STOCK)
        return {**response, "cached": False}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed for {ticker}: {str(e)}",
        )


# ── SEARCH ───────────────────────────────────

@app.get("/search/{query}")
async def search_stocks(
    query: str = Path(..., min_length=1, max_length=50),
) -> list[dict]:
    """Yahoo Finance symbol search with caching."""
    query = query.strip().upper()
    cache_key = f"search:{query}"

    cached = cache.get(cache_key)
    if cached:
        return cached

    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinIntel/2.0)", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()

        quotes = r.json().get("quotes", [])
        results = [
            {
                "symbol":   q["symbol"],
                "name":     q.get("shortname") or q.get("longname", ""),
                "exchange": q.get("exchange", ""),
                "type":     q.get("quoteType", "EQUITY"),
            }
            for q in quotes if "symbol" in q
        ][:10]

        cache.set(cache_key, results, CACHE_TTL_SEARCH)
        return results

    except httpx.TimeoutException:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Search service timed out.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Search failed: {e}")


# ── NEWS ─────────────────────────────────────

@app.get("/news/{ticker}")
def get_stock_news(ticker: str = Path(..., min_length=1, max_length=20)) -> dict:
    """Scrape, score sentiment, and return news for a given ticker."""
    ticker    = normalize_ticker(ticker)
    cache_key = f"news:{ticker}"

    cached = cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        raw_news = get_news(ticker)
        if not raw_news:
            return {"news": [], "sentiment": {}, "cached": False}

        formatted: list[dict] = []
        texts: list[str] = []

        for i, item in enumerate(raw_news[:12]):
            text = item.get("text", "") or ""
            texts.append(text)
            formatted.append({
                "id":       i + 1,
                "title":    item.get("title") or "Market Update",
                "url":      item.get("url") or item.get("link", ""),
                "time":     item.get("date", ""),
                "source":   item.get("source", "News"),
                "text":     text[:300],
                "sentiment": None,   # filled below
            })

        # Batch sentiment scoring
        try:
            sentiments = run_sentiment_model(texts)
            sent_labels = sentiments.get("labels", []) if isinstance(sentiments, dict) else []
            for i, label in enumerate(sent_labels):
                if i < len(formatted):
                    formatted[i]["sentiment"] = label.lower() if isinstance(label, str) else None
        except Exception as e:
            print(f"⚠️  Sentiment scoring failed: {e}")

        response = safe_json({"news": formatted, "sentiment": sentiments if "sentiments" in dir() else {}})
        cache.set(cache_key, response, CACHE_TTL_NEWS)
        return {**response, "cached": False}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"News fetch failed: {e}")


@app.get("/news/market")
def get_market_news() -> dict:
    """General market news (no specific ticker)."""
    cache_key = "news:market"
    cached = cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        # Use a broad term so the scraper returns general market news
        raw_news  = get_news("MARKET")
        formatted = [
            {
                "id":       i + 1,
                "title":    item.get("title", "Market Update"),
                "url":      item.get("url") or item.get("link", ""),
                "time":     item.get("date", ""),
                "source":   item.get("source", "News"),
                "text":     (item.get("text", "") or "")[:300],
                "sentiment": None,
            }
            for i, item in enumerate(raw_news[:10])
        ]
        response = safe_json({"news": formatted, "sentiment": {}})
        cache.set(cache_key, response, CACHE_TTL_NEWS)
        return {**response, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── PORTFOLIO ────────────────────────────────

# In-process store — swap for a real DB (SQLite / Postgres) in production
_portfolio: list[dict] = []


@app.get("/portfolio")
def get_portfolio() -> list[dict]:
    enriched: list[dict] = []
    for item in _portfolio:
        sym, qty, buy_price = item["symbol"], item["quantity"], item["price"]
        try:
            info = yf.Ticker(sym).fast_info
            current = float(getattr(info, "last_price", None) or buy_price)
        except Exception:
            current = buy_price

        pnl         = round((current - buy_price) * qty, 4)
        pnl_percent = round(((current - buy_price) / buy_price) * 100, 2) if buy_price else 0.0

        enriched.append({
            "symbol":        sym,
            "quantity":      qty,
            "buy_price":     buy_price,
            "current_price": round(current, 4),
            "pnl":           pnl,
            "pnl_percent":   pnl_percent,
        })

    return safe_json(enriched)


@app.post("/portfolio/add", status_code=status.HTTP_201_CREATED)
def add_to_portfolio(item: PortfolioItem) -> dict:
    entry = {
        "symbol":   normalize_ticker(item.symbol),
        "quantity": item.quantity,
        "price":    item.price,
        "added_at": time.time(),
    }
    _portfolio.append(entry)
    return {"message": "Added successfully", "data": entry}


@app.delete("/portfolio/remove/{symbol}")
def remove_from_portfolio(symbol: str = Path(..., min_length=1, max_length=20)) -> dict:
    global _portfolio
    sym    = normalize_ticker(symbol)
    before = len(_portfolio)
    _portfolio = [p for p in _portfolio if p["symbol"] != sym]
    if len(_portfolio) == before:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{sym} not found in portfolio.")
    return {"message": f"{sym} removed successfully."}


# ── CACHE MANAGEMENT ─────────────────────────

@app.delete("/cache/{key}")
def invalidate_cache(key: str) -> dict:
    """Manually bust a cache entry. Useful for testing or forced refresh."""
    cache.invalidate(key)
    return {"message": f"Cache entry '{key}' invalidated."}


@app.get("/cache/stats")
def cache_stats() -> dict:
    return cache.stats()