"""FastAPI surface for FinIntel analysis."""

from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import analyze_stock, analyze_stocks


logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinIntel API",
    version="1.0.0",
    description="Unified stock intelligence API for forecasting, pattern detection, fundamentals, risk, and sentiment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, description="Ticker list to analyze")
    period: str = Field(default="2y", description="Historical lookback period")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/analyze/{ticker}")
def analyze_ticker(ticker: str, period: str = Query(default="2y", description="Historical lookback period")) -> dict:
    payload = analyze_stock(ticker, period=period)
    if payload.get("status") == "error":
        raise HTTPException(status_code=400, detail=payload)
    return payload


@app.post("/api/analyze/batch")
def analyze_batch(request: BatchRequest) -> dict:
    payload = analyze_stocks(request.tickers, period=request.period)
    return {"status": "success", "results": payload}


@app.get("/")
def root() -> dict:
    return {
        "service": "FinIntel API",
        "status": "ok",
        "endpoints": ["/api/health", "/api/analyze/{ticker}", "/api/analyze/batch"],
    }


__all__ = ["app"]