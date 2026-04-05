"""Complete orchestration pipeline for FinIntel.

This module provides modular entry points that can be used by FastAPI or
other application layers to run the full stock analysis pipeline.

Public functions:
    get_technical_result(ticker)
    get_sentiment_result(ticker)
    get_fundamental_result(ticker)
    fuse_and_decide(tech, fund, senti, risk)
    run_complete_pipeline(ticker)
"""

from __future__ import annotations

import importlib
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TECH_MODEL_PATH = ROOT / "models" / "technical_model" / "gru_stock_classifier-2.keras"
TECH_FEATURE_PATH = ROOT / "models" / "technical_model" / "feature_columns.json"
SENTIMENT_MODEL_PATH = ROOT / "models" / "sentiment_model" / "sentiment_expert_model_v1"

_PRICE_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_SENTIMENT_MODEL = None
_SENTIMENT_TOKENIZER = None
_SENTIMENT_CACHE: Dict[str, Dict[str, Any]] = {}
_SENTIMENT_CACHE_TTL_HOURS = 24


def _clean_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _is_valid_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(numeric) or math.isinf(numeric))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas scalar types to JSON-safe Python types."""

    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as exc:
        logger.warning("Unable to import %s: %s", module_path, exc)
        return None


def _load_yfinance_module():
    try:
        return importlib.import_module("yfinance")
    except Exception as exc:
        logger.warning("yfinance unavailable in orchestration layer: %s", exc)
        return None


def _get_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    cache_key = (_clean_ticker(ticker), period)
    if cache_key in _PRICE_CACHE:
        logger.info("Using cached price data for %s", cache_key[0])
        return _PRICE_CACHE[cache_key].copy()

    logger.info("Fetching price data for %s (%s)", cache_key[0], period)

    stock_feature_module = _load_module("backend.preprocessing.stock_feature_scraper")
    if stock_feature_module is not None and hasattr(stock_feature_module, "fetch_price_data"):
        try:
            price_df = stock_feature_module.fetch_price_data(cache_key[0], period=period)
            if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                _PRICE_CACHE[cache_key] = price_df.copy()
                return price_df.copy()
        except Exception as exc:
            logger.warning("backend.preprocessing.stock_feature_scraper.fetch_price_data failed: %s", exc)

    yfinance_module = _load_yfinance_module()
    if yfinance_module is None:
        logger.error("No price data source available for %s", cache_key[0])
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    try:
        price_df = yfinance_module.download(cache_key[0], period=period, progress=False)
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df = price_df[["Open", "High", "Low", "Close", "Volume"]]
        _PRICE_CACHE[cache_key] = price_df.copy()
        return price_df.copy()
    except Exception as exc:
        logger.exception("Price fetch failed for %s: %s", cache_key[0], exc)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _load_technical_model_runner():
    stock_feature_module = _load_module("backend.preprocessing.stock_feature_scraper")
    if stock_feature_module is not None and hasattr(stock_feature_module, "run_technical_model"):
        return stock_feature_module.run_technical_model
    return None


def _load_sentiment_model():
    global _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER
    if _SENTIMENT_MODEL is not None and _SENTIMENT_TOKENIZER is not None:
        return _SENTIMENT_TOKENIZER, _SENTIMENT_MODEL

    if not SENTIMENT_MODEL_PATH.exists():
        logger.info("Sentiment model path not found; using fallback sentiment scoring")
        return None, None

    try:
        transformers_module = importlib.import_module("transformers")
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        logger.warning("Sentiment model dependencies unavailable: %s", exc)
        return None, None

    try:
        _SENTIMENT_TOKENIZER = transformers_module.AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_PATH))
        _SENTIMENT_MODEL = transformers_module.AutoModelForSequenceClassification.from_pretrained(
            str(SENTIMENT_MODEL_PATH)
        )
        _SENTIMENT_MODEL.eval()
        _SENTIMENT_MODEL.to(torch_module.device("cpu"))
        logger.info("Loaded sentiment model from %s", SENTIMENT_MODEL_PATH)
        return _SENTIMENT_TOKENIZER, _SENTIMENT_MODEL
    except Exception as exc:
        logger.warning("Failed to load sentiment model: %s", exc)
        _SENTIMENT_MODEL = None
        _SENTIMENT_TOKENIZER = None
        return None, None


def _fallback_sentiment_score(text: str) -> float:
    text_lower = text.lower()
    positive_words = ["beat", "bull", "buy", "growth", "gain", "up", "profit", "strong", "surge"]
    negative_words = ["bear", "sell", "loss", "down", "risk", "drop", "weak", "concern", "miss"]
    pos = sum(word in text_lower for word in positive_words)
    neg = sum(word in text_lower for word in negative_words)
    score = 0.5 + 0.1 * (pos - neg)
    return float(np.clip(score, 0.0, 1.0))


def _score_sentiment_texts(texts: List[str]) -> List[float]:
    texts = [str(text).strip() for text in texts if str(text).strip()]
    if not texts:
        return []

    tokenizer, model = _load_sentiment_model()
    if tokenizer is None or model is None:
        logger.info("Using fallback sentiment scoring for %d texts", len(texts))
        return [_fallback_sentiment_score(text) for text in texts]

    torch_module = importlib.import_module("torch")
    scores: List[float] = []

    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch_module.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch_module.softmax(logits, dim=0).cpu().numpy()
                score = float(probabilities[2] * 1.0 + probabilities[1] * 0.5 + probabilities[0] * 0.0)
                scores.append(score)
        except Exception as exc:
            logger.warning("Sentiment scoring failed for one item: %s", exc)
            scores.append(_fallback_sentiment_score(text))

    return scores


def _extract_risk_result(details: Dict[str, Any]) -> Dict[str, Any]:
    risk = details.get("risk", {}) if isinstance(details, dict) else {}
    if isinstance(risk, dict):
        return {
            "risk_score": _safe_float(risk.get("risk_score", 0.0)),
            "details": risk,
        }
    return {"risk_score": 0.0, "details": {}}


def _get_cached_sentiment(ticker: str) -> Dict[str, Any] | None:
    entry = _SENTIMENT_CACHE.get(ticker)
    if not entry:
        return None
    ts = entry.get("timestamp")
    if not isinstance(ts, datetime):
        return None
    if datetime.now(timezone.utc) - ts > timedelta(hours=_SENTIMENT_CACHE_TTL_HOURS):
        return None
    return entry


def _store_sentiment_cache(ticker: str, sentiment_score: float, num_articles: int, news_volume_score: float) -> None:
    _SENTIMENT_CACHE[ticker] = {
        "sentiment_score": float(sentiment_score),
        "num_articles": int(num_articles),
        "news_volume_score": float(news_volume_score),
        "timestamp": datetime.now(timezone.utc),
    }


def _compute_news_volume_score(num_articles: int) -> float:
    if num_articles <= 0:
        return 0.0
    return float(np.clip(num_articles / 20.0, 0.0, 1.0))


def get_technical_result(ticker: str) -> dict:
    """Fetch price data and run the technical model."""

    cleaned_ticker = _clean_ticker(ticker)
    logger.info("Starting technical analysis for %s", cleaned_ticker)

    price_df = _get_price_data(cleaned_ticker)
    if price_df.empty:
        logger.warning("No price data available for technical analysis: %s", cleaned_ticker)
        return {"technical_score": 0.5, "signal": "HOLD", "confidence": "low"}

    runner = _load_technical_model_runner()
    if runner is not None:
        try:
            result = runner(price_df)
            if isinstance(result, dict):
                logger.info("Technical model completed for %s", cleaned_ticker)
                return result
        except Exception as exc:
            logger.exception("Technical model failed for %s: %s", cleaned_ticker, exc)

    try:
        ma20 = price_df["Close"].rolling(20).mean().iloc[-1]
        ma50 = price_df["Close"].rolling(50).mean().iloc[-1]
        score = float(np.clip(0.5 + np.sign(ma20 - ma50) * 0.25, 0.0, 1.0))
        signal = "BUY" if ma20 > ma50 else ("SELL" if ma20 < ma50 else "HOLD")
        confidence = "high" if abs(ma20 - ma50) / (ma50 + 1e-9) > 0.01 else "medium"
        logger.info("Technical fallback completed for %s", cleaned_ticker)
        return {"technical_score": score, "signal": signal, "confidence": confidence}
    except Exception as exc:
        logger.exception("Technical fallback failed for %s: %s", cleaned_ticker, exc)
        return {"technical_score": 0.5, "signal": "HOLD", "confidence": "low"}


def get_sentiment_result(ticker: str) -> dict:
    """Fetch news, score each item, and aggregate sentiment."""

    cleaned_ticker = _clean_ticker(ticker)
    logger.info("Starting sentiment analysis for %s", cleaned_ticker)

    cached_entry = _get_cached_sentiment(cleaned_ticker)

    news_module = _load_module("backend.scraper.news_scraper")
    if news_module is None or not hasattr(news_module, "get_news"):
        logger.warning("News scraper unavailable for %s", cleaned_ticker)
        return {"sentiment_score": 0.5, "num_articles": 0}

    try:
        news_items = news_module.get_news(cleaned_ticker) or []
    except Exception as exc:
        logger.exception("News fetching failed for %s: %s", cleaned_ticker, exc)
        return {"sentiment_score": 0.5, "num_articles": 0}

    texts = []
    for item in news_items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)

    if not texts:
        if cached_entry:
            logger.info("No sentiment data found for %s; reusing cached sentiment", cleaned_ticker)
            return {
                "sentiment_score": float(cached_entry["sentiment_score"]),
                "num_articles": int(cached_entry["num_articles"]),
                "news_volume_score": float(cached_entry.get("news_volume_score", 0.0)),
                "used_cache": True,
            }
        logger.info("No sentiment data found, using neutral fallback")
        return {"sentiment_score": 0.5, "num_articles": 0, "news_volume_score": 0.0, "used_cache": False}

    scores = _score_sentiment_texts(texts)
    if not scores:
        if cached_entry:
            logger.info("Sentiment scoring empty for %s; reusing cached sentiment", cleaned_ticker)
            return {
                "sentiment_score": float(cached_entry["sentiment_score"]),
                "num_articles": int(cached_entry["num_articles"]),
                "news_volume_score": float(cached_entry.get("news_volume_score", 0.0)),
                "used_cache": True,
            }
        logger.info("No sentiment data found, using neutral fallback")
        return {"sentiment_score": 0.5, "num_articles": 0, "news_volume_score": 0.0, "used_cache": False}

    base_sentiment_score = float(np.mean(scores))
    news_volume_score = _compute_news_volume_score(len(texts))
    sentiment_score = float(np.clip((0.85 * base_sentiment_score) + (0.15 * news_volume_score), 0.0, 1.0))
    _store_sentiment_cache(cleaned_ticker, sentiment_score, len(texts), news_volume_score)

    logger.info("Sentiment completed for %s with %d articles", cleaned_ticker, len(texts))
    return {
        "sentiment_score": sentiment_score,
        "num_articles": len(texts),
        "news_volume_score": news_volume_score,
        "used_cache": False,
    }


def get_fundamental_result(ticker: str) -> dict:
    """Fetch financial data, compute returns, and run the fundamental pipeline."""

    cleaned_ticker = _clean_ticker(ticker)
    logger.info("Starting fundamental analysis for %s", cleaned_ticker)

    financial_module = _load_module("backend.scraper.fundamental_financial_scraper")
    fundamental_module = _load_module("backend.aggregation.fundamentalFunctions.fundamental_models")

    financial_data = {}
    if financial_module is not None and hasattr(financial_module, "get_financial_data"):
        try:
            financial_data = financial_module.get_financial_data(cleaned_ticker)
        except Exception as exc:
            logger.exception("Financial data fetch failed for %s: %s", cleaned_ticker, exc)
            financial_data = {}

    if fundamental_module is None or not hasattr(fundamental_module, "run_full_fundamental_pipeline"):
        logger.warning("Fundamental pipeline unavailable for %s", cleaned_ticker)
        return {"fundamental_score": 0.0, "risk_score": 0.0, "details": {}}

    price_df = _get_price_data(cleaned_ticker)
    if price_df.empty or "Close" not in price_df.columns:
        logger.warning("No price data available for risk analysis: %s", cleaned_ticker)
        returns = np.array([], dtype=np.float64)
    else:
        returns = price_df["Close"].pct_change().dropna().to_numpy(dtype=np.float64)

    try:
        pipeline_result = fundamental_module.run_full_fundamental_pipeline(financial_data, returns)
    except Exception as exc:
        logger.exception("Fundamental pipeline failed for %s: %s", cleaned_ticker, exc)
        return {"fundamental_score": 0.0, "risk_score": 0.0, "details": {}}

    fundamental_score = _safe_float(pipeline_result.get("fundamental", {}).get("fundamental_score", 0.0))
    risk_score = _safe_float(pipeline_result.get("risk", {}).get("risk_score", 0.0))

    logger.info("Fundamental analysis completed for %s", cleaned_ticker)
    return {
        "fundamental_score": fundamental_score,
        "risk_score": risk_score,
        "details": _to_builtin(pipeline_result),
    }


def fuse_and_decide(tech, fund, senti, risk):
    tech_score = tech.get('technical_score', 0.5)
    fund_score = fund.get('fundamental_score', 0.0) if isinstance(fund, dict) else 0.0
    senti_score = senti.get('sentiment_score', 0.5)
    risk_score = risk.get('risk_score', 0.0)

    final_score = 0.35 * tech_score + 0.30 * fund_score + 0.20 * senti_score + 0.15 * risk_score

    if final_score >= 0.70:
        decision = 'BUY'
    elif final_score >= 0.45:
        decision = 'HOLD'
    else:
        decision = 'SELL'

    return float(final_score), decision


def run_complete_pipeline(ticker: str) -> dict:
    """Run the full analysis pipeline for a ticker."""

    cleaned_ticker = _clean_ticker(ticker)
    logger.info("Starting complete pipeline for %s", cleaned_ticker)

    technical = {"technical_score": 0.5, "signal": "HOLD", "confidence": "low"}
    sentiment = {"sentiment_score": 0.5, "num_articles": 0}
    fundamental = {"fundamental_score": 0.0, "risk_score": 0.0, "details": {}}
    risk = {"risk_score": 0.0, "details": {}}

    try:
        technical = get_technical_result(cleaned_ticker)
    except Exception as exc:
        logger.exception("Technical step failed for %s: %s", cleaned_ticker, exc)

    try:
        sentiment = get_sentiment_result(cleaned_ticker)
    except Exception as exc:
        logger.exception("Sentiment step failed for %s: %s", cleaned_ticker, exc)

    try:
        fundamental = get_fundamental_result(cleaned_ticker)
    except Exception as exc:
        logger.exception("Fundamental step failed for %s: %s", cleaned_ticker, exc)

    risk = _extract_risk_result(fundamental.get("details", {}))
    if not risk.get("risk_score", 0.0):
        risk = {
            "risk_score": _safe_float(fundamental.get("risk_score", 0.0)),
            "details": fundamental.get("details", {}).get("risk", {}) if isinstance(fundamental.get("details", {}), dict) else {},
        }

    try:
        final_score, decision = fuse_and_decide(technical, fundamental, sentiment, risk)
    except Exception as exc:
        logger.exception("Fusion step failed for %s: %s", cleaned_ticker, exc)
        final_score, decision = 0.0, "SELL"

    logger.info("Pipeline completed for %s: %s (%.4f)", cleaned_ticker, decision, final_score)

    result = {
        "ticker": cleaned_ticker,
        "technical": technical,
        "sentiment": sentiment,
        "fundamental": fundamental,
        "risk": risk,
        "final_score": float(final_score),
        "decision": decision,
    }
    return _to_builtin(result)


__all__ = [
    "get_technical_result",
    "get_sentiment_result",
    "get_fundamental_result",
    "fuse_and_decide",
    "run_complete_pipeline",
]
