"""Unified FinIntel orchestration pipeline.

This module combines:
* market data and technical indicators
* transformer-based forecasting
* GRU technical signal
* fundamentals
* risk
* sentiment
* pattern detection
* final decision engine

The output is intentionally frontend-friendly and JSON serializable.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import TimeSeriesTransformerForPrediction

from backend.aggregation.fundamentalFunctions.fundamental_models import (
    fundamental_analysis,
    risk_analysis,
)
from backend.orchestration.complete_pipeline import get_sentiment_result
from backend.preprocessing.stock_feature_scraper import build_technical_features, fetch_price_data, run_technical_model

from decision_engine import decide_action
from pattern_detection import detect_pattern


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_PERIOD = "2y"
HISTORY_OUTPUT_POINTS = 120
CACHE_TTL_SECONDS = 300


def _clean_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


class AnalysisCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[Tuple[str, str], Tuple[datetime, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None

            timestamp, payload = entry
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if age > self.ttl_seconds:
                self._store.pop(key, None)
                return None

            return copy.deepcopy(payload)

    def set(self, key: Tuple[str, str], value: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (datetime.now(timezone.utc), copy.deepcopy(value))


@dataclass(frozen=True)
class TransformerArtifacts:
    model_dir: Path
    metadata: Dict[str, Any]
    ticker_to_id: Dict[str, int]
    ticker_scalers: Dict[str, StandardScaler]
    context_length: int
    prediction_length: int
    history_length: int
    time_feature_columns: Tuple[str, ...]
    lags_sequence: Tuple[int, ...]
    device: torch.device


class OrchestrationError(RuntimeError):
    pass


def _candidate_transformer_dirs() -> List[Path]:
    return [
        ROOT / "finintel_ts_transformer" / "exported_assets",
        ROOT / "backend" / "models" / "exported_assets",
    ]


def _candidate_transformer_model_dirs() -> List[Path]:
    return [candidate / "model" for candidate in _candidate_transformer_dirs()]


def _candidate_feature_columns_path() -> Path:
    return ROOT / "backend" / "models" / "technical_model" / "feature_columns.json"


@lru_cache(maxsize=1)
def _load_transformer_artifacts() -> Optional[TransformerArtifacts]:
    metadata = None
    ticker_scalers = None
    ticker_to_id = None
    model_dir = None

    for candidate in _candidate_transformer_dirs():
        metadata_path = candidate / "metadata.json"
        ticker_to_id_path = candidate / "ticker_encoder.json"
        scalers_path = candidate / "ticker_scalers.pkl"
        candidate_model = candidate / "model"
        if metadata_path.exists() and ticker_to_id_path.exists() and scalers_path.exists() and candidate_model.exists():
            model_dir = candidate_model
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ticker_to_id = json.loads(ticker_to_id_path.read_text(encoding="utf-8"))
            with open(scalers_path, "rb") as handle:
                ticker_scalers = pickle.load(handle)
            break

    if model_dir is None or metadata is None or ticker_to_id is None or ticker_scalers is None:
        logger.warning("Transformer artifact bundle not found; pipeline will use a deterministic fallback forecast.")
        return None

    context_length = int(metadata.get("context_length", 60))
    prediction_length = int(metadata.get("prediction_length", 30))
    history_length = int(metadata.get("history_length", context_length + 7))
    time_feature_columns = tuple(metadata.get("time_feature_columns", ["day_of_week", "month"]))
    lags_sequence = tuple(metadata.get("lags_sequence", [1, 2, 3, 4, 5, 6, 7]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return TransformerArtifacts(
        model_dir=model_dir,
        metadata=metadata,
        ticker_to_id=ticker_to_id,
        ticker_scalers=ticker_scalers,
        context_length=context_length,
        prediction_length=prediction_length,
        history_length=history_length,
        time_feature_columns=time_feature_columns,
        lags_sequence=lags_sequence,
        device=device,
    )


@lru_cache(maxsize=1)
def _load_transformer_model() -> Optional[TimeSeriesTransformerForPrediction]:
    artifacts = _load_transformer_artifacts()
    if artifacts is None:
        return None

    try:
        model = TimeSeriesTransformerForPrediction.from_pretrained(str(artifacts.model_dir))
        model.to(artifacts.device)
        model.eval()
        return model
    except Exception as exc:
        logger.exception("Failed to load transformer model from %s: %s", artifacts.model_dir, exc)
        return None


@lru_cache(maxsize=1)
def _load_feature_columns() -> Optional[List[str]]:
    path = _candidate_feature_columns_path()
    if not path.exists():
        return None
    try:
        return list(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Failed reading technical feature columns: %s", exc)
        return None


def _normalize_sequences(sample_sequences: np.ndarray) -> np.ndarray:
    sequences = np.asarray(sample_sequences)
    if sequences.ndim == 4:
        if sequences.shape[0] == 1:
            sequences = sequences[0]
        elif sequences.shape[1] == 1:
            sequences = sequences[:, 0]
    if sequences.ndim == 3 and sequences.shape[-1] == 1:
        sequences = sequences[..., 0]
    if sequences.ndim == 3 and sequences.shape[1] == 1:
        sequences = sequences[:, 0, :]
    if sequences.ndim == 1:
        sequences = sequences[np.newaxis, :]
    return sequences


def _build_calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    day_of_week = index.dayofweek.astype(np.float32) / 6.0
    month = (index.month.astype(np.float32) - 1.0) / 11.0
    return np.column_stack([day_of_week, month]).astype(np.float32)


def _future_business_dates(last_timestamp: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_timestamp) + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=periods)


def _latest_indicator_snapshot(price_df: pd.DataFrame) -> Dict[str, Any]:
    feature_columns = _load_feature_columns()
    if not feature_columns:
        return {}

    try:
        features = build_technical_features(price_df, feature_columns)
        if features.empty:
            return {}
        latest_row = features.ffill().bfill().iloc[-1].to_dict()
        return _to_builtin(latest_row)
    except Exception as exc:
        logger.debug("Unable to build indicator snapshot: %s", exc)
        return {}


def _build_history_points(price_df: pd.DataFrame, limit: int = HISTORY_OUTPUT_POINTS) -> List[Dict[str, Any]]:
    history = price_df.sort_index().tail(limit)
    return [
        {"time": pd.Timestamp(index).isoformat(), "price": round(_safe_float(row["Close"]), 4)}
        for index, row in history.iterrows()
    ]


def _fit_or_reuse_scaler(artifacts: Optional[TransformerArtifacts], ticker: str, returns: pd.Series) -> StandardScaler:
    if artifacts is not None:
        scaler = artifacts.ticker_scalers.get(ticker)
        if scaler is not None:
            return scaler

    scaler = StandardScaler()
    values = returns.dropna().to_numpy(dtype=np.float32).reshape(-1, 1)
    if len(values) == 0:
        values = np.zeros((1, 1), dtype=np.float32)
    scaler.fit(values)
    return scaler


def _forecast_with_transformer(price_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    artifacts = _load_transformer_artifacts()
    model = _load_transformer_model()

    close = price_df["Close"].astype(np.float64)
    log_returns = np.log(close).diff()
    if len(price_df) < 20:
        raise OrchestrationError("Not enough price data for transformer forecasting.")

    prediction_length = artifacts.prediction_length if artifacts else 30
    history_length = artifacts.history_length if artifacts else 67
    time_feature_columns = artifacts.time_feature_columns if artifacts else ("day_of_week", "month")
    ticker_id = (artifacts.ticker_to_id.get(ticker, 0) if artifacts else 0)

    scaler = _fit_or_reuse_scaler(artifacts, ticker, log_returns)
    history_frame = price_df.tail(history_length).copy()
    if len(history_frame) < history_length:
        pad_count = history_length - len(history_frame)
        first_row = history_frame.iloc[0]
        first_index = pd.Timestamp(history_frame.index[0])
        pad_index = pd.bdate_range(end=first_index - pd.offsets.BDay(1), periods=pad_count)
        pad_frame = pd.DataFrame(
            {column: [first_row[column]] * pad_count for column in history_frame.columns},
            index=pad_index,
        )
        history_frame = pd.concat([pad_frame, history_frame], axis=0)

    history_index = pd.DatetimeIndex(history_frame.index)
    history_returns = np.log(history_frame["Close"]).diff().fillna(0.0)
    history_scaled = scaler.transform(history_returns.to_numpy(dtype=np.float32).reshape(-1, 1)).reshape(-1)
    history_time_features = _build_calendar_features(history_index)
    future_index = _future_business_dates(pd.Timestamp(price_df.index[-1]), prediction_length)
    future_time_features = _build_calendar_features(future_index)
    last_close = float(history_frame["Close"].iloc[-1])

    if model is None or artifacts is None:
        logger.info("Using deterministic fallback forecast for %s", ticker)
        steps = np.arange(1, prediction_length + 1, dtype=np.float64)
        drift = float(np.nanmean(log_returns.dropna().tail(20))) if not log_returns.dropna().empty else 0.0
        volatility = float(np.nanstd(log_returns.dropna().tail(20))) if not log_returns.dropna().empty else 0.01
        projected_returns = np.clip(drift + np.linspace(0.0, volatility * 0.25, prediction_length), -0.25, 0.25)
        price_sequence = last_close * np.exp(np.cumsum(projected_returns))
        lower = last_close * np.exp(np.cumsum(projected_returns - volatility * 1.96))
        upper = last_close * np.exp(np.cumsum(projected_returns + volatility * 1.96))
        std = np.full(prediction_length, volatility, dtype=np.float64)
        confidence_score = float(np.clip(1.0 - min(0.9, volatility * 4.0), 0.0, 1.0))
        return {
            "sequence": price_sequence.tolist(),
            "curve": [
                {"time": ts.isoformat(), "predicted": round(float(value), 4)}
                for ts, value in zip(future_index, price_sequence)
            ],
            "confidence": {
                "score": round(confidence_score, 4),
                "mean": price_sequence.tolist(),
                "std": std.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "average_band_width_pct": round(float(np.mean((upper - lower) / (price_sequence + 1e-9))), 4),
            },
            "future_dates": [ts.isoformat() for ts in future_index],
        }

    if ticker not in artifacts.ticker_to_id:
        logger.info("Ticker %s not in transformer encoder; using fallback ID 0", ticker)

    model_inputs = {
        "past_values": torch.tensor(history_scaled, dtype=torch.float32, device=artifacts.device).unsqueeze(0),
        "past_time_features": torch.tensor(history_time_features, dtype=torch.float32, device=artifacts.device).unsqueeze(0),
        "future_time_features": torch.tensor(future_time_features, dtype=torch.float32, device=artifacts.device).unsqueeze(0),
        "past_observed_mask": torch.ones((1, history_length), dtype=torch.float32, device=artifacts.device),
        "static_categorical_features": torch.tensor([[ticker_id]], dtype=torch.long, device=artifacts.device),
    }

    with torch.no_grad():
        generated = model.generate(
            past_values=model_inputs["past_values"],
            past_time_features=model_inputs["past_time_features"],
            future_time_features=model_inputs["future_time_features"],
            past_observed_mask=model_inputs["past_observed_mask"],
            static_categorical_features=model_inputs["static_categorical_features"],
        )

    samples = _normalize_sequences(generated.sequences.detach().cpu().numpy())
    if samples.shape[1] != prediction_length and samples.shape[0] == prediction_length:
        samples = samples.T

    if samples.ndim != 2:
        samples = samples.reshape(-1, prediction_length)

    sampled_returns = scaler.inverse_transform(samples.reshape(-1, 1)).reshape(samples.shape)
    sampled_prices = last_close * np.exp(np.cumsum(sampled_returns, axis=1))

    mean_prices = sampled_prices.mean(axis=0)
    std_prices = sampled_prices.std(axis=0)
    lower_prices = np.percentile(sampled_prices, 2.5, axis=0)
    upper_prices = np.percentile(sampled_prices, 97.5, axis=0)

    confidence_score = float(
        np.clip(
            1.0 - np.mean((upper_prices - lower_prices) / (mean_prices + 1e-9)) / 2.0,
            0.0,
            1.0,
        )
    )

    return {
        "sequence": mean_prices.tolist(),
        "curve": [
            {"time": ts.isoformat(), "predicted": round(float(value), 4)}
            for ts, value in zip(future_index, mean_prices)
        ],
        "confidence": {
            "score": round(confidence_score, 4),
            "mean": mean_prices.tolist(),
            "std": std_prices.tolist(),
            "lower": lower_prices.tolist(),
            "upper": upper_prices.tolist(),
            "band": [
                {
                    "time": ts.isoformat(),
                    "lower": round(float(lower), 4),
                    "upper": round(float(upper), 4),
                }
                for ts, lower, upper in zip(future_index, lower_prices, upper_prices)
            ],
            "average_band_width_pct": round(float(np.mean((upper_prices - lower_prices) / (mean_prices + 1e-9))), 4),
        },
        "future_dates": [ts.isoformat() for ts in future_index],
    }


def _prepare_market_data(ticker: str, period: str) -> Dict[str, Any]:
    price_df = fetch_price_data(ticker, period=period)
    if price_df is None or price_df.empty:
        raise OrchestrationError(f"No market data available for {ticker}.")

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    price_df = price_df.sort_index().copy()
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.dropna(subset=["Close"]).copy()
    if price_df.empty:
        raise OrchestrationError(f"No valid close data available for {ticker}.")

    history_points = _build_history_points(price_df)
    latest_close = float(price_df["Close"].iloc[-1])
    latest_indicators = _latest_indicator_snapshot(price_df)
    returns = price_df["Close"].pct_change().dropna().to_numpy(dtype=np.float64)

    return {
        "price_df": price_df,
        "history_points": history_points,
        "last_close": latest_close,
        "latest_indicators": latest_indicators,
        "returns": returns,
    }


def _normalize_signal_signal(raw_signal: Mapping[str, Any], label_key: str, score_key: str) -> Dict[str, Any]:
    return {
        label_key: raw_signal.get(label_key),
        score_key: round(_safe_float(raw_signal.get(score_key, 0.5)), 4),
        "confidence": raw_signal.get("confidence"),
        "details": _to_builtin(raw_signal),
    }


class StockAnalysisOrchestrator:
    def __init__(self, cache_ttl_seconds: int = CACHE_TTL_SECONDS):
        self.cache = AnalysisCache(ttl_seconds=cache_ttl_seconds)

    def analyze_stock(self, ticker: str, period: str = DEFAULT_PERIOD) -> Dict[str, Any]:
        cleaned_ticker = _clean_ticker(ticker)
        cache_key = (cleaned_ticker, period)
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached["meta"]["cache_hit"] = True
            return cached

        if not cleaned_ticker:
            return {
                "status": "error",
                "ticker": "",
                "error": "Ticker is required.",
            }

        try:
            market = _prepare_market_data(cleaned_ticker, period)
            price_df = market["price_df"]

            transformer = _forecast_with_transformer(price_df, cleaned_ticker)
            gru_result = run_technical_model(price_df)

            financial_data = {}
            try:
                from backend.scraper.fundamental_financial_scraper import get_financial_data

                financial_data = get_financial_data(cleaned_ticker)
            except Exception as exc:
                logger.warning("Fundamental data fetch failed for %s: %s", cleaned_ticker, exc)
                financial_data = {}

            fundamental_result = fundamental_analysis(financial_data)
            risk_result = risk_analysis(market["returns"])
            sentiment_result = get_sentiment_result(cleaned_ticker)
            pattern_result = detect_pattern(transformer["sequence"])
            decision_result = decide_action(
                {
                    "market": {"last_close": market["last_close"]},
                    "prediction": transformer,
                    "signals": {
                        "gru": gru_result,
                        "fundamental": fundamental_result,
                        "risk": risk_result,
                        "sentiment": sentiment_result,
                    },
                    "pattern": {
                        "type": pattern_result.get("pattern"),
                        "strength": pattern_result.get("strength", 0.0),
                        "description": pattern_result.get("description", ""),
                    },
                }
            )

            response = {
                "status": "success",
                "ticker": cleaned_ticker,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "market": {
                    "last_close": round(market["last_close"], 4),
                    "history": market["history_points"],
                    "latest_indicators": market["latest_indicators"],
                    "period": period,
                },
                "prediction": {
                    "sequence": transformer["sequence"],
                    "curve": transformer["curve"],
                    "confidence": transformer["confidence"],
                },
                "pattern": {
                    "type": pattern_result.get("pattern"),
                    "strength": pattern_result.get("strength", 0.0),
                    "description": pattern_result.get("description", ""),
                    "metrics": pattern_result.get("metrics", {}),
                },
                "signals": {
                    "gru": {
                        "signal": gru_result.get("signal", "HOLD"),
                        "technical_score": round(_safe_float(gru_result.get("technical_score", 0.5)), 4),
                        "confidence": gru_result.get("confidence", "medium"),
                        "details": _to_builtin(gru_result),
                    },
                    "fundamental": {
                        "signal": "POSITIVE" if fundamental_result.get("fundamental_score", 0.0) >= 0.6 else "NEUTRAL",
                        "fundamental_score": round(_safe_float(fundamental_result.get("fundamental_score", 0.0)), 4),
                        "details": _to_builtin(fundamental_result),
                    },
                    "risk": {
                        "signal": "LOW" if risk_result.get("risk_score", 0.0) >= 0.67 else ("MEDIUM" if risk_result.get("risk_score", 0.0) >= 0.45 else "HIGH"),
                        "risk_score": round(_safe_float(risk_result.get("risk_score", 0.0)), 4),
                        "details": _to_builtin(risk_result),
                    },
                    "sentiment": {
                        "signal": "BULLISH" if sentiment_result.get("sentiment_score", 0.5) >= 0.6 else ("BEARISH" if sentiment_result.get("sentiment_score", 0.5) <= 0.4 else "NEUTRAL"),
                        "sentiment_score": round(_safe_float(sentiment_result.get("sentiment_score", 0.5)), 4),
                        "article_count": int(sentiment_result.get("num_articles", 0) or 0),
                        "details": _to_builtin(sentiment_result),
                    },
                },
                "decision": decision_result,
                "summary": {
                    "headline": f"{cleaned_ticker} is {decision_result['action']} with {round(decision_result['confidence'] * 100)}% confidence",
                    "explanation": decision_result["rationale"],
                },
                "meta": {
                    "cache_hit": False,
                    "transformer_available": _load_transformer_model() is not None,
                    "period": period,
                    "model_bundle": str(_load_transformer_artifacts().model_dir) if _load_transformer_artifacts() else None,
                },
            }

            response = _to_builtin(response)
            self.cache.set(cache_key, response)
            return response

        except Exception as exc:
            logger.exception("Analysis failed for %s", cleaned_ticker)
            return {
                "status": "error",
                "ticker": cleaned_ticker,
                "error": str(exc),
            }

    def analyze_stocks(self, tickers: Sequence[str], period: str = DEFAULT_PERIOD) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for ticker in tickers:
            results.append(self.analyze_stock(ticker, period=period))
        return results


_ORCHESTRATOR = StockAnalysisOrchestrator()


def analyze_stock(ticker: str, period: str = DEFAULT_PERIOD) -> Dict[str, Any]:
    return _ORCHESTRATOR.analyze_stock(ticker, period=period)


def analyze_stocks(tickers: Sequence[str], period: str = DEFAULT_PERIOD) -> List[Dict[str, Any]]:
    return _ORCHESTRATOR.analyze_stocks(tickers, period=period)


__all__ = ["StockAnalysisOrchestrator", "analyze_stock", "analyze_stocks"]