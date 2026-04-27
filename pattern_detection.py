"""Pattern detection utilities for forecast sequences.

The detector works on a predicted price sequence and returns a compact
classification that is easy to surface in both the API and frontend.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _to_1d_array(predicted_sequence: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(predicted_sequence, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim > 1:
        values = values.reshape(-1)
    values = values[np.isfinite(values)]
    return values


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    value = numerator / denominator
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _linear_slope(series: np.ndarray) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=np.float64)
    slope = np.polyfit(x, series, 1)[0]
    return float(slope)


def _segment_slope(series: np.ndarray, start: int, end: int) -> float:
    segment = series[start:end]
    if len(segment) < 2:
        return 0.0
    x = np.arange(len(segment), dtype=np.float64)
    return float(np.polyfit(x, segment, 1)[0])


def detect_pattern(predicted_sequence: Sequence[float] | np.ndarray) -> Dict[str, Any]:
    """Classify a forecast path into a price-action pattern.

    Returns:
        {
            "pattern": "uptrend | downtrend | sideways | breakout | reversal | volatile",
            "strength": float in [0, 1],
            "description": "human readable explanation"
        }
    """

    prices = _to_1d_array(predicted_sequence)
    if prices.size < 4:
        return {
            "pattern": "insufficient_data",
            "strength": 0.0,
            "description": "Not enough forecast points to detect a stable pattern.",
        }

    first_price = float(prices[0])
    last_price = float(prices[-1])
    price_range = float(np.max(prices) - np.min(prices))
    pct_change = _safe_ratio(last_price - first_price, abs(first_price) + 1e-9)

    diffs = np.diff(prices)
    volatility = float(np.std(diffs) / (np.mean(np.abs(prices)) + 1e-9))
    slope = _linear_slope(prices)
    first_slope = _segment_slope(prices, 0, max(3, len(prices) // 3))
    last_slope = _segment_slope(prices, max(0, len(prices) - max(3, len(prices) // 3)), len(prices))

    positive_ratio = float(np.mean(diffs > 0)) if len(diffs) else 0.0
    negative_ratio = float(np.mean(diffs < 0)) if len(diffs) else 0.0
    turning_point = np.sign(first_slope) != np.sign(last_slope) and abs(first_slope - last_slope) > abs(slope) * 1.2
    breakout_move = abs(diffs[-1]) > max(np.std(diffs) * 2.0, np.mean(np.abs(diffs)) * 2.5, 1e-9)

    uptrend_score = max(0.0, min(1.0, 0.45 * positive_ratio + 0.35 * max(0.0, pct_change) + 0.20 * max(0.0, slope / (abs(first_price) + 1e-9))))
    downtrend_score = max(0.0, min(1.0, 0.45 * negative_ratio + 0.35 * max(0.0, -pct_change) + 0.20 * max(0.0, -slope / (abs(first_price) + 1e-9))))
    sideways_score = max(0.0, min(1.0, 1.0 - min(1.0, abs(pct_change) * 6.0) - min(0.6, abs(slope) / (abs(first_price) + 1e-9) * 25.0) - min(0.4, volatility * 8.0)))

    if breakout_move and abs(pct_change) > 0.03:
        pattern = "breakout"
        direction = "upward" if pct_change >= 0 else "downward"
        strength = max(0.55, min(1.0, 0.5 + abs(pct_change) * 4.0 + min(0.3, volatility * 2.0)))
        description = (
            f"{direction.capitalize()} breakout detected from the forecast path. "
            f"The terminal move is unusually large relative to recent forecast volatility."
        )
    elif turning_point and abs(pct_change) > 0.02:
        pattern = "reversal"
        direction = "bullish" if last_slope > 0 else "bearish"
        strength = max(0.55, min(1.0, 0.45 + abs(first_slope - last_slope) / (abs(slope) + 1e-9) * 0.2 + abs(pct_change) * 3.0))
        description = (
            f"Potential {direction} reversal: the early forecast slope differs materially from the later slope."
        )
    elif uptrend_score >= max(downtrend_score, sideways_score):
        pattern = "uptrend"
        strength = float(np.clip(uptrend_score, 0.0, 1.0))
        description = "Forecast prices show higher highs and a positive slope across the horizon."
    elif downtrend_score >= max(uptrend_score, sideways_score):
        pattern = "downtrend"
        strength = float(np.clip(downtrend_score, 0.0, 1.0))
        description = "Forecast prices show lower lows and a negative slope across the horizon."
    elif sideways_score >= 0.5:
        pattern = "sideways"
        strength = float(np.clip(sideways_score, 0.0, 1.0))
        description = "Forecast prices remain range-bound with limited directional conviction."
    else:
        pattern = "volatile"
        strength = float(np.clip(0.5 + min(0.5, volatility * 3.0), 0.0, 1.0))
        description = "Forecast path is mixed and noisy, with no clear directional structure."

    return {
        "pattern": pattern,
        "strength": round(float(np.clip(strength, 0.0, 1.0)), 4),
        "description": description,
        "metrics": {
            "slope": round(float(slope), 6),
            "pct_change": round(float(pct_change), 6),
            "volatility": round(float(volatility), 6),
            "range": round(float(price_range), 6),
            "positive_step_ratio": round(float(positive_ratio), 4),
            "negative_step_ratio": round(float(negative_ratio), 4),
        },
    }


__all__ = ["detect_pattern"]