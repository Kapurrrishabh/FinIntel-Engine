"""Decision engine for FinIntel.

The engine converts heterogeneous model outputs into a single action:
BUY, SELL, or HOLD, with a transparent confidence score and rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(float(value), low, high))


def _safe_score(section: Mapping[str, Any], key: str, default: float = 0.5) -> float:
    try:
        return _clamp(section.get(key, default))
    except Exception:
        return default


def _pattern_score(pattern: Mapping[str, Any], last_close: float, forecast_last: float) -> float:
    pattern_type = str(pattern.get("type") or pattern.get("pattern") or "").lower()
    strength = _clamp(pattern.get("strength", 0.5))
    direction = 1.0 if forecast_last >= last_close else -1.0

    if pattern_type == "uptrend":
        return _clamp(0.58 + 0.42 * strength)
    if pattern_type == "downtrend":
        return _clamp(0.42 - 0.42 * strength)
    if pattern_type == "sideways":
        return _clamp(0.50 - 0.08 * strength)
    if pattern_type == "breakout":
        return _clamp(0.50 + 0.35 * strength * direction)
    if pattern_type == "reversal":
        return _clamp(0.50 + 0.25 * strength * direction)
    if pattern_type == "volatile":
        return _clamp(0.48 - 0.12 * strength)
    return 0.50


def _transformer_score(prediction: Mapping[str, Any], last_close: float) -> float:
    sequence = prediction.get("sequence") or []
    confidence = prediction.get("confidence") or {}
    if not sequence:
        return 0.5

    forecast_last = float(sequence[-1])
    forecast_first = float(sequence[0])
    horizon_return = (forecast_last - forecast_first) / (abs(forecast_first) + 1e-9)
    terminal_return = (forecast_last - last_close) / (abs(last_close) + 1e-9)
    band_width = confidence.get("average_band_width_pct", 0.0)
    band_penalty = _clamp(float(band_width) / 0.20, 0.0, 1.0)
    trend_component = 1.0 / (1.0 + np.exp(-5.0 * float(horizon_return)))
    terminal_component = 1.0 / (1.0 + np.exp(-5.0 * float(terminal_return)))
    confidence_component = 1.0 - band_penalty

    return _clamp(0.45 * trend_component + 0.35 * terminal_component + 0.20 * confidence_component)


def decide_action(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate a final decision from the aggregated analysis payload."""

    market = analysis.get("market", {})
    prediction = analysis.get("prediction", {})
    signals = analysis.get("signals", {})
    pattern = analysis.get("pattern", {})

    last_close = float(market.get("last_close", 0.0) or 0.0)
    forecast_sequence = prediction.get("sequence") or []
    forecast_last = float(forecast_sequence[-1]) if forecast_sequence else last_close

    transformer_score = _transformer_score(prediction, last_close)
    gru_score = _safe_score(signals.get("gru", {}), "technical_score", 0.5)
    fundamental_score = _safe_score(signals.get("fundamental", {}), "fundamental_score", 0.5)
    risk_score = _safe_score(signals.get("risk", {}), "risk_score", 0.5)
    sentiment_score = _safe_score(signals.get("sentiment", {}), "sentiment_score", 0.5)
    pattern_score = _pattern_score(pattern, last_close, forecast_last)

    weights = {
        "transformer": 0.30,
        "gru": 0.18,
        "fundamental": 0.18,
        "risk": 0.14,
        "sentiment": 0.10,
        "pattern": 0.10,
    }

    composite_score = (
        weights["transformer"] * transformer_score
        + weights["gru"] * gru_score
        + weights["fundamental"] * fundamental_score
        + weights["risk"] * risk_score
        + weights["sentiment"] * sentiment_score
        + weights["pattern"] * pattern_score
    )

    pattern_type = str(pattern.get("type") or pattern.get("pattern") or "").lower()
    strong_bearish_pattern = pattern_type in {"downtrend", "reversal"} and pattern.get("strength", 0.0) >= 0.7 and forecast_last <= last_close
    strong_bullish_pattern = pattern_type in {"uptrend", "breakout"} and pattern.get("strength", 0.0) >= 0.7 and forecast_last >= last_close

    if strong_bullish_pattern and composite_score >= 0.52:
        action = "BUY"
    elif strong_bearish_pattern and composite_score <= 0.60:
        action = "SELL"
    elif composite_score >= 0.67:
        action = "BUY"
    elif composite_score >= 0.45:
        action = "HOLD"
    else:
        action = "SELL"

    confidence_signal = _clamp(
        0.35 * transformer_score
        + 0.15 * abs(composite_score - 0.5) * 2.0
        + 0.15 * pattern.get("strength", 0.5)
        + 0.15 * risk_score
        + 0.10 * fundamental_score
        + 0.10 * sentiment_score
    )

    rationale = (
        f"Composite score {composite_score:.2f} driven by transformer={transformer_score:.2f}, "
        f"gru={gru_score:.2f}, fundamental={fundamental_score:.2f}, risk={risk_score:.2f}, "
        f"sentiment={sentiment_score:.2f}, pattern={pattern_score:.2f}."
    )

    return {
        "action": action,
        "confidence": round(confidence_signal, 4),
        "score": round(float(composite_score), 4),
        "rationale": rationale,
        "weights": weights,
        "component_scores": {
            "transformer": round(transformer_score, 4),
            "gru": round(gru_score, 4),
            "fundamental": round(fundamental_score, 4),
            "risk": round(risk_score, 4),
            "sentiment": round(sentiment_score, 4),
            "pattern": round(pattern_score, 4),
        },
    }


__all__ = ["decide_action"]