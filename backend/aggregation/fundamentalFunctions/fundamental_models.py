from typing import Dict, Any, Optional, Union
import numpy as np
import logging


logger = logging.getLogger(__name__)


def score_metric(
    value: float,
    threshold_good: float,
    threshold_bad: float,
    reverse: bool = False
) -> float:
    
    # Handle edge cases
    if np.isnan(value) or np.isinf(value):
        return 0.0
    
    if reverse:
        if value <= threshold_good:
            return 1.0
        elif value >= threshold_bad:
            return 0.0
        else:
            denominator = threshold_bad - threshold_good
            if denominator == 0:
                return 0.5
            return (threshold_bad - value) / denominator
    else:
        if value >= threshold_good:
            return 1.0
        elif value <= threshold_bad:
            return 0.0
        else:
            denominator = threshold_good - threshold_bad
            if denominator == 0:
                return 0.5
            return (value - threshold_bad) / denominator


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


# FUNDAMENTAL ANALYSIS

def fundamental_analysis(financial_data: Dict[str, float]) -> Dict[str, Any]:
    
    # Required keys validation
    required_keys = [
        "rev_t", "rev_prev", "net_income", "equity", "debt",
        "current_assets", "current_liabilities", "ebit", "interest_expense"
    ]
    
    for key in required_keys:
        if key not in financial_data:
            raise ValueError(f"Missing required key: {key}")
    
    # Extract values with safe defaults
    rev_t = financial_data.get("rev_t", 0)
    rev_prev = financial_data.get("rev_prev", 0)
    net_income = financial_data.get("net_income", 0)
    equity = financial_data.get("equity", 0)
    debt = financial_data.get("debt", 0)
    current_assets = financial_data.get("current_assets", 0)
    current_liabilities = financial_data.get("current_liabilities", 0)
    ebit = financial_data.get("ebit", 0)
    interest_expense = financial_data.get("interest_expense", 0)
    
    # Calculate raw metrics with safe division
    revenue_growth = safe_divide(rev_t - rev_prev, rev_prev, default=0.0)
    net_profit_margin = safe_divide(net_income, rev_t, default=0.0)
    roe = safe_divide(net_income, equity, default=0.0)
    debt_to_equity = safe_divide(debt, equity, default=float('inf'))
    if not current_assets or not current_liabilities:
        current_ratio = None
    else:
        current_ratio = safe_divide(current_assets, current_liabilities, default=0.0)
    interest_coverage = safe_divide(ebit, interest_expense, default=0.0)
    
    # Store raw metrics
    metrics = {
        "revenue_growth": round(revenue_growth, 4),
        "net_profit_margin": round(net_profit_margin, 4),
        "return_on_equity": round(roe, 4),
        "debt_to_equity": round(debt_to_equity, 4) if not np.isinf(debt_to_equity) else None,
        "current_ratio": round(current_ratio, 4) if current_ratio is not None else None,
        "interest_coverage": round(interest_coverage, 4)
    }

    if current_ratio is None:
        logger.info("Liquidity data missing, skipping liquidity evaluation")
    
    # Calculate normalized sub-scores (0 to 1)
    if current_ratio is None:
        liquidity_score = None
    else:
        liquidity_score = min(current_ratio / 2, 1.0)

    sub_scores = {
        "revenue_growth": round(score_metric(revenue_growth, 0.10, 0.0), 4),
        "profitability": round(score_metric(net_profit_margin, 0.15, 0.05), 4),
        "return_on_equity": round(score_metric(roe, 0.18, 0.08), 4),
        "debt_health": round(score_metric(debt_to_equity, 1.0, 2.0, reverse=True), 4),
        "liquidity": round(liquidity_score, 4) if liquidity_score is not None else None,
        "interest_coverage": round(score_metric(interest_coverage, 5.0, 2.0), 4)
    }
    
    # Calculate fundamental score from available (non-missing) sub-scores only.
    valid_scores = [score for score in sub_scores.values() if score is not None]
    fundamental_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    return {
        "fundamental_score": round(fundamental_score, 4),
        "metrics": metrics,
        "sub_scores": sub_scores
    }



# RISK ANALYSIS


def risk_analysis(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> Dict[str, Any]:
    
    # Input validation
    if returns is None or len(returns) == 0:
        return {
            "risk_score": 0.0,
            "metrics": {
                "volatility": None,
                "max_drawdown": None,
                "sharpe_ratio": None
            }
        }
    
    # Convert to numpy array if needed
    returns = np.asarray(returns, dtype=np.float64)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return {
            "risk_score": 0.0,
            "metrics": {
                "volatility": None,
                "max_drawdown": None,
                "sharpe_ratio": None
            }
        }
    
    # Calculate annualized volatility
    daily_std = np.std(returns, ddof=1)
    volatility = daily_std * np.sqrt(trading_days)
    
    # Calculate annualized return
    avg_daily_return = np.mean(returns)
    annualized_return = avg_daily_return * trading_days
    
    # Calculate Sharpe Ratio
    if volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0
    
    # Calculate Maximum Drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(np.min(drawdowns))
    
    # Handle edge cases
    if np.isnan(max_drawdown) or np.isinf(max_drawdown):
        max_drawdown = 0.0
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        sharpe_ratio = 0.0
    if np.isnan(volatility) or np.isinf(volatility):
        volatility = 1.0  # Assume high risk

    volatility_score = score_metric(volatility, 0.15, 0.40, reverse=True)
    drawdown_score = score_metric(max_drawdown, 0.20, 0.60, reverse=True)
    sharpe_score = score_metric(sharpe_ratio, 1.5, 0.5, reverse=False)
    
    risk_score = (
        volatility_score * 0.40 +
        drawdown_score * 0.30 +
        sharpe_score * 0.30
    )
    
    return {
        "risk_score": round(risk_score, 4),
        "metrics": {
            "volatility": round(volatility, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe_ratio, 4)
        }
    }



# GATEKEEPING (RULE-BASED FILTERS)


def gatekeeping(
    fundamental_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    config: Optional[Dict[str, float]] = None
) -> Dict[str, str]:
   
    # Default configuration
    default_config = {
        "min_interest_coverage_score": 0.1,
        "max_drawdown_limit": 0.65,
        "min_fundamental_score": 0.25,
        "min_liquidity_score": 0.2,
        "min_debt_health_score": 0.15
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    cfg = default_config
    
    # Extract values safely
    sub_scores = fundamental_result.get("sub_scores", {})
    risk_metrics = risk_result.get("metrics", {})
    fundamental_score = fundamental_result.get("fundamental_score", 0)
    
    # Rule 1: Interest Coverage Check
    interest_coverage_score = sub_scores.get("interest_coverage", 0)
    if interest_coverage_score < cfg["min_interest_coverage_score"]:
        return {
            "status": "REJECT",
            "reason": f"Interest coverage too low (score: {interest_coverage_score:.2f}). "
                     f"Company may struggle to service debt."
        }
    
    # Rule 2: Maximum Drawdown Check
    max_drawdown = risk_metrics.get("max_drawdown", 0)
    if max_drawdown and max_drawdown > cfg["max_drawdown_limit"]:
        return {
            "status": "REJECT",
            "reason": f"Extreme drawdown risk ({max_drawdown:.1%}). "
                     f"Historical losses exceed safe threshold."
        }
    
    # Rule 3: Minimum Fundamental Score
    if fundamental_score < cfg["min_fundamental_score"]:
        return {
            "status": "REJECT",
            "reason": f"Fundamental score too low ({fundamental_score:.2f}). "
                     f"Company financials are weak."
        }
    
    # Rule 4: Liquidity Check
    liquidity_score = sub_scores.get("liquidity", None)
    if liquidity_score is None:
        logger.info("Liquidity data missing, skipping liquidity evaluation")
    elif liquidity_score < cfg["min_liquidity_score"]:
        return {
            "status": "REJECT",
            "reason": f"Liquidity concerns (score: {liquidity_score:.2f}). "
                     f"Company may face short-term obligations issues."
        }
    
    # Rule 5: Debt Health Check
    debt_health_score = sub_scores.get("debt_health", 0)
    if debt_health_score < cfg["min_debt_health_score"]:
        return {
            "status": "REJECT",
            "reason": f"High debt burden (score: {debt_health_score:.2f}). "
                     f"Debt-to-equity ratio is concerning."
        }
    
    # All checks passed
    return {
        "status": "PASS",
        "reason": "All gatekeeping criteria satisfied."
    }



# FULL PIPELINE WRAPPER


def run_full_fundamental_pipeline(
    financial_data: Dict[str, float],
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    gatekeeping_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    
    # Run fundamental analysis
    try:
        fundamental_result = fundamental_analysis(financial_data)
    except (ValueError, KeyError, TypeError) as e:
        fundamental_result = {
            "fundamental_score": 0.0,
            "metrics": {},
            "sub_scores": {},
            "error": str(e)
        }
    
    # Run risk analysis
    try:
        risk_result = risk_analysis(returns, risk_free_rate=risk_free_rate)
    except (ValueError, TypeError) as e:
        risk_result = {
            "risk_score": 0.0,
            "metrics": {
                "volatility": None,
                "max_drawdown": None,
                "sharpe_ratio": None
            },
            "error": str(e)
        }
    
    # Run gatekeeping
    gatekeeping_result = gatekeeping(
        fundamental_result,
        risk_result,
        config=gatekeeping_config
    )
    
    
    fundamental_score = fundamental_result.get("fundamental_score", 0)
    risk_score = risk_result.get("risk_score", 0)
    
    if gatekeeping_result["status"] == "PASS":
        # Weighted combination: 60% fundamental, 40% risk-adjusted
        combined_score = (fundamental_score * 0.6) + (risk_score * 0.4)
    else:
        combined_score = 0.0
    
    # Generate recommendation
    if gatekeeping_result["status"] == "REJECT":
        recommendation = f"AVOID - {gatekeeping_result['reason']}"
    elif combined_score >= 0.75:
        recommendation = "STRONG BUY - Excellent fundamentals and risk profile"
    elif combined_score >= 0.60:
        recommendation = "BUY - Good fundamentals with acceptable risk"
    elif combined_score >= 0.45:
        recommendation = "HOLD - Average fundamentals, monitor closely"
    elif combined_score >= 0.30:
        recommendation = "WEAK HOLD - Below average, consider reducing position"
    else:
        recommendation = "SELL - Poor fundamentals and/or high risk"
    
    return {
        "fundamental": fundamental_result,
        "risk": risk_result,
        "gatekeeping": gatekeeping_result,
        "combined_score": round(combined_score, 4),
        "recommendation": recommendation
    }


