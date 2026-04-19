from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def normalize_weights(weights) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).flatten()
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1 / len(weights))
    return weights / total


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    weights = normalize_weights(weights)
    values = returns.to_numpy() @ weights
    return pd.Series(values, index=returns.index, name="Portfolio Return")


def var_cvar(return_series, confidence_level: float) -> tuple[float, float]:
    series = np.asarray(return_series, dtype=float)
    var = np.percentile(series, 100 - confidence_level)
    tail = series[series <= var]
    cvar = tail.mean() if len(tail) else var
    return float(var), float(cvar)


def portfolio_metrics(
    weights,
    returns: pd.DataFrame,
    mean_returns,
    cov_matrix,
    risk_free_rate: float,
    confidence_level: float,
) -> dict[str, float]:
    weights = normalize_weights(weights)
    mean_returns = np.asarray(mean_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    annual_return = float(mean_returns @ weights)
    annual_volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0
    daily_returns = portfolio_returns(returns, weights)
    var, cvar = var_cvar(daily_returns, confidence_level)
    return {
        "return": annual_return * 100,
        "volatility": annual_volatility * 100,
        "sharpe": sharpe,
        "var": var * 100,
        "cvar": cvar * 100,
    }


def risk_contribution_table(returns: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    rows = []
    for asset in returns.columns:
        var, cvar = var_cvar(returns[asset], confidence_level)
        rows.append(
            {
                "Asset": asset,
                f"VaR ({confidence_level}%)": var * 100,
                f"CVaR ({confidence_level}%)": cvar * 100,
                "Volatility (%)": returns[asset].std() * np.sqrt(TRADING_DAYS) * 100,
            }
        )
    return pd.DataFrame(rows)


def simulate_portfolio_performance(
    returns: pd.DataFrame, weights_by_strategy: dict[str, np.ndarray], initial_value: float = 100
) -> pd.DataFrame:
    performance = {}
    for name, weights in weights_by_strategy.items():
        daily = portfolio_returns(returns, weights)
        performance[name] = (1 + daily).cumprod() * initial_value
    return pd.DataFrame(performance, index=returns.index)


def drawdown_from_returns(return_series) -> pd.Series:
    series = pd.Series(return_series)
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def drawdown_table(returns: pd.DataFrame, weights_by_strategy: dict[str, np.ndarray]) -> pd.DataFrame:
    data = {
        name: drawdown_from_returns(portfolio_returns(returns, weights)).to_numpy()
        for name, weights in weights_by_strategy.items()
    }
    return pd.DataFrame(data, index=returns.index)

