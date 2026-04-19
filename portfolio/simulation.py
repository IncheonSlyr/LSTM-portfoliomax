from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_gbm(
    prices: pd.DataFrame,
    weights,
    n_sims: int,
    n_days: int,
    initial_value: float = 100,
    seed: int = 7,
) -> np.ndarray:
    """Simulate portfolio values from correlated GBM log returns."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean().to_numpy()
    cov = log_returns.cov().to_numpy()
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(mu, cov, size=(n_sims, n_days - 1))
    simple_asset_returns = np.exp(shocks) - 1
    portfolio_daily_returns = simple_asset_returns @ weights
    values = np.empty((n_sims, n_days))
    values[:, 0] = initial_value
    values[:, 1:] = initial_value * np.cumprod(1 + portfolio_daily_returns, axis=1)
    return values

