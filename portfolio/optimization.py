from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .risk import normalize_weights


def _equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1 / n_assets)


def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate: float) -> np.ndarray:
    """Long-only maximum Sharpe portfolio with sum(weights)=1."""
    mean_returns = np.asarray(mean_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n_assets = len(mean_returns)

    def objective(weights):
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if vol <= 0:
            return 1e6
        return -((mean_returns @ weights) - risk_free_rate) / vol

    result = minimize(
        objective,
        _equal_weights(n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return normalize_weights(result.x if result.success else _equal_weights(n_assets))


def optimize_min_variance(cov_matrix) -> np.ndarray:
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n_assets = cov_matrix.shape[0]
    result = minimize(
        lambda w: w.T @ cov_matrix @ w,
        _equal_weights(n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return normalize_weights(result.x if result.success else _equal_weights(n_assets))


def optimize_min_cvar(
    returns: pd.DataFrame,
    confidence_level: float,
    mean_returns=None,
    target_return: float | None = None,
) -> np.ndarray:
    """Long-only historical CVaR optimizer using a linear programming formulation."""
    returns_array = returns.to_numpy(dtype=float)
    periods, n_assets = returns_array.shape
    alpha = confidence_level / 100

    # Variables are weights(n), eta(1), tail_slacks(T).
    n_vars = n_assets + 1 + periods
    c = np.zeros(n_vars)
    c[n_assets] = 1.0
    c[n_assets + 1 :] = 1 / ((1 - alpha) * periods)

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    a_ub = []
    b_ub = []
    for t in range(periods):
        row = np.zeros(n_vars)
        row[:n_assets] = -returns_array[t]
        row[n_assets] = -1.0
        row[n_assets + 1 + t] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    if target_return is not None and mean_returns is not None:
        row = np.zeros(n_vars)
        row[:n_assets] = -np.asarray(mean_returns, dtype=float)
        a_ub.append(row)
        b_ub.append(-target_return)

    bounds = [(0.0, 1.0)] * n_assets + [(None, None)] + [(0.0, None)] * periods
    result = linprog(
        c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    return normalize_weights(result.x[:n_assets] if result.success else _equal_weights(n_assets))


def build_strategy_weights(
    returns: pd.DataFrame,
    mean_returns,
    cov_matrix,
    risk_free_rate: float,
    confidence_level: float,
    target_return: float,
) -> dict[str, np.ndarray]:
    n_assets = returns.shape[1]
    return {
        "Equal Weight": _equal_weights(n_assets),
        "Max Sharpe": optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate),
        "Min CVaR": optimize_min_cvar(
            returns,
            confidence_level,
            mean_returns=mean_returns,
            target_return=target_return,
        ),
    }

