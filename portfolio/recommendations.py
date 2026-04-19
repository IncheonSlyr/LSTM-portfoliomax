from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def build_recommendations(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    weights,
    investment_amount: float,
) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float)
    n_assets = len(weights)
    equal_weight = 1 / n_assets
    rows = []

    for i, asset in enumerate(prices.columns):
        weight = weights[i]
        amount = weight * investment_amount
        recent_window = min(30, len(prices) - 1)
        recent_return = (prices[asset].iloc[-1] / prices[asset].iloc[-recent_window] - 1) * 100
        volatility = returns[asset].std() * np.sqrt(TRADING_DAYS) * 100

        if weight > equal_weight * 1.5:
            action = "STRONG BUY"
        elif weight > equal_weight * 1.1:
            action = "BUY"
        elif weight > equal_weight * 0.7:
            action = "HOLD"
        elif weight > equal_weight * 0.3:
            action = "REDUCE"
        else:
            action = "SELL/AVOID"

        rows.append(
            {
                "Stock": asset,
                "Action": action,
                "Weight (%)": weight * 100,
                "Amount ($)": amount,
                "Current Price ($)": prices[asset].iloc[-1],
                "Approx Shares": int(amount / prices[asset].iloc[-1]) if amount > 0 else 0,
                "30-Day Return (%)": recent_return,
                "Volatility (%)": volatility,
            }
        )

    return pd.DataFrame(rows).sort_values("Weight (%)", ascending=False)
