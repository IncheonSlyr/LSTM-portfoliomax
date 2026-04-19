from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def generate_sample_data(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic sample stock prices for demos and tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
    specs = {
        "AAPL": (150, 0.0005, 0.020),
        "GOOGL": (2800, 0.0004, 0.018),
        "MSFT": (300, 0.0006, 0.019),
        "AMZN": (3300, 0.0003, 0.022),
        "TSLA": (700, 0.0002, 0.035),
    }
    prices = {
        ticker: start * np.exp(np.cumsum(rng.normal(mu, sigma, n_days)))
        for ticker, (start, mu, sigma) in specs.items()
    }
    return pd.DataFrame(prices, index=dates)


def load_price_csv(source) -> pd.DataFrame:
    """Load a price CSV with a Date column or date-like first column."""
    df = pd.read_csv(source)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().all():
            df[first] = parsed
            df = df.set_index(first)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index().dropna(how="all").ffill().dropna()
    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV must contain a date column and at least two price columns.")
    return df


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels into clean daily returns."""
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def annualized_mean_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def asset_statistics(returns: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    mean_returns = annualized_mean_returns(returns)
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (mean_returns - risk_free_rate) / annual_vol.replace(0, np.nan)
    return pd.DataFrame(
        {
            "Annual Return (%)": mean_returns * 100,
            "Volatility (%)": annual_vol * 100,
            "Sharpe Ratio": sharpe,
            "Min Daily Return (%)": returns.min() * 100,
            "Max Daily Return (%)": returns.max() * 100,
        }
    )

