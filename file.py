from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "JNJ", "WMT", "V", "PG"]


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    output_path: Path,
) -> None:
    data = yf.download(tickers, start=start, end=end, progress=True)
    if data.empty:
        raise RuntimeError("No data returned from yfinance.")

    if hasattr(data.columns, "get_level_values") and "Adj Close" in data.columns.get_level_values(0):
        prices = data["Adj Close"]
    elif hasattr(data.columns, "get_level_values") and "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index_label="Date")

    print(f"Downloaded {len(prices.columns)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Shape: {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download adjusted close prices for the portfolio dashboard.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols to download.")
    parser.add_argument("--start", default="2021-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default="2024-01-01", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--output", default="data/stocks.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_prices(args.tickers, args.start, args.end, Path(args.output))
