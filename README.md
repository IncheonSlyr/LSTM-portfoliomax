# Portfolio Risk Assessment & Stochastic Optimization

Streamlit dashboard for portfolio risk analysis, simulation, volatility forecasting, and long-only portfolio optimization.

## What it does

- Loads stock price data from CSV or generates sample prices.
- Computes annualized returns, volatility, Sharpe ratio, VaR, CVaR, and drawdown.
- Runs correlated geometric Brownian motion Monte Carlo simulations.
- Optimizes three portfolio strategies:
  - Equal Weight
  - Max Sharpe, solved with constrained SLSQP optimization
  - Min CVaR, solved with a historical CVaR linear program
- Forecasts volatility with an LSTM when TensorFlow is installed, otherwise falls back to EWMA.
- Generates downloadable allocation recommendations for research use.

## Project structure

```text
.
├── app.py                    # Streamlit UI only
├── portfolio/
│   ├── data.py               # loading, sample data, returns, asset stats
│   ├── forecasting.py        # volatility forecast and validation diagnostics
│   ├── optimization.py       # constrained Max Sharpe and Min CVaR optimizers
│   ├── recommendations.py    # allocation recommendation table
│   ├── risk.py               # VaR, CVaR, metrics, performance, drawdown
│   └── simulation.py         # Monte Carlo GBM simulation
├── tests/
│   └── test_portfolio_core.py
├── data/stocks.csv           # example price data
├── file.py                   # yfinance data download helper
└── experiments/              # optional non-dashboard experiments
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow is optional. Without it, the app still runs and uses the EWMA volatility fallback.

## Run

```bash
streamlit run app.py
```

## CSV format

Use a `Date` column and one column per asset:

```csv
Date,AAPL,MSFT,GOOGL
2024-01-02,185.64,370.87,138.17
2024-01-03,184.25,370.60,138.92
```

## Smoke checks

The included tests avoid optional TensorFlow and focus on core analytics:

```bash
python -m unittest discover -s tests
python -m py_compile app.py portfolio/*.py file.py experiments/*.py
```

## Notes and limitations

This is a research and education dashboard, not financial advice. The models depend on historical data, simplified assumptions, and optimizer constraints. Validate any strategy out of sample before relying on it.
