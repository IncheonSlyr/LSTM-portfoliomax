import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from portfolio.data import (
    asset_statistics,
    annualized_covariance,
    annualized_mean_returns,
    calculate_returns,
    generate_sample_data,
    load_price_csv,
)
from portfolio.forecasting import forecast_volatility
from portfolio.optimization import build_strategy_weights
from portfolio.recommendations import build_recommendations
from portfolio.risk import (
    drawdown_table,
    portfolio_metrics,
    portfolio_returns,
    risk_contribution_table,
    simulate_portfolio_performance,
    var_cvar,
)
from portfolio.simulation import monte_carlo_gbm


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio Risk Assessment & Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric {
        background-color: #f7fafc;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    h1 { color: #1f77b4; padding-bottom: 1rem; }
    h2 { color: #2c3e50; padding-top: 1rem; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem; }
    h3 { color: #34495e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_sample_data() -> pd.DataFrame:
    return generate_sample_data()


@st.cache_data
def cached_monte_carlo(prices, weights, n_sims, n_days):
    return monte_carlo_gbm(prices, weights, n_sims, n_days)


@st.cache_data
def cached_forecast(stock_returns, window_size, epochs):
    return forecast_volatility(stock_returns, window_size, epochs)


def format_metrics(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "Annual Return (%)": f"{metrics['return']:.2f}",
        "Volatility (%)": f"{metrics['volatility']:.2f}",
        "Sharpe Ratio": f"{metrics['sharpe']:.3f}",
        "VaR (%)": f"{metrics['var']:.3f}",
        "CVaR (%)": f"{metrics['cvar']:.3f}",
    }


st.title("📈 Portfolio Risk Assessment & Stochastic Optimization")
st.markdown("### Risk analytics with Monte Carlo simulation, VaR/CVaR, forecasting, and constrained optimization")

st.sidebar.header("Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV with dates and stock prices", type=["csv"])

if uploaded_file is not None:
    try:
        data = load_price_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully")
    except Exception as exc:
        st.sidebar.error(f"Could not load file: {exc}")
        data = cached_sample_data()
else:
    data = cached_sample_data()
    st.sidebar.info("Using generated sample data")

st.sidebar.markdown("### Optimization Parameters")
target_return = st.sidebar.slider("Target Annual Return (%)", 5, 30, 12, 1) / 100
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0, 0.1) / 100
confidence_level = st.sidebar.slider("VaR/CVaR Confidence Level (%)", 90, 99, 95, 1)

st.sidebar.markdown("### Monte Carlo Parameters")
mc_simulations = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000, 1000)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 10, 252, 30, 10)

st.sidebar.markdown("### Forecasting Parameters")
lstm_window = st.sidebar.slider("Look-back Window", 5, 30, 10, 5)
lstm_epochs = st.sidebar.slider("Training Epochs", 10, 100, 30, 10)

returns = calculate_returns(data)
mean_returns = annualized_mean_returns(returns)
cov_matrix = annualized_covariance(returns)
n_assets = len(data.columns)
stats_df = asset_statistics(returns, risk_free_rate)

weights_by_strategy = build_strategy_weights(
    returns,
    mean_returns.values,
    cov_matrix.values,
    risk_free_rate,
    confidence_level,
    target_return,
)
metrics_by_strategy = {
    name: portfolio_metrics(weights, returns, mean_returns.values, cov_matrix.values, risk_free_rate, confidence_level)
    for name, weights in weights_by_strategy.items()
}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Monte Carlo",
        "Risk Metrics",
        "Forecasting",
        "Optimization",
        "Recommendations",
    ]
)

with tab1:
    st.header("Data Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        normalized = data / data.iloc[0] * 100
        for col in normalized.columns:
            ax.plot(normalized.index, normalized[col], label=col, linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price (Base=100)")
        ax.set_title("Normalized Stock Prices", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.metric("Number of Stocks", n_assets)
        st.metric("Trading Days", len(data))
        st.metric("Date Range", f"{data.index[0].date()} to {data.index[-1].date()}")
        st.metric("Avg Annual Return", f"{mean_returns.mean() * 100:.2f}%")
        st.metric("Avg Volatility", f"{returns.std().mean() * np.sqrt(252) * 100:.2f}%")

    st.subheader("Asset Statistics")
    st.dataframe(
        stats_df.style.format(
            {
                "Annual Return (%)": "{:.2f}",
                "Volatility (%)": "{:.2f}",
                "Sharpe Ratio": "{:.3f}",
                "Min Daily Return (%)": "{:.2f}",
                "Max Daily Return (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Daily Return Correlations", fontweight="bold")
    st.pyplot(fig)
    plt.close()

with tab2:
    st.header("Monte Carlo Simulation")
    st.caption("Uses correlated geometric Brownian motion estimated from historical log returns.")
    mc_weights = weights_by_strategy["Equal Weight"]
    with st.spinner("Running Monte Carlo simulations..."):
        mc_results = cached_monte_carlo(data, mc_weights, mc_simulations, forecast_days)

    final_values = mc_results[:, -1]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(min(100, mc_simulations)):
            ax.plot(mc_results[i], alpha=0.1, color="steelblue", linewidth=0.5)
        percentiles = np.percentile(mc_results, [5, 50, 95], axis=0)
        ax.plot(percentiles[1], color="darkblue", linewidth=2, label="Median")
        ax.plot(percentiles[0], color="red", linestyle="--", linewidth=2, label="5th percentile")
        ax.plot(percentiles[2], color="green", linestyle="--", linewidth=2, label="95th percentile")
        ax.set_xlabel("Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(f"{mc_simulations:,} Simulated Paths", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(final_values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        for value, color, label in [
            (np.median(final_values), "darkblue", "Median"),
            (np.percentile(final_values, 5), "red", "5th percentile"),
            (np.percentile(final_values, 95), "green", "95th percentile"),
        ]:
            ax.axvline(value, color=color, linestyle="--", linewidth=2, label=f"{label}: ${value:.2f}")
        ax.set_xlabel("Final Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution after {forecast_days} Days", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Value", f"${np.mean(final_values):.2f}")
    col2.metric("Median Value", f"${np.median(final_values):.2f}")
    col3.metric("Std Deviation", f"${np.std(final_values):.2f}")
    col4.metric("Probability of Profit", f"{(final_values > 100).mean() * 100:.1f}%")

with tab3:
    st.header("VaR and CVaR")
    selected_strategy = st.selectbox("Strategy", list(weights_by_strategy.keys()), key="risk_strategy")
    selected_returns = portfolio_returns(returns, weights_by_strategy[selected_strategy])
    var, cvar = var_cvar(selected_returns, confidence_level)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"VaR ({confidence_level}%)", f"{var * 100:.3f}%")
    col2.metric(f"CVaR ({confidence_level}%)", f"{cvar * 100:.3f}%")
    col3.metric("Annualized VaR", f"{var * np.sqrt(252) * 100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(selected_returns * 100, bins=50, color="steelblue", alpha=0.7, edgecolor="black", density=True)
        ax.axvline(var * 100, color="red", linestyle="--", linewidth=2, label="VaR")
        ax.axvline(cvar * 100, color="darkred", linestyle="--", linewidth=2, label="CVaR")
        ax.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"{selected_strategy} Return Distribution", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        levels = [90, 95, 97.5, 99, 99.5]
        values = [var_cvar(selected_returns, level)[0] * 100 for level in levels]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar([f"{level}%" for level in levels], values, color="coral", alpha=0.75, edgecolor="black")
        ax.set_xlabel("Confidence Level")
        ax.set_ylabel("VaR (%)")
        ax.set_title("VaR Sensitivity", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close()

    st.subheader("Asset Risk Table")
    st.dataframe(
        risk_contribution_table(returns, confidence_level).style.format(
            {
                f"VaR ({confidence_level}%)": "{:.3f}",
                f"CVaR ({confidence_level}%)": "{:.3f}",
                "Volatility (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

with tab4:
    st.header("Volatility Forecast")
    selected_stock = st.selectbox("Select Stock", data.columns)
    with st.spinner(f"Training forecast model for {selected_stock}..."):
        historical_vol, forecast, diagnostics = cached_forecast(returns[selected_stock], lstm_window, lstm_epochs)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_vol.index, historical_vol * 100, color="steelblue", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility (%)")
        ax.set_title(f"{selected_stock} Historical Volatility", fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, len(forecast) + 1), forecast * 100, color="coral", marker="o", linewidth=2)
        ax.axhline(historical_vol.iloc[-1] * 100, color="steelblue", linestyle="--", label="Current volatility")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Volatility (%)")
        ax.set_title(f"{selected_stock} Forecast", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", diagnostics["model"])
    col2.metric("Current Volatility", f"{historical_vol.iloc[-1] * 100:.2f}%")
    col3.metric("Avg Forecast Volatility", f"{np.mean(forecast) * 100:.2f}%")
    mae = diagnostics["validation_mae"]
    col4.metric("Validation MAE", "n/a" if mae is None else f"{mae * 100:.2f}%")

with tab5:
    st.header("Portfolio Optimization")
    st.caption("Max Sharpe and Min CVaR are solved as constrained long-only portfolios.")

    comparison_df = pd.DataFrame(
        [
            {"Strategy": name, **format_metrics(metrics)}
            for name, metrics in metrics_by_strategy.items()
        ]
    )
    numeric_comparison = pd.DataFrame(
        [
            {"Strategy": name, **metrics}
            for name, metrics in metrics_by_strategy.items()
        ]
    )
    st.dataframe(comparison_df, use_container_width=True)

    cols = st.columns(3)
    for idx, (name, weights) in enumerate(weights_by_strategy.items()):
        with cols[idx]:
            st.markdown(f"#### {name}")
            weights_df = pd.DataFrame({"Asset": data.columns, "Weight (%)": weights * 100})
            st.dataframe(weights_df.style.format({"Weight (%)": "{:.2f}"}), use_container_width=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(weights, labels=data.columns, autopct="%1.1f%%", startangle=90)
            ax.set_title(name, fontweight="bold")
            st.pyplot(fig)
            plt.close()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        for i, row in numeric_comparison.iterrows():
            ax.scatter(row["volatility"], row["return"], s=400, color=colors[i], edgecolors="black", label=row["Strategy"])
            ax.annotate(row["Strategy"], (row["volatility"], row["return"]), ha="center", va="center", fontsize=9)
        ax.set_xlabel("Volatility (%)")
        ax.set_ylabel("Annual Return (%)")
        ax.set_title("Risk-Return Trade-off", fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        performance_df = simulate_portfolio_performance(returns, weights_by_strategy)
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, col in enumerate(performance_df.columns):
            ax.plot(performance_df.index, performance_df[col], linewidth=2.5, color=colors[i], label=col)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Historical Portfolio Performance", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.subheader("Maximum Drawdown")
    dd_df = drawdown_table(returns, weights_by_strategy)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(dd_df.columns):
        ax.plot(dd_df.index, dd_df[col] * 100, linewidth=2, color=colors[i], label=col)
        ax.fill_between(dd_df.index, 0, dd_df[col] * 100, alpha=0.15, color=colors[i])
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Over Time", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tab6:
    st.header("Portfolio Recommendations")
    investment_amount = st.number_input("Total Investment Amount ($)", min_value=100, max_value=10000000, value=10000, step=1000)
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
    strategy_for_profile = {
        "Conservative": "Equal Weight",
        "Moderate": "Min CVaR",
        "Aggressive": "Max Sharpe",
    }[risk_profile]
    recommended_weights = weights_by_strategy[strategy_for_profile]
    recommendations_df = build_recommendations(data, returns, recommended_weights, investment_amount)

    st.info(
        f"Selected strategy: {strategy_for_profile}. These are quantitative model outputs for research use, not financial advice."
    )
    st.dataframe(
        recommendations_df.style.format(
            {
                "Weight (%)": "{:.2f}",
                "Amount ($)": "${:,.2f}",
                "Current Price ($)": "${:.2f}",
                "30-Day Return (%)": "{:+.2f}",
                "Volatility (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        csv = recommendations_df.to_csv(index=False)
        st.download_button(
            label="Download recommendation CSV",
            data=csv,
            file_name=f"portfolio_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.metric("Total Stocks", len(recommendations_df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.pie(recommended_weights, labels=data.columns, autopct="%1.1f%%", startangle=90)
    ax1.set_title(f"{strategy_for_profile} Allocation", fontweight="bold")
    ax2.barh(data.columns, recommended_weights * investment_amount, edgecolor="black")
    ax2.set_xlabel("Investment Amount ($)")
    ax2.set_title(f"Dollar Allocation (${investment_amount:,.0f})", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.caption(
    "Research dashboard. Models use historical data and simplified assumptions; validate independently before investment use."
)
