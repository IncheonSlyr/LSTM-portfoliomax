import unittest

import numpy as np

from portfolio.data import (
    annualized_covariance,
    annualized_mean_returns,
    calculate_returns,
    generate_sample_data,
)
from portfolio.optimization import build_strategy_weights
from portfolio.risk import portfolio_metrics, portfolio_returns, var_cvar
from portfolio.simulation import monte_carlo_gbm


class PortfolioCoreTests(unittest.TestCase):
    def setUp(self):
        self.prices = generate_sample_data(n_days=120, seed=123)
        self.returns = calculate_returns(self.prices)
        self.mean_returns = annualized_mean_returns(self.returns)
        self.cov_matrix = annualized_covariance(self.returns)

    def test_strategy_weights_are_valid(self):
        weights_by_strategy = build_strategy_weights(
            self.returns,
            self.mean_returns.values,
            self.cov_matrix.values,
            risk_free_rate=0.01,
            confidence_level=95,
            target_return=0.08,
        )
        self.assertEqual(set(weights_by_strategy), {"Equal Weight", "Max Sharpe", "Min CVaR"})
        for weights in weights_by_strategy.values():
            self.assertTrue(np.all(weights >= -1e-8))
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)

    def test_var_cvar_and_metrics_are_finite(self):
        weights = np.full(self.returns.shape[1], 1 / self.returns.shape[1])
        daily = portfolio_returns(self.returns, weights)
        var, cvar = var_cvar(daily, 95)
        self.assertLessEqual(cvar, var)
        metrics = portfolio_metrics(
            weights,
            self.returns,
            self.mean_returns.values,
            self.cov_matrix.values,
            risk_free_rate=0.01,
            confidence_level=95,
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_monte_carlo_shape(self):
        weights = np.full(self.returns.shape[1], 1 / self.returns.shape[1])
        sims = monte_carlo_gbm(self.prices, weights, n_sims=25, n_days=15, seed=9)
        self.assertEqual(sims.shape, (25, 15))
        self.assertTrue(np.all(sims[:, 0] == 100))
        self.assertTrue(np.all(sims > 0))


if __name__ == "__main__":
    unittest.main()
