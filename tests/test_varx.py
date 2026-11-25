"""
tests/test_varx.py
Unit tests for the VARX module.
"""

import unittest
import numpy as np
from rmgarchpy.varx import (
    _build_lag_matrices,
    varx_fit,
    varx_forecast,
    varx_irf,
    varx_fevd,
    varx_select_lag
)

class TestVARX(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        np.random.seed(123)
        self.n_obs = 200
        self.k = 2  # 2 endogenous variables
        self.m = 1  # 1 exogenous variable
        
        # Generate simple random walk data
        self.Y = np.random.randn(self.n_obs, self.k)
        self.X = np.random.randn(self.n_obs, self.m)
        
    def test_build_lag_matrices_no_exog(self):
        """Test lag matrix construction without exogenous variables."""
        lags_y = 2
        Z, Y_target = _build_lag_matrices(self.Y, X=None, lags_y=lags_y, lags_x=0, constant=True)
        
        # Expected shape: (n - max_lag, k * lags_y + 1)
        expected_n_obs = self.n_obs - lags_y
        expected_features = self.k * lags_y + 1 
        
        self.assertEqual(Z.shape, (expected_n_obs, expected_features))
        self.assertEqual(Y_target.shape, (expected_n_obs, self.k))
        
        # Check if the last column is all 1s (constant term)
        np.testing.assert_array_equal(Z[:, -1], np.ones(expected_n_obs))

    def test_build_lag_matrices_with_exog(self):
        """Test lag matrix construction with exogenous variables."""
        lags_y = 1
        lags_x = 1
        Z, Y_target = _build_lag_matrices(self.Y, X=self.X, lags_y=lags_y, lags_x=lags_x, constant=True)
        
        expected_n_obs = self.n_obs - max(lags_y, lags_x)
        expected_features = self.k * lags_y + self.m * lags_x + 1
        
        self.assertEqual(Z.shape, (expected_n_obs, expected_features))

    def test_varx_fit_ols(self):
        """Test OLS estimation accuracy on simulated data."""
        # DGP: Y_t = 0.5 * Y_{t-1} + 0.2 * X_t + e_t
        A1 = np.array()
        B1 = np.array()
        
        Y = np.zeros((self.n_obs, self.k))
        X = np.random.randn(self.n_obs, self.m)
        
        for t in range(1, self.n_obs):
            Y[t] = A1 @ Y[t-1] + B1.flatten() * X[t] + np.random.randn(self.k) * 0.1
            
        fit = varx_fit(Y, X, lags_y=1, lags_x=1, constant=False)
        
        # Check shapes
        self.assertEqual(fit['coef'].shape, (self.k, self.k * 1 + self.m * 1))
        self.assertEqual(fit['residuals'].shape, (self.n_obs - 1, self.k))
        
        # Check if estimated coefficients are close to true DGP
        # coef structure: [A1_11, A1_12, B1_1], [A1_21, A1_22, B1_2]
        np.testing.assert_allclose(fit['coef'][0, :2], A1[0], atol=0.1)
        np.testing.assert_allclose(fit['coef'][1, :2], A1[1], atol=0.1)
        np.testing.assert_allclose(fit['coef'][0, 2], B1[0, 0], atol=0.1)

    def test_varx_forecast(self):
        """Test forecast functionality."""
        fit = varx_fit(self.Y, self.X, lags_y=1, lags_x=1, constant=True)
        n_ahead = 5
        X_future = np.zeros((n_ahead, self.m))
        
        forecast = varx_forecast(fit, self.Y, self.X, n_ahead=n_ahead, X_future=X_future)
        
        self.assertIn('forecast', forecast)
        self.assertEqual(forecast['forecast'].shape, (n_ahead, self.k))
        
        # Since data is random walk-like and future X is 0, forecast should not be NaN
        self.assertFalse(np.any(np.isnan(forecast['forecast'])))

    def test_varx_irf(self):
        """Test Impulse Response Function."""
        fit = varx_fit(self.Y, self.X, lags_y=2, lags_x=1, constant=True)
        n_ahead = 10
        
        irf_result = varx_irf(fit, n_ahead=n_ahead)
        
        self.assertIn('irf', irf_result)
        # Shape should be (horizon, k, k)
        self.assertEqual(irf_result['irf'].shape, (n_ahead, self.k, self.k))
        
        # Horizon 0 should be the Cholesky factor of Sigma (or Identity if not orth)
        # We just check it's not all zeros
        self.assertFalse(np.allclose(irf_result['irf'][0], 0))

    def test_varx_fevd(self):
        """Test Forecast Error Variance Decomposition."""
        fit = varx_fit(self.Y, self.X, lags_y=1, lags_x=1, constant=True)
        n_ahead = 5
        
        fevd_result = varx_fevd(fit, n_ahead=n_ahead)
        
        self.assertIn('fevd', fevd_result)
        self.assertEqual(fevd_result['fevd'].shape, (n_ahead, self.k, self.k))
        
        # Check if rows sum to 1 (100%)
        for h in range(n_ahead):
            for i in range(self.k):
                self.assertAlmostEqual(np.sum(fevd_result['fevd'][h, i, :]), 1.0, places=5)

    def test_varx_select_lag(self):
        """Test lag selection criteria."""
        result = varx_select_lag(self.Y, self.X, max_lags=3, criterion="AIC")
        
        self.assertIn('optimal_lag', result)
        self.assertIn('criteria', result)
        self.assertTrue(1 <= result['optimal_lag'] <= 3)
        self.assertEqual(len(result['criteria']), 3)

if __name__ == "__main__":
    unittest.main()