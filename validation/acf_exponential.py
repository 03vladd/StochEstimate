"""
Autocorrelation Structure Test - Exponential Fit

Tests whether autocorrelation follows exponential decay pattern.
OU processes should have ACF(k) = exp(-θk).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.graphics.tsaplots import acf
from dataclasses import dataclass


@dataclass
class ACFExponentialFitResult:
    """Result from ACF exponential fit test"""
    passed: bool
    theta: float
    r_squared: float
    n_lags: int
    detailed_result: str


def test_autocorrelation_exponential(series: pd.Series,
                                     n_lags: int = 20,
                                     r_squared_threshold: float = 0.85,
                                     critical_level: float = 0.05) -> ACFExponentialFitResult:
    """
    Test for exponential decay in autocorrelation using exponential model fit

    We fit: ACF(k) = exp(-θ*k)

    where θ is the mean reversion speed.

    Args:
        series: pandas Series with time series data
        n_lags: Number of lags to compute ACF for
        r_squared_threshold: Minimum R² for good fit (default 0.85)
        critical_level: Not used here, for consistency with other tests

    Returns:
        ACFExponentialFitResult with test outcomes
    """
    # Compute ACF
    acf_values = acf(series, nlags=n_lags, fft=False)
    lags = np.arange(1, len(acf_values))
    observed_acf = acf_values[1:]

    # Define exponential model
    def exponential_model(k, theta):
        return np.exp(-theta * k)

    try:
        # Fit exponential model
        popt, _ = curve_fit(
            exponential_model,
            lags,
            observed_acf,
            p0=[0.1],
            bounds=([0.001], [3.0]),
            maxfev=5000
        )
        theta_fit = popt[0]

        # Compute R² as goodness of fit
        predicted_acf = exponential_model(lags, theta_fit)
        residuals = observed_acf - predicted_acf
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((observed_acf - np.mean(observed_acf)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Test passes if R² > threshold and θ in reasonable range
        passed = (r_squared > r_squared_threshold) and (0.01 < theta_fit < 2.0)

        if passed:
            detailed_result = (
                f"Theta (mean reversion speed): {theta_fit:.6f}\n"
                f"R² (goodness of fit): {r_squared:.6f}\n"
                f"Lags tested: {n_lags}\n"
                f"R² threshold: {r_squared_threshold}\n"
                f"\nConclusion: ACF follows exponential decay (consistent with OU)."
            )
        else:
            detailed_result = (
                f"Theta (mean reversion speed): {theta_fit:.6f}\n"
                f"R² (goodness of fit): {r_squared:.6f}\n"
                f"Lags tested: {n_lags}\n"
                f"R² threshold: {r_squared_threshold}\n"
                f"\nConclusion: ACF does not follow exponential decay pattern."
            )

        return ACFExponentialFitResult(
            passed=passed,
            theta=theta_fit,
            r_squared=r_squared,
            n_lags=n_lags,
            detailed_result=detailed_result
        )

    except Exception as e:
        return ACFExponentialFitResult(
            passed=False,
            theta=None,
            r_squared=None,
            n_lags=n_lags,
            detailed_result=f"Exponential fit failed: {str(e)}"
        )