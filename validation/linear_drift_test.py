import numpy as np
import pandas as pd
from scipy.stats import linregress
from dataclasses import dataclass

@dataclass
class LinearDriftTestResult:
    passed: bool
    slope: float
    intercept: float
    p_value: float
    slope_std_error: float
    r_squared: float
    detailed_result: str


# Test for linear drift, checking if the slope is significantly different from zero at any point in time.

def test_linear_drift(series: pd.Series, critical_level: float = 0.05) -> LinearDriftTestResult:

    # Create time variable: t = 0, 1, 2, ..., n-1
    n = len(series)
    t = np.arange(n)

    # Fit linear regression: series ~ intercept + slope*t
    slope, intercept, r_value, p_value, std_err = linregress(t, series.values)

    r_squared = r_value ** 2

    # Test passes if slope is NOT significantly different from zero
    # i.e., if p-value > critical_level (fail to reject H0: slope=0)
    passed = p_value > critical_level

    if passed:
        detailed_result = (
            f"Slope: {slope:.6f}\n"
            f"Intercept: {intercept:.6f}\n"
            f"P-value: {p_value:.6f} (> {critical_level})\n"
            f"Slope std error: {std_err:.6f}\n"
            f"R²: {r_squared:.6f}\n"
            f"\nConclusion: No significant linear trend detected."
        )
    else:
        detailed_result = (
            f"Slope: {slope:.6f}\n"
            f"Intercept: {intercept:.6f}\n"
            f"P-value: {p_value:.6f} (≤ {critical_level})\n"
            f"Slope std error: {std_err:.6f}\n"
            f"R²: {r_squared:.6f}\n"
            f"\nConclusion: Significant linear trend detected."
        )

    return LinearDriftTestResult(
        passed=passed,
        slope=slope,
        intercept=intercept,
        p_value=p_value,
        slope_std_error=std_err,
        r_squared=r_squared,
        detailed_result=detailed_result
    )