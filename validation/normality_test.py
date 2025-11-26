import numpy as np
import pandas as pd
from scipy.stats import shapiro
from dataclasses import dataclass


@dataclass
class NormalityTestResult:
    """Result from normality test"""
    passed: bool
    shapiro_statistic: float
    p_value: float
    n_observations: int
    skewness: float
    kurtosis: float
    detailed_result: str


# Test for normality using Shapiro-Wilk test; if p-value < critical_level, we reject H0 = data is normally distributed.

def test_normality(series: pd.Series, critical_level: float = 0.05) -> NormalityTestResult:
    # Compute first differences (returns)
    returns = series.diff().dropna()

    # Shapiro-Wilk test
    shapiro_statistic, p_value = shapiro(returns)

    n_observations = len(returns)

    # Compute skewness and kurtosis for additional context
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Test passes if p-value > critical_level (fail to reject H0: normal)
    passed = p_value > critical_level

    if passed:
        detailed_result = (
            f"Shapiro-Wilk statistic: {shapiro_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (> {critical_level})\n"
            f"Observations: {n_observations}\n"
            f"Skewness: {skewness:.6f}\n"
            f"Kurtosis (excess): {kurtosis:.6f}\n"
            f"\nConclusion: Returns appear normally distributed."
        )
    else:
        detailed_result = (
            f"Shapiro-Wilk statistic: {shapiro_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (≤ {critical_level})\n"
            f"Observations: {n_observations}\n"
            f"Skewness: {skewness:.6f}\n"
            f"Kurtosis (excess): {kurtosis:.6f}\n"
            f"\nConclusion: Returns deviate from normality."
        )

    return NormalityTestResult(
        passed=passed,
        shapiro_statistic=shapiro_statistic,
        p_value=p_value,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        detailed_result=detailed_result
    )