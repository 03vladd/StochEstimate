import numpy as np
import pandas as pd
from scipy.stats import levene
from dataclasses import dataclass


@dataclass
class ConstantVolatilityTestResult:
    passed: bool
    volatility1: float
    volatility2: float
    volatility_ratio: float
    levene_statistic: float
    p_value: float
    detailed_result: str

# Test for constant volatility by splitting the series into two halves and comparing their variances using Levene's test.

def test_constant_volatility(series: pd.Series, critical_level: float = 0.05) -> ConstantVolatilityTestResult:
    # Split series into two halves
    mid = len(series) // 2
    first_half = series.iloc[:mid].values
    second_half = series.iloc[mid:].values

    # Compute volatility (standard deviation) for each half
    vol_first = np.std(first_half)
    vol_second = np.std(second_half)

    # Compute ratio (larger / smaller for easy interpretation)
    vol_ratio = max(vol_first, vol_second) / min(vol_first, vol_second)

    # Levene's test for equality of variances
    # More robust than F-test for non-normal distributions
    levene_statistic, p_value = levene(first_half, second_half)

    # Test passes if p-value > critical_level (fail to reject H0: equal variances)
    passed = p_value > critical_level

    if passed:
        detailed_result = (
            f"First half volatility (σ): {vol_first:.6f}\n"
            f"Second half volatility (σ): {vol_second:.6f}\n"
            f"Volatility ratio: {vol_ratio:.4f}\n"
            f"Levene statistic: {levene_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (> {critical_level})\n"
            f"\nConclusion: Volatility appears constant over time."
        )
    else:
        detailed_result = (
            f"First half volatility (σ): {vol_first:.6f}\n"
            f"Second half volatility (σ): {vol_second:.6f}\n"
            f"Volatility ratio: {vol_ratio:.4f}\n"
            f"Levene statistic: {levene_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (≤ {critical_level})\n"
            f"\nConclusion: Volatility appears to vary over time."
        )

    return ConstantVolatilityTestResult(
        passed=passed,
        volatility1=vol_first,
        volatility2=vol_second,
        volatility_ratio=vol_ratio,
        levene_statistic=levene_statistic,
        p_value=p_value,
        detailed_result=detailed_result
    )