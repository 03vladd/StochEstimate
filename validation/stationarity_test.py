import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass

@dataclass
class StationarityTestResult:
    passed: bool
    adf_statistic: float
    p_value: float
    critical_value_5pct: float #around -2.86 for a 5% significance level
    n_observations: int
    detailed_result: str


# We use augmented Dickey-Fuller test to check for stationarity; if p-value < critical_level, we reject H0 = series has unit root (non-stationary).

def test_stationarity_adf(series: pd.Series, critical_level: float = 0.05) -> StationarityTestResult:
    result = adfuller(series, autolag='AIC') # AIC = Akaike Information Criterion (choose lag that explains data best, without overfitting)
    adf_statistic = result[0]
    p_value = result[1]
    n_lags_used = result[2]
    n_observations = result[3]
    critical_value_5pct = result[4]['5%']


    passed = p_value < critical_level

    if passed:
        detailed_result = (
            f"ADF statistic: {adf_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (< {critical_level})\n"
            f"Critical value (5%): {critical_value_5pct:.6f}\n"
            f"Lags used: {n_lags_used}\n"
            f"Observations: {n_observations}\n"
            f"\nConclusion: Series is likely stationary."
        )
    else:
        detailed_result = (
            f"ADF statistic: {adf_statistic:.6f}\n"
            f"P-value: {p_value:.6f} (>= {critical_level})\n"
            f"Critical value (5%): {critical_value_5pct:.6f}\n"
            f"Lags used: {n_lags_used}\n"
            f"Observations: {n_observations}\n"
            f"\nConclusion: Series is likely non-stationary."
        )
    return StationarityTestResult(
        passed=passed,
        adf_statistic=adf_statistic,
        p_value=p_value,
        critical_value_5pct=critical_value_5pct,
        n_observations=n_observations,
        detailed_result=detailed_result
    )

