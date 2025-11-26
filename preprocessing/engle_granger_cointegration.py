import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Perform Engle-Granger two-step cointegration test between two price series
def engle_granger_cointegration(price_a, price_b, significance_level=0.05):

    # Step 1: OLS Regression
    # We want to find: price_a = intercept + hedge_ratio * price_b + residual
    # Using numpy for simple linear regression
    X = np.column_stack([np.ones(len(price_b)), price_b])  # Add intercept column
    coefficients = np.linalg.lstsq(X, price_a, rcond=None)[0]
    intercept, hedge_ratio = coefficients

    # Step 2: Extract the spread (residuals)
    spread = price_a - (intercept + hedge_ratio * price_b)

    # Step 3: Test if spread is stationary using ADF test
    adf_result = adfuller(spread, autolag='AIC')
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]

    # Determine cointegration
    cointegrated = adf_pvalue < significance_level

    return {
        'cointegrated': cointegrated,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'adf_pvalue': adf_pvalue,
        'adf_statistic': adf_statistic,
        'intercept': intercept,
        'significance_level': significance_level
    }