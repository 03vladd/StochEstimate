"""
OU Process Generator

Utility module for generating synthetic Ornstein-Uhlenbeck processes
for testing and validation purposes.
"""

import numpy as np
import pandas as pd


def generate_ou_process(
        mu: float,
        theta: float,
        sigma: float,
        n_steps: int = 1000,
        dt: float = 1.0,
        initial_value: float = 0.0,
        seed: int = None
) -> pd.Series:
    """
    Generate an Ornstein-Uhlenbeck process

    dX(t) = θ(μ - X(t))dt + σ dW(t)

    Args:
        mu: Long-term mean
        theta: Mean reversion speed (higher = faster reversion)
        sigma: Volatility
        n_steps: Number of time steps
        dt: Time step size
        initial_value: Starting value X(0)
        seed: Random seed for reproducibility

    Returns:
        pandas Series with OU process values
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros(n_steps)
    X[0] = initial_value

    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i - 1] + theta * (mu - X[i - 1]) * dt + sigma * dW

    return pd.Series(X, index=pd.date_range(start='2020-01-01', periods=n_steps, freq='D'))


def add_jump_contamination(
        series: pd.Series,
        jump_rate: float,
        jump_scale: float = 5.0,
        seed: int = None
) -> pd.Series:
    """
    Contaminate an OU series with jump outliers.

    Replaces a random fraction of observations with spikes drawn from
    N(0, jump_scale * sigma), where sigma is estimated from the series.
    This tests estimator robustness to heavy-tailed / non-Gaussian noise.

    Args:
        series:     Input OU time series
        jump_rate:  Fraction of points to replace with jumps (e.g., 0.05 = 5%)
        jump_scale: Jump size relative to series std dev (default: 5.0)
        seed:       Random seed for reproducibility

    Returns:
        Contaminated pd.Series (same index as input)
    """
    rng = np.random.default_rng(seed)
    values = series.values.copy().astype(float)
    n = len(values)

    sigma_est = np.std(values)
    n_jumps = int(np.round(jump_rate * n))

    if n_jumps > 0:
        jump_indices = rng.choice(n, size=n_jumps, replace=False)
        jump_magnitudes = rng.normal(0, jump_scale * sigma_est, size=n_jumps)
        values[jump_indices] = jump_magnitudes

    return pd.Series(values, index=series.index)


def generate_random_walk(
        n_steps: int = 1000,
        drift: float = 0.0,
        volatility: float = 1.0,
        seed: int = None
) -> pd.Series:
    """
    Generate a random walk (non-stationary process)

    X(t) = X(t-1) + drift + σ*ε

    Args:
        n_steps: Number of time steps
        drift: Drift term (trend)
        volatility: Volatility of increments
        seed: Random seed for reproducibility

    Returns:
        pandas Series with random walk values
    """
    if seed is not None:
        np.random.seed(seed)

    increments = np.random.normal(drift, volatility, n_steps)
    values = np.cumsum(increments)

    return pd.Series(values, index=pd.date_range(start='2020-01-01', periods=n_steps, freq='D'))