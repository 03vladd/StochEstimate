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