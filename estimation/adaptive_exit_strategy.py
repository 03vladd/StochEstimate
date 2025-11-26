"""
Adaptive Exit Strategy Module
=============================

Location: estimation/adaptive_exit_strategy.py

Purpose: Helper functions for calculating adaptive exit thresholds
that decay daily based on mean reversion speed (theta).

These functions are imported and used by backtester_adaptive.py
"""

import numpy as np


def calculate_daily_exit_threshold(
        entry_day: int,
        current_day: int,
        mu: float,
        entry_spread: float,
        theta: float,
        exit_ratio: float = 0.5
) -> float:
    """
    Calculate exit threshold for current day (discrete daily decay).

    Called once per trading day. Uses exponential decay of mean reversion distance.

    Theory:
    - OU process predicts remaining distance shrinks as: distance(t) = distance(0) * e^(-θ*t)
    - As time passes, the spread should be closer to equilibrium
    - So we tighten our exit threshold (require less recovery) as days pass

    BUT: In practice, this made performance worse (see test_adaptive_backtest.py results)
    Keep for reference/research, but fixed thresholds work better for real data.

    Args:
        entry_day: Day index when position was entered (e.g., day 50)
        current_day: Today's day index (e.g., day 55)
        mu: Long-term mean of the spread (from MLE)
        entry_spread: The spread value when position was entered
        theta: Mean reversion speed (from MLE, units: per day)
        exit_ratio: What fraction of entry distance triggers exit (default 0.5)

    Returns:
        float: Exit threshold to use throughout today

    Example:
    --------
    entry_day = 50, current_day = 55, theta = 0.1, mu = 0.0, entry_spread = -3.5

    days_held = 55 - 50 = 5 days
    decay_factor = e^(-0.1 * 5) = 0.606
    entry_distance = -3.5 - 0.0 = -3.5
    remaining_distance = -3.5 * 0.606 = -2.121
    exit_threshold = 0.0 + (-2.121 * 0.5) = -1.061

    Interpretation: We entered at -3.5 (oversold). After 5 days, expect
    remaining oversold distance of ~2.1. Exit when spread recovers to -1.06
    (50% of remaining).
    """

    days_held = current_day - entry_day
    decay_factor = np.exp(-theta * days_held)
    entry_distance = entry_spread - mu
    remaining_distance = entry_distance * decay_factor
    exit_threshold = mu + (remaining_distance * exit_ratio)

    return exit_threshold


def get_expected_reversion_window(theta: float) -> float:
    """
    Calculate the half-life of mean reversion (time to 50% reversion).

    For an OU process, the time for 50% of the initial deviation to be
    recovered is called the "half-life."

    Math: e^(-θ*t) = 0.5 → t = ln(2) / θ

    Args:
        theta: Mean reversion speed (per day)

    Returns:
        float: Half-life in days

    Example:
    --------
    theta = 0.05 per day
    half_life = ln(2) / 0.05 = 13.86 days

    Interpretation: After ~14 days, expect 50% of initial misprice to be recovered.
    """
    if theta <= 0:
        return np.inf

    half_life = np.log(2) / theta
    return half_life


def get_max_holding_days(theta: float, hard_cap: int = 30) -> int:
    """
    Determine maximum days to hold a position based on mean reversion speed.

    After 3 half-lives, ~87.5% of the deviation has been recovered.
    Beyond that, returns diminish significantly.

    Formula: max_hold = min(3 * ln(2)/θ, hard_cap)

    This makes max_hold adaptive:
    - Fast reversion pairs (high θ): Lower max_hold (e.g., 10 days)
    - Slow reversion pairs (low θ): Higher max_hold (e.g., 30 days capped)

    Args:
        theta: Mean reversion speed (per day)
        hard_cap: Absolute maximum (default 30 days)

    Returns:
        int: Maximum days to hold before forcibly closing

    Example:
    --------
    Fast pair: θ = 0.2/day
      half_life = ln(2)/0.2 = 3.47 days
      max_hold = min(3*3.47, 30) = 10 days

    Slow pair: θ = 0.02/day
      half_life = ln(2)/0.02 = 34.66 days
      max_hold = min(3*34.66, 30) = 30 days (capped)
    """
    if theta <= 0:
        return hard_cap

    half_life = np.log(2) / theta
    max_hold = min(int(3 * half_life), hard_cap)
    return max(max_hold, 1)  # At least 1 day


if __name__ == "__main__":
    """
    Demo showing how these functions work.
    Run this to understand the behavior.
    """

    print("=" * 80)
    print("ADAPTIVE EXIT STRATEGY: FUNCTION DEMONSTRATIONS")
    print("=" * 80)
    print()

    # Example 1: Calculate daily decay
    print("Example 1: Daily Exit Threshold Decay")
    print("-" * 80)
    theta = 0.0404  # Your JPM-GS pair
    mu = 0.0
    entry_spread = -3.5
    entry_day = 50

    print(f"Parameters: θ={theta:.4f}, μ={mu:.1f}, entry_spread={entry_spread:.2f}")
    print()
    print("Exit thresholds over time:")
    print(f"{'Day':<5} {'Days Held':<12} {'Exit Threshold':<15} {'Change':<15}")
    print("-" * 80)

    prev_threshold = None
    for current_day in range(entry_day, entry_day + 11):
        threshold = calculate_daily_exit_threshold(
            entry_day=entry_day,
            current_day=current_day,
            mu=mu,
            entry_spread=entry_spread,
            theta=theta,
            exit_ratio=0.5
        )

        change = ""
        if prev_threshold is not None:
            diff = threshold - prev_threshold
            change = f"{diff:+.4f}"

        days_held = current_day - entry_day
        print(f"{current_day:<5} {days_held:<12} {threshold:<15.4f} {change:<15}")
        prev_threshold = threshold

    print()
    print("Observation: Thresholds get tighter (move toward μ) each day")
    print()

    # Example 2: Reversion window
    print("Example 2: Expected Reversion Windows")
    print("-" * 80)

    thetas = [0.02, 0.05, 0.10, 0.15, 0.20]
    print(f"{'θ (speed)':<15} {'Half-life (days)':<20} {'3 Half-lives':<20}")
    print("-" * 80)

    for t in thetas:
        half = get_expected_reversion_window(t)
        three_half = 3 * half
        print(f"{t:<15.4f} {half:<20.2f} {three_half:<20.2f}")

    print()
    print("Observation: Slower reversion (lower θ) = longer half-life")
    print()

    # Example 3: Max holding days
    print("Example 3: Adaptive Max Holding Days")
    print("-" * 80)
    print(f"{'θ (speed)':<15} {'Max Hold Days':<20} {'Interpretation':<40}")
    print("-" * 80)

    for t in thetas:
        max_days = get_max_holding_days(t)
        interp = f"Wait up to {max_days} days"
        print(f"{t:<15.4f} {max_days:<20} {interp:<40}")

    print()
    print("=" * 80)