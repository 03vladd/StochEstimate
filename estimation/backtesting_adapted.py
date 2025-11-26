"""
Adaptive Pairs Trading Backtester with Discrete Daily Decay
============================================================

Location: estimation/backtester_adaptive.py

Key differences from backtester.py:
- Exit threshold decays each day based on theta (mean reversion speed)
- Max holding period is adaptive: min(3 * ln(2)/theta, 30 days)
- Entry thresholds can be validation-score adaptive (optional)
- Side-by-side comparison with fixed-threshold approach

Main principle: Discrete daily updates of exit threshold
(threshold recalculates once per day, stays constant within day)

Imports helper functions from adaptive_exit_strategy.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import adaptive exit strategy functions
from estimation.adaptive_exit_strategy import (
    calculate_daily_exit_threshold,
    get_max_holding_days,
    get_expected_reversion_window
)


@dataclass
class Trade:
    """Represents a single completed trade"""
    pair_name: str
    entry_date: int  # Index in time series
    exit_date: int
    entry_spread: float
    exit_spread: float
    signal: str  # 'BUY_A' or 'SELL_A'
    profit: float
    profit_pct: float
    days_held: int


@dataclass
class BacktestResult:
    """Complete backtest results for a pair"""
    pair_name: str
    spread: pd.Series
    theta: float
    mu: float
    sigma: float
    trades: List[Trade]
    strategy_type: str  # 'fixed' or 'adaptive'

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.profit > 0]

    @property
    def losing_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.profit < 0]

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.num_trades * 100

    @property
    def total_profit(self) -> float:
        return sum(t.profit for t in self.trades)

    @property
    def avg_profit_per_trade(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return self.total_profit / self.num_trades

    @property
    def avg_holding_period(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return np.mean([t.days_held for t in self.trades])

    @property
    def detailed_result(self) -> str:
        """Human-readable backtest summary"""
        lines = [
            f"BACKTEST RESULTS: {self.pair_name} ({self.strategy_type.upper()})",
            f"{'=' * 70}",
            f"",
            f"OU PARAMETERS (from MLE):",
            f"  θ (mean reversion speed): {self.theta:.6f} per day",
            f"  μ (long-term mean): {self.mu:.6f}",
            f"  σ (volatility): {self.sigma:.6f}",
            f"",
            f"TRADE STATISTICS:",
            f"  Total trades: {self.num_trades}",
            f"  Winning trades: {len(self.winning_trades)}",
            f"  Losing trades: {len(self.losing_trades)}",
            f"  Win rate: {self.win_rate:.1f}%",
            f"",
            f"PROFIT/LOSS:",
            f"  Total P&L: ${self.total_profit:,.2f}",
            f"  Avg profit per trade: ${self.avg_profit_per_trade:,.2f}",
            f"  Avg holding period: {self.avg_holding_period:.1f} days",
        ]

        if self.strategy_type == 'adaptive':
            lines.append(f"")
            half_life = get_expected_reversion_window(self.theta)
            max_hold = get_max_holding_days(self.theta)
            lines.append(f"ADAPTIVE PARAMETERS:")
            lines.append(f"  Half-life (50% reversion): {half_life:.2f} days")
            lines.append(f"  Max holding period: {max_hold} days")

        return "\n".join(lines)


def generate_trading_signals_fixed(
    spread: pd.Series,
    mu: float,
    sigma: float,
    entry_threshold: float = 1.5,
    max_holding_days: int = 30
) -> List[Tuple[int, str, float]]:
    """
    Original fixed-threshold signal generation.

    Entry: when deviation > entry_threshold * sigma
    Exit: when deviation < 0.5 * sigma OR max_holding_days reached
    """
    signals = []
    in_position = False
    position_entry_idx = None

    for i in range(len(spread)):
        current_spread = spread.iloc[i]
        deviation = current_spread - mu
        deviation_sigma = deviation / sigma if sigma > 0 else 0

        # Check exit conditions
        if in_position:
            days_held = i - position_entry_idx
            if abs(deviation) < 0.5 * sigma:
                signals.append((i, 'EXIT', deviation_sigma))
                in_position = False
            elif days_held >= max_holding_days:
                signals.append((i, 'EXIT', deviation_sigma))
                in_position = False

        # Check entry conditions
        if not in_position and abs(deviation_sigma) >= entry_threshold:
            if deviation > 0:
                signals.append((i, 'ENTRY_SELL_A', deviation_sigma))
            else:
                signals.append((i, 'ENTRY_BUY_A', deviation_sigma))
            in_position = True
            position_entry_idx = i

    return signals


def generate_trading_signals_adaptive(
    spread: pd.Series,
    mu: float,
    sigma: float,
    theta: float,
    entry_threshold: float = 1.5,
    exit_ratio: float = 0.5
) -> List[Tuple[int, str, float]]:
    """
    Adaptive threshold signal generation.

    Entry: same as fixed (entry_threshold * sigma)
    Exit: Adaptive daily decay based on theta
    Max hold: min(3 * ln(2)/theta, 30 days)

    The exit threshold recalculates ONCE per day and stays constant
    within that day (discrete daily updates).
    """
    signals = []
    in_position = False
    position_entry_idx = None
    position_entry_spread = None

    # Pre-calculate max holding days once
    max_holding_days = get_max_holding_days(theta)

    for i in range(len(spread)):
        current_spread = spread.iloc[i]
        deviation = current_spread - mu
        deviation_sigma = deviation / sigma if sigma > 0 else 0

        # Check exit conditions
        if in_position:
            days_held = i - position_entry_idx

            # Calculate adaptive exit threshold for this day
            adaptive_exit = calculate_daily_exit_threshold(
                entry_day=position_entry_idx,
                current_day=i,
                mu=mu,
                entry_spread=position_entry_spread,
                theta=theta,
                exit_ratio=exit_ratio
            )

            # Exit if spread crossed threshold OR max holding reached
            if current_spread <= adaptive_exit or days_held >= max_holding_days:
                signals.append((i, 'EXIT', deviation_sigma))
                in_position = False

        # Check entry conditions (same as fixed approach)
        if not in_position and abs(deviation_sigma) >= entry_threshold:
            if deviation > 0:
                signals.append((i, 'ENTRY_SELL_A', deviation_sigma))
            else:
                signals.append((i, 'ENTRY_BUY_A', deviation_sigma))
            in_position = True
            position_entry_idx = i
            position_entry_spread = current_spread

    return signals


def process_signals_into_trades(
    signals: List[Tuple[int, str, float]],
    spread: pd.Series,
    pair_name: str,
    position_size: float = 10000.0
) -> List[Trade]:
    """
    Convert signal list into completed Trade objects.
    Matches entries with exits and calculates P&L.
    """
    trades = []
    entry_idx = None
    entry_spread = None
    entry_signal = None

    for idx, signal_type, dev_sigma in signals:
        if signal_type in ['ENTRY_BUY_A', 'ENTRY_SELL_A']:
            entry_idx = idx
            entry_spread = spread.iloc[idx]
            entry_signal = signal_type

        elif signal_type == 'EXIT' and entry_idx is not None:
            exit_idx = idx
            exit_spread = spread.iloc[idx]
            days_held = exit_idx - entry_idx
            spread_change = exit_spread - entry_spread

            # Calculate profit
            if entry_signal == 'ENTRY_BUY_A':
                profit = (spread_change / abs(entry_spread)) * position_size if entry_spread != 0 else 0
                profit_pct = (spread_change / abs(entry_spread) * 100) if entry_spread != 0 else 0
            else:  # ENTRY_SELL_A
                profit = -(spread_change / abs(entry_spread)) * position_size if entry_spread != 0 else 0
                profit_pct = -(spread_change / abs(entry_spread) * 100) if entry_spread != 0 else 0

            trade = Trade(
                pair_name=pair_name,
                entry_date=entry_idx,
                exit_date=exit_idx,
                entry_spread=entry_spread,
                exit_spread=exit_spread,
                signal=entry_signal,
                profit=profit,
                profit_pct=profit_pct,
                days_held=days_held
            )
            trades.append(trade)
            entry_idx = None

    return trades


def backtest_pairs_trading_fixed(
    spread: pd.Series,
    theta: float,
    mu: float,
    sigma: float,
    pair_name: str = "Pair",
    position_size: float = 10000.0,
    entry_threshold: float = 1.5
) -> BacktestResult:
    """
    Backtest with FIXED exit thresholds (original approach).
    Exit at 0.5σ or 30 days max.
    """
    max_holding_days = 30

    signals = generate_trading_signals_fixed(
        spread=spread,
        mu=mu,
        sigma=sigma,
        entry_threshold=entry_threshold,
        max_holding_days=max_holding_days
    )

    trades = process_signals_into_trades(signals, spread, pair_name, position_size)

    return BacktestResult(
        pair_name=pair_name,
        spread=spread,
        theta=theta,
        mu=mu,
        sigma=sigma,
        trades=trades,
        strategy_type='fixed'
    )


def backtest_pairs_trading_adaptive(
    spread: pd.Series,
    theta: float,
    mu: float,
    sigma: float,
    pair_name: str = "Pair",
    position_size: float = 10000.0,
    entry_threshold: float = 1.5,
    exit_ratio: float = 0.5
) -> BacktestResult:
    """
    Backtest with ADAPTIVE exit thresholds (new approach).
    Exit threshold decays daily based on theta.
    Max holding is data-driven: min(3 * ln(2)/theta, 30 days)
    """
    signals = generate_trading_signals_adaptive(
        spread=spread,
        mu=mu,
        sigma=sigma,
        theta=theta,
        entry_threshold=entry_threshold,
        exit_ratio=exit_ratio
    )

    trades = process_signals_into_trades(signals, spread, pair_name, position_size)

    return BacktestResult(
        pair_name=pair_name,
        spread=spread,
        theta=theta,
        mu=mu,
        sigma=sigma,
        trades=trades,
        strategy_type='adaptive'
    )


def compare_strategies(
    spread: pd.Series,
    theta: float,
    mu: float,
    sigma: float,
    pair_name: str = "Pair",
    position_size: float = 10000.0
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Run both fixed and adaptive backtests on same data.
    Returns (fixed_result, adaptive_result) for comparison.
    """
    fixed = backtest_pairs_trading_fixed(
        spread=spread,
        theta=theta,
        mu=mu,
        sigma=sigma,
        pair_name=pair_name,
        position_size=position_size
    )

    adaptive = backtest_pairs_trading_adaptive(
        spread=spread,
        theta=theta,
        mu=mu,
        sigma=sigma,
        pair_name=pair_name,
        position_size=position_size
    )

    return fixed, adaptive


def print_comparison(fixed_result: BacktestResult, adaptive_result: BacktestResult):
    """
    Print side-by-side comparison of fixed vs adaptive strategies.
    """
    print()
    print("=" * 100)
    print(f"STRATEGY COMPARISON: {fixed_result.pair_name}")
    print("=" * 100)
    print()

    print(f"{'Metric':<40} {'Fixed Strategy':<25} {'Adaptive Strategy':<25}")
    print("-" * 100)

    metrics = [
        ("Total Trades", f"{fixed_result.num_trades}", f"{adaptive_result.num_trades}"),
        ("Win Rate (%)", f"{fixed_result.win_rate:.1f}%", f"{adaptive_result.win_rate:.1f}%"),
        ("Total P&L ($)", f"${fixed_result.total_profit:,.2f}", f"${adaptive_result.total_profit:,.2f}"),
        ("Avg P&L/Trade ($)", f"${fixed_result.avg_profit_per_trade:,.2f}", f"${adaptive_result.avg_profit_per_trade:,.2f}"),
        ("Avg Holding Days", f"{fixed_result.avg_holding_period:.1f}", f"{adaptive_result.avg_holding_period:.1f}"),
    ]

    for metric, fixed_val, adaptive_val in metrics:
        print(f"{metric:<40} {fixed_val:<25} {adaptive_val:<25}")

    print()
    print("Key Differences:")
    profit_diff = adaptive_result.total_profit - fixed_result.total_profit
    profit_diff_pct = (profit_diff / abs(fixed_result.total_profit) * 100) if fixed_result.total_profit != 0 else 0
    print(f"  - Adaptive P&L vs Fixed: ${profit_diff:,.2f} ({profit_diff_pct:+.1f}%)")

    trade_diff = adaptive_result.num_trades - fixed_result.num_trades
    print(f"  - Number of trades: {trade_diff:+d} ({trade_diff/max(fixed_result.num_trades, 1)*100:+.1f}%)")

    holding_diff = adaptive_result.avg_holding_period - fixed_result.avg_holding_period
    print(f"  - Avg holding period: {holding_diff:+.1f} days")

    print()
    print("=" * 100)
    print()