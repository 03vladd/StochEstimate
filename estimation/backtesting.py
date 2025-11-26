"""
Trading Signal Generator and Backtester for OU Pairs

Generates buy/sell signals based on spread deviation from mean,
then backtests trading strategy on historical data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


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

    @property
    def num_trades(self) -> int:
        """Total number of completed trades"""
        return len(self.trades)

    @property
    def winning_trades(self) -> List[Trade]:
        """Trades that made profit"""
        return [t for t in self.trades if t.profit > 0]

    @property
    def losing_trades(self) -> List[Trade]:
        """Trades that lost money"""
        return [t for t in self.trades if t.profit < 0]

    @property
    def win_rate(self) -> float:
        """Percentage of trades that were profitable"""
        if self.num_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.num_trades * 100

    @property
    def total_profit(self) -> float:
        """Sum of all trade profits"""
        return sum(t.profit for t in self.trades)

    @property
    def avg_profit_per_trade(self) -> float:
        """Average profit per trade"""
        if self.num_trades == 0:
            return 0.0
        return self.total_profit / self.num_trades

    @property
    def avg_holding_period(self) -> float:
        """Average number of days held per trade"""
        if self.num_trades == 0:
            return 0.0
        return np.mean([t.days_held for t in self.trades])

    @property
    def detailed_result(self) -> str:
        """Human-readable backtest summary"""
        lines = [
            f"BACKTEST RESULTS: {self.pair_name}",
            f"{'=' * 70}",
            f"",
            f"OU PARAMETERS (from MLE):",
            f"  θ (mean reversion speed): {self.theta:.6f} per day",
            f"  μ (long-term mean): {self.mu:.6f}",
            f"  σ (volatility): {self.sigma:.6f}",
            f"",
            f"TRADE STATISTICS:",
            f"  Total trades: {self.num_trades}",
            f"  Winning trades: {len(self.winning_trades)}"]

        lines.append(f"  Losing trades: {len(self.losing_trades)}")
        lines.append(f"  Win rate: {self.win_rate:.1f}%")
        lines.append(f"")
        lines.append(f"PROFIT/LOSS:")
        lines.append(f"  Total P&L: ${self.total_profit:,.2f}")
        lines.append(f"  Avg profit per trade: ${self.avg_profit_per_trade:,.2f}")
        lines.append(f"  Avg holding period: {self.avg_holding_period:.1f} days")
        lines.append(f"")

        if self.num_trades > 0:
            lines.append(f"TRADE DETAILS:")
            lines.append(f"{'-' * 70}")
            for i, trade in enumerate(self.trades, 1):
                lines.append(f"Trade {i}: {trade.signal}")
                lines.append(f"  Entry (day {trade.entry_date}): spread = {trade.entry_spread:.2f}")
                lines.append(f"  Exit  (day {trade.exit_date}): spread = {trade.exit_spread:.2f}")
                lines.append(f"  P&L: ${trade.profit:,.2f} ({trade.profit_pct:.2f}%)")
                lines.append(f"  Held: {trade.days_held} days")
                lines.append(f"")

        return "\n".join(lines)


def generate_trading_signals(
        spread: pd.Series,
        mu: float,
        sigma: float,
        entry_threshold: float = 1.5,
        max_holding_days: int = 30
) -> List[Tuple[int, str, float]]:
    """
    Generate buy/sell signals based on spread deviation from mean.

    Args:
        spread: pd.Series of spread values
        mu: long-term mean of spread
        sigma: volatility of spread
        entry_threshold: number of sigma for entry (default 1.5)
        max_holding_days: max days to hold a position

    Returns:
        List of (index, signal_type, deviation_sigma) tuples
        signal_type: 'ENTRY_BUY_A', 'ENTRY_SELL_A', 'EXIT'
    """
    signals = []
    in_position = False
    position_entry_idx = None
    position_type = None

    for i in range(len(spread)):
        current_spread = spread.iloc[i]
        deviation = current_spread - mu
        deviation_sigma = deviation / sigma if sigma > 0 else 0

        # Check if we should exit current position
        if in_position:
            days_held = i - position_entry_idx

            # Exit if we've reverted to mean
            if abs(deviation) < 0.5 * sigma:
                signals.append((i, 'EXIT', deviation_sigma))
                in_position = False
            # Exit if max holding period reached
            elif days_held >= max_holding_days:
                signals.append((i, 'EXIT', deviation_sigma))
                in_position = False

        # Check if we should enter new position
        if not in_position and abs(deviation_sigma) >= entry_threshold:
            if deviation > 0:
                # Spread is high: Asset A is overvalued
                signals.append((i, 'ENTRY_SELL_A', deviation_sigma))
                position_type = 'SELL_A'
            else:
                # Spread is low: Asset A is undervalued
                signals.append((i, 'ENTRY_BUY_A', deviation_sigma))
                position_type = 'BUY_A'

            in_position = True
            position_entry_idx = i

    return signals


def backtest_pairs_trading(
        spread: pd.Series,
        theta: float,
        mu: float,
        sigma: float,
        pair_name: str = "Pair",
        position_size: float = 10000.0,
        entry_threshold: float = 1.5,
        max_holding_days: int = 30
) -> BacktestResult:
    """
    Backtest pairs trading strategy on historical spread data.

    Args:
        spread: pd.Series of historical spread values
        theta: mean reversion speed (from MLE)
        mu: long-term mean (from MLE)
        sigma: volatility (from MLE)
        pair_name: name of pair being tested
        position_size: dollar amount per trade
        entry_threshold: std devs for entry signal (default 1.5)
        max_holding_days: max days to hold position

    Returns:
        BacktestResult with all trade details
    """

    # Generate signals
    signals = generate_trading_signals(spread, mu, sigma, entry_threshold, max_holding_days)

    # Process signals into trades
    trades = []
    entry_idx = None
    entry_spread = None
    entry_signal = None

    for idx, signal_type, dev_sigma in signals:
        if signal_type in ['ENTRY_BUY_A', 'ENTRY_SELL_A']:
            # Start new position
            entry_idx = idx
            entry_spread = spread.iloc[idx]
            entry_signal = signal_type

        elif signal_type == 'EXIT' and entry_idx is not None:
            # Close position and calculate P&L
            exit_idx = idx
            exit_spread = spread.iloc[idx]
            days_held = exit_idx - entry_idx

            # Calculate profit
            # For simplicity: profit = (price change) * (position size / entry price)
            # We use the position size directly as the base
            spread_change = exit_spread - entry_spread

            if entry_signal == 'ENTRY_BUY_A':
                # We're long the spread: profit if spread goes up
                profit = (spread_change / abs(entry_spread)) * position_size if entry_spread != 0 else 0
            else:  # ENTRY_SELL_A
                # We're short the spread: profit if spread goes down
                profit = -(spread_change / abs(entry_spread)) * position_size if entry_spread != 0 else 0

            profit_pct = (spread_change / abs(entry_spread) * 100) if entry_spread != 0 else 0

            trade = Trade(
                pair_name=pair_name,
                entry_date=entry_idx,
                exit_date=exit_idx,
                entry_spread=entry_spread,
                exit_spread=exit_spread,
                signal=entry_signal,
                profit=profit,
                profit_pct=profit_pct if entry_signal == 'ENTRY_BUY_A' else -profit_pct,
                days_held=days_held
            )

            trades.append(trade)
            entry_idx = None
            entry_spread = None
            entry_signal = None

    return BacktestResult(
        pair_name=pair_name,
        spread=spread,
        theta=theta,
        mu=mu,
        sigma=sigma,
        trades=trades
    )