"""
Hybrid Pipeline: Brute Force + Full Analysis

Phase 1: Tests many pairs (42), identifies cointegrated ones
Phase 2: Runs complete analysis (Validation → MLE → Backtesting) on cointegrated pairs
Phase 3: Stores all results and generates comprehensive report

This combines the best of both approaches:
- Broad exploration to find OU-like pairs
- Deep analysis on promising pairs
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from preprocessing.engle_granger_cointegration import engle_granger_cointegration
from validation.validation_framework import validate_series
from estimation.mle import estimate_ou_mle
from estimation.backtesting import backtest_pairs_trading
from preprocessing.validate_real_pairs import FinancialPairsFetcher
from database.db_manager import DatabaseManager
from estimation.lstm_estimator import OULSTMEstimator


def extract_validation_data(report) -> dict:
    """Extract validation statistics for database storage"""
    acf = report.autocorrelation
    # r_squared and theta are None when curve_fit fails; guard against float(None) crash
    acf_r_squared = float(acf.r_squared) if acf.r_squared is not None else 0.0
    acf_theta = float(acf.theta) if acf.theta is not None else 0.0

    return {
        'stationarity_passed': bool(report.stationarity.passed),
        'stationarity_pvalue': float(report.stationarity.p_value),
        'stationarity_statistic': float(report.stationarity.adf_statistic),

        'drift_passed': bool(report.linear_drift.passed),
        'drift_pvalue': float(report.linear_drift.p_value),
        'drift_statistic': float(report.linear_drift.slope),

        'volatility_passed': bool(report.constant_volatility.passed),
        'volatility_pvalue': float(report.constant_volatility.p_value),
        'volatility_statistic': float(report.constant_volatility.levene_statistic),

        'autocorr_passed': bool(acf.passed),
        'autocorr_pvalue': acf_r_squared,   # Stores R² (goodness-of-fit), not a p-value
        'autocorr_statistic': acf_theta,    # Stores theta fitted from ACF exponential

        'normality_passed': bool(report.normality.passed),
        'normality_pvalue': float(report.normality.p_value),
        'normality_statistic': float(report.normality.shapiro_statistic),

        'confidence_level': str(report.get_confidence_level()[0]),
        'tests_passed_count': int(report.count_passing_tests())
    }


class HybridPipeline:
    """Brute force discovery + full analysis on winners"""

    def __init__(self, start_date: str = None, end_date: str = None,
                 lstm_model_path: str = None):
        """Initialize pipeline"""
        self.fetcher = FinancialPairsFetcher(start_date, end_date)
        self.db = DatabaseManager()
        self.discovery_results = {}  # Pairs that cointegrate (Phase 1)
        self.analysis_results = {}   # Full analysis on winners (Phase 2)
        self.cointegrated_pairs = [] # List of (ticker1, ticker2, name) for Phase 2

        # Optional LSTM estimator for comparison with MLE
        self.lstm_estimator = None
        if lstm_model_path is not None:
            try:
                self.lstm_estimator = OULSTMEstimator()
                self.lstm_estimator.load(lstm_model_path)
                print(f"✓ LSTM estimator loaded from {lstm_model_path}")
            except Exception as e:
                print(f"⚠ Could not load LSTM model: {e}")
                self.lstm_estimator = None

    def check_if_pair_cached(self, ticker1: str, ticker2: str) -> Optional[Dict]:
        """Check if pair already exists in database"""
        query = """
            SELECT p.pair_id, p.pair_name, COUNT(pd.price_id) as price_count
            FROM pairs p
            LEFT JOIN price_data pd ON p.pair_id = pd.pair_id AND pd.interval = %s
            WHERE p.ticker1 = %s AND p.ticker2 = %s
            GROUP BY p.pair_id, p.pair_name
        """
        results = self.db.execute_query(query, ('1d', ticker1, ticker2))
        return results[0] if results else None

    def phase_1_discovery(self, pairs_to_test: list, skip_cached: bool = True) -> list:
        """
        Phase 1: Brute force discovery
        Test many pairs, identify cointegrated ones

        Returns:
            List of cointegrated pairs (ticker1, ticker2, name)
        """

        print("\n" + "=" * 100)
        print("PHASE 1: BRUTE FORCE DISCOVERY")
        print("Testing many pairs to identify cointegrated ones")
        print("=" * 100)

        cointegrated = []
        total_pairs = len(pairs_to_test)
        skipped_count = 0
        failed_count = 0

        for pair_idx, pair in enumerate(pairs_to_test, 1):
            ticker1, ticker2, name = pair

            # Check if already cached
            if skip_cached:
                cached = self.check_if_pair_cached(ticker1, ticker2)
                if cached:
                    print(f"[{pair_idx}/{total_pairs}] {name:40s} ✓ SKIPPED (cached)")
                    skipped_count += 1
                    continue

            print(f"[{pair_idx}/{total_pairs}] {name:40s}", end=" ", flush=True)

            try:
                # Fetch and test cointegration
                spread, pair_name, coint_result, raw = self.fetcher.fetch_pair_spread(ticker1, ticker2, name)

                if spread is None:
                    print("✗")
                    failed_count += 1
                    continue

                # Validate to get confidence level
                validation_report = validate_series(spread, name=name)
                confidence_level, _ = validation_report.get_confidence_level()

                # Store in database
                pair_id = self.db.insert_pair(
                    ticker1=ticker1, ticker2=ticker2, pair_name=name, sector=None
                )

                if pair_id is None:
                    print("✗ DB INSERT FAILED")
                    failed_count += 1
                    continue

                # Store cointegration
                coint_id = self.db.insert_cointegration_result(
                    pair_id=pair_id, interval='1d',
                    hedge_ratio=float(coint_result['hedge_ratio']),
                    intercept=float(coint_result['intercept']),
                    adf_statistic=float(coint_result['adf_statistic']),
                    adf_pvalue=float(coint_result['adf_pvalue']),
                    cointegrated=bool(coint_result['cointegrated']),
                    observations_used=int(validation_report.n_observations)
                )

                if coint_id is None:
                    print("✗ COINT STORE FAILED")
                    failed_count += 1
                    continue

                # Store validation
                val_data = extract_validation_data(validation_report)
                val_id = self.db.insert_validation_result(
                    pair_id=pair_id, interval='1d', coint_id=coint_id,
                    stationarity_passed=val_data['stationarity_passed'],
                    stationarity_pvalue=val_data['stationarity_pvalue'],
                    stationarity_statistic=val_data['stationarity_statistic'],
                    drift_passed=val_data['drift_passed'],
                    drift_pvalue=val_data['drift_pvalue'],
                    drift_statistic=val_data['drift_statistic'],
                    volatility_passed=val_data['volatility_passed'],
                    volatility_pvalue=val_data['volatility_pvalue'],
                    volatility_statistic=val_data['volatility_statistic'],
                    autocorr_passed=val_data['autocorr_passed'],
                    autocorr_pvalue=val_data['autocorr_pvalue'],
                    autocorr_statistic=val_data['autocorr_statistic'],
                    normality_passed=val_data['normality_passed'],
                    normality_pvalue=val_data['normality_pvalue'],
                    normality_statistic=val_data['normality_statistic'],
                    confidence_level=val_data['confidence_level'],
                    tests_passed_count=val_data['tests_passed_count']
                )

                if val_id is None:
                    print("✗ VAL STORE FAILED")
                    failed_count += 1
                    continue

                # Store price data
                for timestamp, close_price in spread.items():
                    self.db.insert_price_data(
                        pair_id=pair_id, interval='1d',
                        timestamp=str(timestamp), close=float(close_price)
                    )

                # Track for Phase 2
                if confidence_level in ['HIGH', 'MEDIUM']:
                    cointegrated.append((ticker1, ticker2, name))
                    self.discovery_results[name] = {
                        'cointegration': coint_result,
                        'validation': validation_report,
                        'pair_id': pair_id,
                        'confidence': confidence_level
                    }
                    print(f"✓ COINTEGRATED ({confidence_level})")
                else:
                    print(f"✗ NOT OU-LIKE ({confidence_level})")
                    failed_count += 1

            except Exception as e:
                print(f"✗ EXCEPTION: {str(e)[:40]}")
                failed_count += 1

        print(f"\n{'=' * 100}")
        print(f"PHASE 1 SUMMARY: {len(cointegrated)} cointegrated pairs found")
        print(f"{'=' * 100}\n")

        self.cointegrated_pairs = cointegrated
        return cointegrated

    def phase_2_full_analysis(self):
        """
        Phase 2: Full analysis on cointegrated pairs
        Run: Validation (detailed) → MLE → Backtesting
        """

        if not self.cointegrated_pairs:
            print("No cointegrated pairs to analyze in Phase 2")
            return

        print("\n" + "=" * 100)
        print("PHASE 2: FULL ANALYSIS ON COINTEGRATED PAIRS")
        print("Running: Validation → MLE Estimation → Backtesting")
        print("=" * 100)

        for ticker1, ticker2, name in self.cointegrated_pairs:
            print(f"\n{'=' * 100}")
            print(f"DETAILED ANALYSIS: {name}")
            print(f"{'=' * 100}")

            try:
                # Fetch data again (it's cached in DB)
                spread, pair_name, coint_result, raw = self.fetcher.fetch_pair_spread(ticker1, ticker2, name)

                if spread is None:
                    print("✗ Failed to fetch data")
                    continue

                # A. Detailed Validation (on full spread)
                print(f"\nA. VALIDATION")
                print(f"   Hedge ratio (β): {coint_result['hedge_ratio']:.6f}")
                print(f"   Intercept (α): {coint_result['intercept']:.6f}")
                print(f"   Spread formula: Price_A - {coint_result['hedge_ratio']:.6f} × Price_B")

                validation_report = validate_series(spread, name=name)
                print(validation_report.summary())

                # B. MLE Estimation on training set (first 70%) to avoid look-ahead bias.
                # Using full-sample parameters to trade historically means the strategy
                # "knows" the true equilibrium before it could have been observed.
                n_total = len(spread)
                split_idx = int(n_total * 0.7)
                estimation_spread = spread.iloc[:split_idx]
                backtest_spread = spread.iloc[split_idx:]

                print(f"\nB. MLE PARAMETER ESTIMATION")
                print(f"   Train/test split: {split_idx} obs for estimation, "
                      f"{n_total - split_idx} obs held out for backtesting")

                mle_result = estimate_ou_mle(estimation_spread, verbose=True)
                print(mle_result.detailed_result)

                # B2. LSTM Estimation (if model loaded)
                lstm_result = None
                if self.lstm_estimator is not None:
                    print(f"\nB2. LSTM PARAMETER ESTIMATION")
                    try:
                        lstm_result = self.lstm_estimator.estimate(estimation_spread, n_mc_samples=200)
                        print(lstm_result.detailed_result)
                        self._print_estimation_comparison(name, mle_result, lstm_result)
                    except Exception as e:
                        print(f"   ⚠ LSTM estimation failed: {e}")
                        lstm_result = None

                # C. Backtesting on held-out test set (last 30%) only
                print(f"\nC. BACKTESTING")
                print(f"   Running out-of-sample backtest on {n_total - split_idx} observations...")

                backtest_result = backtest_pairs_trading(
                    spread=backtest_spread,
                    theta=mle_result.theta,
                    mu=mle_result.mu,
                    sigma=mle_result.sigma,
                    pair_name=name,
                    entry_threshold=1.5,
                    max_holding_days=30
                )
                print(backtest_result.detailed_result)

                # Store full results
                self.analysis_results[name] = {
                    'cointegration': coint_result,
                    'validation': validation_report,
                    'mle': mle_result,
                    'lstm': lstm_result,
                    'backtest': backtest_result,
                    'raw_data': raw,
                    'spread': spread
                }

            except Exception as e:
                print(f"✗ Analysis failed: {str(e)}")
                continue

    def _print_estimation_comparison(self, pair_name: str, mle_result, lstm_result):
        """Print side-by-side comparison table of MLE vs LSTM estimates"""
        sep = "\u2500" * 100
        mle_half = f"{1 / mle_result.theta:.1f}" if mle_result.theta > 0 else "N/A"
        lstm_half = f"{0.693147 / lstm_result.theta:.1f}" if lstm_result.theta > 0 else "N/A"

        print(f"\nPARAMETER ESTIMATION COMPARISON: {pair_name}")
        print(sep)
        header = (
            f"{'Parameter':<14}  {'MLE Estimate':>14}  {'MLE 95% CI':>22}"
            f"  {'LSTM Estimate':>14}  {'LSTM 95% CI':>22}  {'Δ (abs)':>8}"
        )
        print(header)
        print(sep)

        def row(label, mle_val, mle_ci, lstm_val, lstm_ci):
            ci_mle = f"[{mle_ci[0]:.6f}, {mle_ci[1]:.6f}]"
            ci_lstm = f"[{lstm_ci[0]:.6f}, {lstm_ci[1]:.6f}]"
            delta = abs(mle_val - lstm_val)
            return (
                f"{label:<14}  {mle_val:>14.6f}  {ci_mle:>22}"
                f"  {lstm_val:>14.6f}  {ci_lstm:>22}  {delta:>8.6f}"
            )

        print(row("θ (speed)", mle_result.theta, mle_result.theta_ci,
                  lstm_result.theta, lstm_result.theta_ci))
        print(row("μ (mean)", mle_result.mu, mle_result.mu_ci,
                  lstm_result.mu, lstm_result.mu_ci))
        print(row("σ (vol)", mle_result.sigma, mle_result.sigma_ci,
                  lstm_result.sigma, lstm_result.sigma_ci))
        print(f"{'Half-life':<14}  {mle_half:>14} days"
              f"{'':>22}  {lstm_half:>14} days")
        print(sep)

    def create_summary_table(self) -> pd.DataFrame:
        """Create comprehensive summary table of all analyzed pairs"""
        rows = []

        for name, result_dict in self.analysis_results.items():
            coint = result_dict['cointegration']
            val = result_dict['validation']
            mle = result_dict['mle']
            bt = result_dict['backtest']
            lstm = result_dict.get('lstm')

            confidence_level, _ = val.get_confidence_level()

            row = {
                'Pair': name,
                'Cointegrated': '✓' if coint['cointegrated'] else '✗',
                'Hedge Ratio (β)': f"{coint['hedge_ratio']:.4f}",
                'Observations': val.n_observations,
                'Validation Tests': f"{val.count_passing_tests()}/4",
                'Confidence': confidence_level,
                # MLE columns
                'MLE θ': f"{mle.theta:.6f}",
                'MLE θ CI': f"[{mle.theta_ci[0]:.6f}, {mle.theta_ci[1]:.6f}]",
                'MLE μ': f"{mle.mu:.6f}",
                'MLE σ': f"{mle.sigma:.6f}",
                'Log-Likelihood': f"{mle.log_likelihood:.2f}",
                # LSTM columns (populated only if model was loaded)
                'LSTM θ': f"{lstm.theta:.6f}" if lstm else "N/A",
                'LSTM θ CI': f"[{lstm.theta_ci[0]:.6f}, {lstm.theta_ci[1]:.6f}]" if lstm else "N/A",
                'LSTM μ': f"{lstm.mu:.6f}" if lstm else "N/A",
                'LSTM σ': f"{lstm.sigma:.6f}" if lstm else "N/A",
                'Δ θ': f"{abs(mle.theta - lstm.theta):.6f}" if lstm else "N/A",
                'Δ σ': f"{abs(mle.sigma - lstm.sigma):.6f}" if lstm else "N/A",
                # Backtest columns
                'Num Trades': bt.num_trades,
                'Win Rate': f"{bt.win_rate:.1f}%",
                'Total P&L': f"${bt.total_profit:,.2f}",
                'Avg Trade P&L': f"${bt.avg_profit_per_trade:,.2f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def print_phase_2_summary(self):
        """Print Phase 2 detailed summary"""
        if not self.analysis_results:
            print("No Phase 2 results to summarize")
            return

        print("\n" + "=" * 100)
        print("PHASE 2: DETAILED RESULTS SUMMARY")
        print("=" * 100)

        summary_df = self.create_summary_table()
        print(summary_df.to_string(index=False))

        # Parameter statistics
        print("\n" + "-" * 100)
        print("PARAMETER STATISTICS (from MLE):")
        print("-" * 100)

        thetas = [r['mle'].theta for r in self.analysis_results.values()]
        sigmas = [r['mle'].sigma for r in self.analysis_results.values()]

        if thetas:
            print(f"\nMean Reversion Speed (θ):")
            print(f"  Mean:   {np.mean(thetas):.6f}")
            print(f"  Median: {np.median(thetas):.6f}")
            print(f"  Range:  [{np.min(thetas):.6f}, {np.max(thetas):.6f}]")

            print(f"\nVolatility (σ):")
            print(f"  Mean:   {np.mean(sigmas):.6f}")
            print(f"  Median: {np.median(sigmas):.6f}")
            print(f"  Range:  [{np.min(sigmas):.6f}, {np.max(sigmas):.6f}]")

        # Backtesting aggregate
        print("\n" + "-" * 100)
        print("AGGREGATE BACKTESTING STATISTICS:")
        print("-" * 100)

        total_trades = sum(r['backtest'].num_trades for r in self.analysis_results.values())
        total_pnl = sum(r['backtest'].total_profit for r in self.analysis_results.values())
        win_count = sum(len(r['backtest'].winning_trades) for r in self.analysis_results.values())

        print(f"  Total trades: {total_trades}")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        if total_trades > 0:
            print(f"  Overall win rate: {(win_count / total_trades) * 100:.1f}%")

    def save_results(self, output_dir: str = "."):
        """Save all results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Phase 2 summary
        if self.analysis_results:
            summary_df = self.create_summary_table()
            summary_file = f"{output_dir}/hybrid_phase2_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Phase 2 summary saved to {summary_file}")

            # Detailed Phase 2 results
            detailed_rows = []
            for name, result_dict in self.analysis_results.items():
                mle = result_dict['mle']
                val = result_dict['validation']
                coint = result_dict['cointegration']
                bt = result_dict['backtest']

                lstm = result_dict.get('lstm')
                detailed_rows.append({
                    'Pair': name,
                    'Hedge_Ratio': coint['hedge_ratio'],
                    'Validation_Tests': val.count_passing_tests(),
                    'Confidence': val.get_confidence_level()[0],
                    # MLE
                    'MLE_Theta': mle.theta,
                    'MLE_Theta_CI_Lower': mle.theta_ci[0],
                    'MLE_Theta_CI_Upper': mle.theta_ci[1],
                    'MLE_Mu': mle.mu,
                    'MLE_Sigma': mle.sigma,
                    'Log_Likelihood': mle.log_likelihood,
                    # LSTM
                    'LSTM_Theta': lstm.theta if lstm else None,
                    'LSTM_Theta_CI_Lower': lstm.theta_ci[0] if lstm else None,
                    'LSTM_Theta_CI_Upper': lstm.theta_ci[1] if lstm else None,
                    'LSTM_Mu': lstm.mu if lstm else None,
                    'LSTM_Sigma': lstm.sigma if lstm else None,
                    'Delta_Theta': abs(mle.theta - lstm.theta) if lstm else None,
                    'Delta_Sigma': abs(mle.sigma - lstm.sigma) if lstm else None,
                    # Backtest
                    'Backtest_Num_Trades': bt.num_trades,
                    'Backtest_Win_Rate': bt.win_rate,
                    'Backtest_Total_PnL': bt.total_profit,
                    'Backtest_Avg_Trade_PnL': bt.avg_profit_per_trade,
                })

            detailed_df = pd.DataFrame(detailed_rows)
            detailed_file = f"{output_dir}/hybrid_phase2_detailed_{timestamp}.csv"
            detailed_df.to_csv(detailed_file, index=False)
            print(f"✓ Phase 2 detailed results saved to {detailed_file}")


def main():
    """Main entry point"""

    # Initialize pipeline
    pipeline = HybridPipeline()

    # Phase 1: Define pairs to test (brute force discovery)
    pairs_to_test = [
        # Tech (7)
        ('MSFT', 'AAPL', 'Microsoft vs Apple'),
        ('NVDA', 'AMD', 'Nvidia vs AMD'),
        ('META', 'GOOGL', 'Meta vs Google'),
        ('TSLA', 'F', 'Tesla vs Ford'),
        ('CRM', 'ADBE', 'Salesforce vs Adobe'),
        ('INTC', 'QCOM', 'Intel vs Qualcomm'),
        ('PYPL', 'SQ', 'PayPal vs Square'),

        # Finance (8)
        ('JPM', 'GS', 'JPMorgan vs Goldman Sachs'),
        ('JPM', 'BAC', 'JPMorgan vs Bank of America'),
        ('BAC', 'WFC', 'Bank of America vs Wells Fargo'),
        ('C', 'GS', 'Citigroup vs Goldman Sachs'),
        ('MS', 'GS', 'Morgan Stanley vs Goldman Sachs'),
        ('BLK', 'AUM', 'BlackRock vs Artisan'),
        ('ICE', 'CME', 'Intercontinental Exchange vs CME'),
        ('SCHW', 'TD', 'Charles Schwab vs Toronto Dominion'),

        # Energy (5)
        ('XOM', 'CVX', 'ExxonMobil vs Chevron'),
        ('COP', 'MPC', 'ConocoPhillips vs Marathon Petroleum'),
        ('PSX', 'VLO', 'Phillips 66 vs Valero'),
        ('EOG', 'SLB', 'EOG Resources vs Schlumberger'),
        ('MRO', 'OXY', 'Marathon Oil vs Occidental'),

        # Consumer (6)
        ('AMZN', 'WMT', 'Amazon vs Walmart'),
        ('MCD', 'YUM', 'McDonalds vs Yum! Brands'),
        ('KO', 'PEP', 'Coca-Cola vs PepsiCo'),
        ('NKE', 'UU', 'Nike vs Ugg'),
        ('HD', 'LOW', 'Home Depot vs Lowe\'s'),
        ('TJX', 'DKS', 'TJX vs Dick\'s Sporting'),

        # Utilities (5)
        ('NEE', 'DUK', 'NextEra vs Duke Energy'),
        ('D', 'SO', 'Dominion vs Southern Company'),
        ('EXC', 'AEP', 'Exelon vs American Electric Power'),
        ('XEL', 'WEC', 'Xcel Energy vs WEC Energy'),
        ('PPL', 'AEE', 'PPL vs Ameren'),

        # Healthcare (6)
        ('JNJ', 'PFE', 'Johnson & Johnson vs Pfizer'),
        ('UNH', 'CI', 'UnitedHealth vs Cigna'),
        ('ABBV', 'LLY', 'AbbVie vs Eli Lilly'),
        ('MRK', 'AMGN', 'Merck vs Amgen'),
        ('TMO', 'ABT', 'Thermo Fisher vs Abbott'),
        ('CVS', 'WBA', 'CVS vs Walgreens'),

        # Industrials (5)
        ('BA', 'LMT', 'Boeing vs Lockheed Martin'),
        ('CAT', 'DE', 'Caterpillar vs Deere'),
        ('GE', 'HON', 'GE vs Honeywell'),
        ('MMM', 'ITW', '3M vs Illinois Tool Works'),
        ('RTX', 'NOC', 'Raytheon vs Northrop Grumman'),
    ]

    # Phase 1: Brute force discovery
    cointegrated = pipeline.phase_1_discovery(pairs_to_test, skip_cached=True)

    # Phase 2: Full analysis on winners
    pipeline.phase_2_full_analysis()

    # Print Phase 2 summary
    pipeline.print_phase_2_summary()

    # Save results
    pipeline.save_results()

    print("\n" + "=" * 100)
    print("HYBRID PIPELINE COMPLETE")
    print("=" * 100)
    print("Phase 1: Discovery - Found cointegrated pairs")
    print("Phase 2: Analysis - Full validation, MLE, backtesting on winners")
    print("=" * 100)

    # Close database connection
    pipeline.db.close_pool()


if __name__ == "__main__":
    main()