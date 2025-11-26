"""
Complete Pipeline: Cointegration → Validation → MLE Estimation → Backtesting

Orchestrates the full workflow for OU parameter estimation and validation on financial pairs.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing.engle_granger_cointegration import engle_granger_cointegration
from validation.validation_framework import validate_series
from estimation.mle import estimate_ou_mle
from estimation.backtesting import backtest_pairs_trading
from preprocessing.validate_real_pairs import FinancialPairsFetcher


class OUPipeline:
    """Complete pipeline for OU analysis on financial pairs"""

    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize pipeline

        Args:
            start_date: YYYY-MM-DD (default: 2 years ago)
            end_date: YYYY-MM-DD (default: today)
        """
        self.fetcher = FinancialPairsFetcher(start_date, end_date)
        self.results = {}

    def run_full_pipeline(self, pairs_to_test: list) -> dict:
        """
        Run complete pipeline on multiple pairs

        Args:
            pairs_to_test: List of (ticker1, ticker2, name) tuples

        Returns:
            Dictionary with complete results for each pair
        """

        print("\n" + "=" * 80)
        print("COMPLETE OU ESTIMATION PIPELINE")
        print("Cointegration → Validation → MLE Estimation → Backtesting")
        print("=" * 80)

        # Step 1: Fetch and cointegrate
        print("\n1. FETCHING DATA & TESTING COINTEGRATION")
        print("-" * 80)
        pairs_data = self.fetcher.fetch_multiple_pairs(pairs_to_test)

        if not pairs_data:
            print("✗ Failed to fetch or cointegrate any pairs.")
            return {}

        print(f"✓ Successfully cointegrated {len(pairs_data)} pairs\n")

        # Step 2: Validate, estimate, and backtest
        print("2. VALIDATION & MLE ESTIMATION & BACKTESTING")
        print("-" * 80)

        for name, (spread, coint_result, raw) in pairs_data.items():
            print(f"\n{'=' * 80}")
            print(f"PROCESSING: {name}")
            print(f"{'=' * 80}")

            # Run validation
            print(f"\nA. VALIDATION")
            print(f"   Hedge ratio (β): {coint_result['hedge_ratio']:.6f}")
            print(f"   Intercept (α): {coint_result['intercept']:.6f}")
            print(f"   Spread formula: Price_A - {coint_result['hedge_ratio']:.6f} × Price_B")

            validation_report = validate_series(spread, name=name)
            print(validation_report.summary())

            # Run MLE estimation
            print(f"B. MLE PARAMETER ESTIMATION")
            print(f"   Running optimization on {len(spread)} observations...")

            mle_result = estimate_ou_mle(spread, verbose=True)
            print(mle_result.detailed_result)

            # Run backtesting
            print(f"\nC. BACKTESTING")
            print(f"   Running backtest with MLE parameters...")

            backtest_result = backtest_pairs_trading(
                spread=spread,
                theta=mle_result.theta,
                mu=mle_result.mu,
                sigma=mle_result.sigma,
                pair_name=name,
                entry_threshold=1.5,
                max_holding_days=30
            )
            print(backtest_result.detailed_result)

            # Store all results
            self.results[name] = {
                'cointegration': coint_result,
                'validation': validation_report,
                'mle': mle_result,
                'backtest': backtest_result,
                'raw_data': raw,
                'spread': spread
            }

        return self.results

    def create_summary_table(self) -> pd.DataFrame:
        """Create comprehensive summary table of all results"""
        rows = []

        for name, result_dict in self.results.items():
            coint = result_dict['cointegration']
            val = result_dict['validation']
            mle = result_dict['mle']
            bt = result_dict['backtest']

            confidence_level, _ = val.get_confidence_level()

            row = {
                'Pair': name,
                'Cointegrated': '✓' if coint['cointegrated'] else '✗',
                'Hedge Ratio (β)': f"{coint['hedge_ratio']:.4f}",
                'Observations': val.n_observations,
                'Validation Tests': f"{val.count_passing_tests()}/4",
                'Confidence': confidence_level,
                'θ (Reversion Speed)': f"{mle.theta:.6f}",
                'θ CI': f"[{mle.theta_ci[0]:.6f}, {mle.theta_ci[1]:.6f}]",
                'μ (Long-term Mean)': f"{mle.mu:.6f}",
                'σ (Volatility)': f"{mle.sigma:.6f}",
                'Log-Likelihood': f"{mle.log_likelihood:.2f}",
                'Num Trades': bt.num_trades,
                'Win Rate': f"{bt.win_rate:.1f}%",
                'Total P&L': f"${bt.total_profit:,.2f}",
                'Avg Trade P&L': f"${bt.avg_profit_per_trade:,.2f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize.")
            return

        print("\n" + "=" * 80)
        print("3. COMPLETE RESULTS SUMMARY")
        print("=" * 80)

        summary_df = self.create_summary_table()
        print(summary_df.to_string(index=False))

        # Statistics by confidence level
        print("\n" + "-" * 80)
        print("CONFIDENCE LEVEL DISTRIBUTION:")
        print("-" * 80)

        confidence_counts = {}
        for name, result_dict in self.results.items():
            conf_level, _ = result_dict['validation'].get_confidence_level()
            confidence_counts[conf_level] = confidence_counts.get(conf_level, 0) + 1

        for level in ['HIGH', 'MEDIUM', 'LOW', 'NOT_OU']:
            count = confidence_counts.get(level, 0)
            print(f"  {level:12s}: {count} pairs")

        # Mean reversion statistics
        print("\n" + "-" * 80)
        print("MEAN REVERSION SPEED (θ) STATISTICS:")
        print("-" * 80)

        thetas = [result_dict['mle'].theta for result_dict in self.results.values()]
        if thetas:
            print(f"  Mean θ:        {np.mean(thetas):.6f}")
            print(f"  Median θ:      {np.median(thetas):.6f}")
            print(f"  Min θ:         {np.min(thetas):.6f}")
            print(f"  Max θ:         {np.max(thetas):.6f}")
            print(f"  Std Dev θ:     {np.std(thetas):.6f}")

            print(f"\n  Interpretation:")
            for percentile in [10, 50, 90]:
                p_val = np.percentile(thetas, percentile)
                half_life = np.log(2) / p_val if p_val > 0 else np.inf
                print(f"    {percentile}th percentile: θ={p_val:.6f} → {half_life:.2f} days to 50% reversion")

        # Backtesting statistics
        print("\n" + "-" * 80)
        print("BACKTESTING STATISTICS:")
        print("-" * 80)

        total_trades = sum(result_dict['backtest'].num_trades for result_dict in self.results.values())
        total_pnl = sum(result_dict['backtest'].total_profit for result_dict in self.results.values())

        print(f"  Total trades across all pairs: {total_trades}")
        print(f"  Total P&L: ${total_pnl:,.2f}")

        if total_trades > 0:
            win_count = sum(len(result_dict['backtest'].winning_trades) for result_dict in self.results.values())
            overall_win_rate = (win_count / total_trades) * 100
            print(f"  Overall win rate: {overall_win_rate:.1f}%")

    def save_results(self, output_dir: str = "."):
        """Save all results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main summary table
        summary_df = self.create_summary_table()
        summary_file = f"{output_dir}/ou_analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to {summary_file}")

        # Detailed results for each pair
        detailed_rows = []
        for name, result_dict in self.results.items():
            mle = result_dict['mle']
            val = result_dict['validation']
            coint = result_dict['cointegration']
            bt = result_dict['backtest']

            detailed_rows.append({
                'Pair': name,
                'Cointegrated': coint['cointegrated'],
                'Hedge_Ratio': coint['hedge_ratio'],
                'Intercept': coint['intercept'],
                'N_Observations': val.n_observations,
                'Validation_Tests_Passed': val.count_passing_tests(),
                'Confidence_Level': val.get_confidence_level()[0],
                'Stationarity_Pass': val.stationarity.passed,
                'Drift_Pass': val.linear_drift.passed,
                'Volatility_Pass': val.constant_volatility.passed,
                'ACF_Pass': val.autocorrelation.passed,
                'Theta_Estimate': mle.theta,
                'Theta_CI_Lower': mle.theta_ci[0],
                'Theta_CI_Upper': mle.theta_ci[1],
                'Mu_Estimate': mle.mu,
                'Mu_CI_Lower': mle.mu_ci[0],
                'Mu_CI_Upper': mle.mu_ci[1],
                'Sigma_Estimate': mle.sigma,
                'Sigma_CI_Lower': mle.sigma_ci[0],
                'Sigma_CI_Upper': mle.sigma_ci[1],
                'Log_Likelihood': mle.log_likelihood,
                'MLE_Converged': mle.success,
                'Backtest_Num_Trades': bt.num_trades,
                'Backtest_Win_Rate': bt.win_rate,
                'Backtest_Total_PnL': bt.total_profit,
                'Backtest_Avg_Trade_PnL': bt.avg_profit_per_trade,
                'Backtest_Avg_Holding_Days': bt.avg_holding_period,
            })

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_file = f"{output_dir}/ou_analysis_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"✓ Detailed results saved to {detailed_file}")

        return summary_file, detailed_file


def main():
    """Main entry point"""

    # Initialize pipeline
    pipeline = OUPipeline()

    # Define pairs to test
    pairs_to_test = [
        ('MSFT', 'AAPL', 'Microsoft vs Apple'),
        ('JPM', 'GS', 'JPMorgan vs Goldman Sachs'),
        ('JPM', 'BAC', 'JPMorgan vs Bank of America'),
        ('XOM', 'CVX', 'ExxonMobil vs Chevron'),
        ('AMZN', 'WMT', 'Amazon vs Walmart'),
        ('NEE', 'DUK', 'NextEra vs Duke Energy'),
        ('JNJ', 'PFE', 'Johnson & Johnson vs Pfizer'),
        ('NVDA', 'AMD', 'Nvidia vs AMD'),
    ]

    # Run full pipeline
    results = pipeline.run_full_pipeline(pairs_to_test)

    # Print summary
    pipeline.print_summary()

    # Save results
    pipeline.save_results()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()