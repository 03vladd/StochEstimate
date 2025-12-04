"""
Phase 2 Only: Full Analysis on Cointegrated Pairs

Loads the 4 cointegrated pairs from database and runs:
- Detailed Validation
- MLE Estimation
- Backtesting

This skips Phase 1 discovery and focuses only on deep analysis of winners.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing.engle_granger_cointegration import engle_granger_cointegration
from validation.validation_framework import validate_series
from estimation.mle import estimate_ou_mle
from estimation.backtesting import backtest_pairs_trading
from preprocessing.validate_real_pairs import FinancialPairsFetcher
from database.db_manager import DatabaseManager


class Phase2Pipeline:
    """Phase 2: Full analysis on cointegrated pairs"""

    def __init__(self):
        self.fetcher = FinancialPairsFetcher()
        self.db = DatabaseManager()
        self.analysis_results = {}

    def load_cointegrated_from_db(self) -> list:
        """
        Load all HIGH/MEDIUM confidence pairs from database

        Returns:
            List of (ticker1, ticker2, pair_name) tuples (deduplicated by pair_id)
        """
        query = """
            SELECT DISTINCT ON (p.pair_id) p.pair_id, p.ticker1, p.ticker2, p.pair_name, v.confidence_level, c.adf_pvalue
            FROM pairs p
            JOIN validation_results v ON p.pair_id = v.pair_id
            JOIN cointegration_results c ON p.pair_id = c.pair_id
            WHERE v.confidence_level IN ('HIGH', 'MEDIUM')
            AND c.cointegrated = true
            AND c.adf_pvalue < 0.05
            ORDER BY p.pair_id, v.confidence_level DESC, c.adf_pvalue ASC
        """
        results = self.db.execute_query(query, ())

        # Convert to list of tuples (no duplicates - handled by DISTINCT ON)
        pairs = [(r['ticker1'], r['ticker2'], r['pair_name']) for r in results]

        print(f"\n✓ Loaded {len(pairs)} HIGH/MEDIUM confidence cointegrated pairs from database:")
        for ticker1, ticker2, name in pairs:
            print(f"  - {name}")

        return pairs

    def run_phase_2(self):
        """
        Phase 2: Full analysis on cointegrated pairs
        Runs: Validation (detailed) → MLE → Backtesting
        """

        # Load pairs from database
        cointegrated_pairs = self.load_cointegrated_from_db()

        if not cointegrated_pairs:
            print("✗ No cointegrated pairs found in database")
            return

        print("\n" + "=" * 100)
        print("PHASE 2: FULL ANALYSIS ON COINTEGRATED PAIRS")
        print("Running: Validation → MLE Estimation → Backtesting")
        print("=" * 100)

        for pair_idx, (ticker1, ticker2, name) in enumerate(cointegrated_pairs, 1):
            print(f"\n{'=' * 100}")
            print(f"[{pair_idx}/{len(cointegrated_pairs)}] DETAILED ANALYSIS: {name}")
            print(f"{'=' * 100}")

            try:
                # Fetch data
                spread, pair_name, coint_result, raw = self.fetcher.fetch_pair_spread(ticker1, ticker2, name)

                if spread is None:
                    print("✗ Failed to fetch data")
                    continue

                # A. DETAILED VALIDATION
                print(f"\nA. VALIDATION (5 Tests on {len(spread)} observations)")
                print(f"   {'─' * 96}")
                print(f"   Hedge ratio (β): {coint_result['hedge_ratio']:.6f}")
                print(f"   Intercept (α): {coint_result['intercept']:.6f}")
                print(f"   Spread formula: Price_A - {coint_result['hedge_ratio']:.6f} × Price_B + {coint_result['intercept']:.6f}")

                validation_report = validate_series(spread, name=name)
                print(validation_report.summary())

                # B. MLE ESTIMATION
                print(f"\nB. MLE PARAMETER ESTIMATION")
                print(f"   {'─' * 96}")
                print(f"   Running maximum likelihood optimization...")

                mle_result = estimate_ou_mle(spread, verbose=True)
                print(mle_result.detailed_result)

                # C. BACKTESTING
                print(f"\nC. BACKTESTING")
                print(f"   {'─' * 96}")
                print(f"   Running backtest with MLE parameters...")
                print(f"   Entry threshold: 1.5σ = ${1.5 * mle_result.sigma:.2f}")
                print(f"   Exit threshold: 0.5σ = ${0.5 * mle_result.sigma:.2f}")
                print(f"   Max holding period: 30 days\n")

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

                # Store results
                self.analysis_results[name] = {
                    'cointegration': coint_result,
                    'validation': validation_report,
                    'mle': mle_result,
                    'backtest': backtest_result,
                    'spread': spread
                }

            except Exception as e:
                print(f"✗ Analysis failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    def create_summary_table(self) -> pd.DataFrame:
        """Create comprehensive summary table"""
        rows = []

        for name, result_dict in self.analysis_results.items():
            coint = result_dict['cointegration']
            val = result_dict['validation']
            mle = result_dict['mle']
            bt = result_dict['backtest']

            confidence_level, _ = val.get_confidence_level()

            row = {
                'Pair': name,
                'Cointegrated': '✓',
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
        """Print comprehensive summary"""
        if not self.analysis_results:
            print("\n✗ No analysis results to summarize")
            return

        print("\n" + "=" * 100)
        print("PHASE 2: DETAILED RESULTS SUMMARY")
        print("=" * 100)

        summary_df = self.create_summary_table()
        print(summary_df.to_string(index=False))

        # Parameter Statistics
        print("\n" + "-" * 100)
        print("PARAMETER STATISTICS (from MLE on Real Data)")
        print("-" * 100)

        thetas = [r['mle'].theta for r in self.analysis_results.values()]
        mus = [r['mle'].mu for r in self.analysis_results.values()]
        sigmas = [r['mle'].sigma for r in self.analysis_results.values()]

        if thetas:
            print(f"\nMean Reversion Speed (θ):")
            print(f"  Mean:     {np.mean(thetas):.6f} per day")
            print(f"  Median:   {np.median(thetas):.6f} per day")
            print(f"  Range:    [{np.min(thetas):.6f}, {np.max(thetas):.6f}]")
            print(f"  Std Dev:  {np.std(thetas):.6f}")

            print(f"\n  Interpretation (half-life to 50% reversion):")
            for name, result in self.analysis_results.items():
                theta = result['mle'].theta
                half_life = np.log(2) / theta if theta > 0 else np.inf
                print(f"    {name:40s}: {half_life:6.2f} days")

            print(f"\nLong-term Mean (μ):")
            print(f"  Mean:   {np.mean(mus):.6f}")
            print(f"  Range:  [{np.min(mus):.6f}, {np.max(mus):.6f}]")
            print(f"  (Near zero indicates pairs are fairly priced)")

            print(f"\nVolatility (σ):")
            print(f"  Mean:   {np.mean(sigmas):.6f}")
            print(f"  Median: {np.median(sigmas):.6f}")
            print(f"  Range:  [{np.min(sigmas):.6f}, {np.max(sigmas):.6f}]")
            print(f"  (Used to set trading thresholds: Entry=1.5σ, Exit=0.5σ)")

        # Backtesting Aggregate
        print("\n" + "-" * 100)
        print("AGGREGATE BACKTESTING STATISTICS")
        print("-" * 100)

        total_trades = sum(r['backtest'].num_trades for r in self.analysis_results.values())
        total_pnl = sum(r['backtest'].total_profit for r in self.analysis_results.values())
        win_count = sum(len(r['backtest'].winning_trades) for r in self.analysis_results.values())

        print(f"  Total trades across all pairs: {total_trades}")
        print(f"  Winning trades: {win_count}")
        print(f"  Losing trades: {total_trades - win_count}")

        if total_trades > 0:
            overall_win_rate = (win_count / total_trades) * 100
            print(f"  Overall win rate: {overall_win_rate:.1f}%")

        print(f"  Total P&L: ${total_pnl:,.2f}")

        if total_trades > 0:
            avg_pnl_per_trade = total_pnl / total_trades
            print(f"  Average P&L per trade: ${avg_pnl_per_trade:,.2f}")

        # Confidence Level Distribution
        print("\n" + "-" * 100)
        print("VALIDATION CONFIDENCE LEVELS")
        print("-" * 100)

        for name, result in self.analysis_results.items():
            conf_level, _ = result['validation'].get_confidence_level()
            tests_passed = result['validation'].count_passing_tests()
            print(f"  {name:40s}: {conf_level:6s} ({tests_passed}/4 tests passed)")

    def save_results(self, output_dir: str = "."):
        """Save results to CSV"""
        if not self.analysis_results:
            print("No results to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary
        summary_df = self.create_summary_table()
        summary_file = f"{output_dir}/phase2_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to {summary_file}")

        # Detailed
        detailed_rows = []
        for name, result_dict in self.analysis_results.items():
            mle = result_dict['mle']
            val = result_dict['validation']
            coint = result_dict['cointegration']
            bt = result_dict['backtest']

            detailed_rows.append({
                'Pair': name,
                'Hedge_Ratio': coint['hedge_ratio'],
                'Adf_Pvalue': coint['adf_pvalue'],
                'Validation_Tests_Passed': val.count_passing_tests(),
                'Confidence_Level': val.get_confidence_level()[0],
                'Theta_Estimate': mle.theta,
                'Theta_CI_Lower': mle.theta_ci[0],
                'Theta_CI_Upper': mle.theta_ci[1],
                'Mu_Estimate': mle.mu,
                'Sigma_Estimate': mle.sigma,
                'Log_Likelihood': mle.log_likelihood,
                'Backtest_Num_Trades': bt.num_trades,
                'Backtest_Win_Rate': bt.win_rate,
                'Backtest_Total_PnL': bt.total_profit,
                'Backtest_Avg_Trade_PnL': bt.avg_profit_per_trade,
                'Backtest_Avg_Holding_Days': bt.avg_holding_period,
            })

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_file = f"{output_dir}/phase2_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"✓ Detailed results saved to {detailed_file}")


def main():
    """Main entry point"""

    print("\n" + "=" * 100)
    print("PHASE 2 ONLY: FULL ANALYSIS ON COINTEGRATED PAIRS")
    print("=" * 100)

    # Initialize pipeline
    pipeline = Phase2Pipeline()

    # Run Phase 2
    pipeline.run_phase_2()

    # Print summary
    pipeline.print_summary()

    # Save results
    pipeline.save_results()

    print("\n" + "=" * 100)
    print("PHASE 2 COMPLETE")
    print("=" * 100)

    # Close database
    pipeline.db.close_pool()


if __name__ == "__main__":
    main()