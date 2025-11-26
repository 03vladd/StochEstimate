"""
Fetch Real Financial Pairs Data and Validate with Engle-Granger Cointegration

Downloads historical price data for financial pairs from Yahoo Finance,
runs Engle-Granger cointegration test, and validates the resulting spread.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from validation.validation_framework import validate_series
from visualization.validation_visualization import plot_comparison
from preprocessing.engle_granger_cointegration import engle_granger_cointegration


class FinancialPairsFetcher:
    """Fetches financial pairs data from Yahoo Finance"""

    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize fetcher

        Args:
            start_date: YYYY-MM-DD format (defaults to 2 years ago)
            end_date: YYYY-MM-DD format (defaults to today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')

        self.start_date = start_date
        self.end_date = end_date

    def fetch_price_series(self, ticker: str) -> pd.Series:
        """
        Fetch closing price for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            pd.Series with closing prices
        """
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date,
                             progress=False, auto_adjust=True)

            if data.empty:
                print(f"✗ No data for {ticker}")
                return None

            return data['Close']

        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return None

    def fetch_pair_spread(self, ticker1: str, ticker2: str, name: str = None) -> tuple:
        """
        Fetch two assets, test cointegration, and create stationary spread

        Args:
            ticker1: First ticker
            ticker2: Second ticker
            name: Custom pair name

        Returns:
            (spread_series, name, cointegration_result, raw_data)
            Returns (None, name, None, None) if cointegration fails
        """
        if name is None:
            name = f"{ticker1}-{ticker2}"

        print(f"Fetching {name}...", end=" ", flush=True)

        # Fetch both prices
        price1 = self.fetch_price_series(ticker1)
        price2 = self.fetch_price_series(ticker2)

        if price1 is None or price2 is None:
            print("✗ Failed")
            return None, name, None, None

        # Align dates
        common_dates = price1.index.intersection(price2.index)
        price1 = price1.loc[common_dates]
        price2 = price2.loc[common_dates]

        if len(price1) < 100:
            print(f"✗ Insufficient data ({len(price1)} observations)")
            return None, name, None, None

        # Ensure they're Series (not DataFrames)
        if isinstance(price1, pd.DataFrame):
            price1 = price1.iloc[:, 0]
        if isinstance(price2, pd.DataFrame):
            price2 = price2.iloc[:, 0]

        print(f"({len(price1)} obs) - Testing cointegration...", end=" ", flush=True)

        # Test cointegration using Engle-Granger
        coint_result = engle_granger_cointegration(price1.values, price2.values)

        if not coint_result['cointegrated']:
            print(f"✗ NOT cointegrated (p={coint_result['adf_pvalue']:.4f})")
            return None, name, coint_result, {'price1': price1, 'price2': price2}

        # Extract the stationary spread
        spread = pd.Series(coint_result['spread'], index=price1.index)

        print(f"✓ Cointegrated (p={coint_result['adf_pvalue']:.4f})")

        return spread, name, coint_result, {'price1': price1, 'price2': price2}

    def fetch_multiple_pairs(self, pairs_list: list) -> dict:
        """
        Fetch multiple pairs at once

        Args:
            pairs_list: List of tuples (ticker1, ticker2, name) or (ticker1, ticker2)

        Returns:
            Dict of {name: (spread_series, cointegration_result, raw_data)}
        """
        results = {}

        for pair in pairs_list:
            if len(pair) == 3:
                ticker1, ticker2, name = pair
            else:
                ticker1, ticker2 = pair
                name = f"{ticker1}-{ticker2}"

            spread, name, coint_result, raw = self.fetch_pair_spread(ticker1, ticker2, name)

            if spread is not None:
                results[name] = (spread, coint_result, raw)

        return results


def run_validation_on_pairs(pairs_dict: dict) -> dict:
    """
    Run validation framework on cointegrated spreads

    Args:
        pairs_dict: Dict from fetch_multiple_pairs

    Returns:
        Dict of {pair_name: (cointegration_result, ValidationReport)}
    """
    reports = {}

    print("\nRunning validation framework on cointegrated spreads...")
    print("=" * 70)

    for name, (spread, coint_result, raw) in pairs_dict.items():
        print(f"\nValidating {name}...")
        print(f"  Hedge ratio (β): {coint_result['hedge_ratio']:.6f}")
        print(f"  Intercept (α): {coint_result['intercept']:.6f}")
        print(f"  Spread formula: Price_A - {coint_result['hedge_ratio']:.6f} × Price_B + {coint_result['intercept']:.6f}")

        report = validate_series(spread, name=name)
        reports[name] = (coint_result, report)
        print(report.summary())

    return reports


def create_summary_table(reports: dict) -> pd.DataFrame:
    """Create summary table of all validation results"""
    rows = []

    for name, (coint_result, report) in reports.items():
        confidence_level, _ = report.get_confidence_level()

        row = {
            'Pair': name,
            'Cointegrated': '✓' if coint_result['cointegrated'] else '✗',
            'Hedge Ratio': f"{coint_result['hedge_ratio']:.4f}",
            'Observations': report.n_observations,
            'Tests Passed': report.count_passing_tests(),
            'Confidence': confidence_level,
            'Stationarity': '✓' if report.stationarity.passed else '✗',
            'Drift': '✓' if report.linear_drift.passed else '✗',
            'Volatility': '✓' if report.constant_volatility.passed else '✗',
            'ACF': '✓' if report.autocorrelation.passed else '✗',
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    print("=" * 70)
    print("REAL FINANCIAL PAIRS VALIDATION WITH ENGLE-GRANGER COINTEGRATION")
    print("=" * 70)

    # Initialize fetcher (2 years of data)
    fetcher = FinancialPairsFetcher()

    # Define pairs to test
    pairs_to_test = [
        # Tech stocks
        ('MSFT', 'AAPL', 'Microsoft vs Apple'),

        # Financials
        ('JPM', 'GS', 'JPMorgan vs Goldman Sachs'),
        ('JPM', 'BAC', 'JPMorgan vs Bank of America'),

        # Energy
        ('XOM', 'CVX', 'ExxonMobil vs Chevron'),

        # Consumer
        ('AMZN', 'WMT', 'Amazon vs Walmart'),

        # Utilities (typically more stable, potentially more OU-like)
        ('NEE', 'DUK', 'NextEra vs Duke Energy'),

        # Healthcare
        ('JNJ', 'PFE', 'Johnson & Johnson vs Pfizer'),

        # Semiconductors
        ('NVDA', 'AMD', 'Nvidia vs AMD'),
    ]

    print("\n1. FETCHING DATA & TESTING COINTEGRATION")
    print("-" * 70)
    pairs_data = fetcher.fetch_multiple_pairs(pairs_to_test)

    if not pairs_data:
        print("Failed to fetch or cointegrate any pairs.")
        return

    print(f"\n✓ Successfully cointegrated {len(pairs_data)} pairs")

    # Run validation
    print("\n2. VALIDATING COINTEGRATED SPREADS")
    print("-" * 70)
    reports = run_validation_on_pairs(pairs_data)

    # Summary table
    print("\n3. SUMMARY TABLE")
    print("-" * 70)
    summary = create_summary_table(reports)
    print(summary.to_string(index=False))

    # Statistics
    print("\n4. STATISTICS")
    print("-" * 70)
    high_conf = sum(1 for name, (c, r) in reports.items() if r.get_confidence_level()[0] == 'HIGH')
    medium_conf = sum(1 for name, (c, r) in reports.items() if r.get_confidence_level()[0] == 'MEDIUM')
    low_conf = sum(1 for name, (c, r) in reports.items() if r.get_confidence_level()[0] == 'LOW')
    not_ou = sum(1 for name, (c, r) in reports.items() if r.get_confidence_level()[0] == 'NOT_OU')

    print(f"HIGH confidence:   {high_conf} pairs")
    print(f"MEDIUM confidence: {medium_conf} pairs")
    print(f"LOW confidence:    {low_conf} pairs")
    print(f"NOT OU:            {not_ou} pairs")

    # Save results
    print("\n5. SAVING RESULTS")
    print("-" * 70)
    summary.to_csv('validation_results_cointegrated.csv', index=False)
    print("✓ Summary saved to validation_results_cointegrated.csv")

    # Create comparison plot
    try:
        reports_only = {name: report for name, (_, report) in reports.items()}
        plot_comparison(reports_only, save_path="cointegrated_pairs_comparison.png")
        print("✓ Comparison plot saved to cointegrated_pairs_comparison.png")
    except Exception as e:
        print(f"⚠ Could not create comparison plot: {e}")

    # Create individual diagnostic plots for high/medium confidence pairs
    print("\n6. GENERATING DIAGNOSTIC PLOTS")
    print("-" * 70)

    for name, (coint_result, report) in reports.items():
        confidence, _ = report.get_confidence_level()

        if confidence in ['HIGH', 'MEDIUM']:
            try:
                spread, coint_result, raw = pairs_data[name]
                filename = f"diagnostic_{name.replace(' vs ', '_').replace(' ', '_')}.png"
                report.plot_diagnostics(spread, save_path=filename)
            except Exception as e:
                print(f"⚠ Could not create plot for {name}: {e}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nResults saved:")
    print("- validation_results_cointegrated.csv (summary table)")
    print("- cointegrated_pairs_comparison.png (comparison plot)")
    print("- diagnostic_*.png (individual pair diagnostics)")


if __name__ == "__main__":
    main()