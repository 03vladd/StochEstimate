"""
Screening script to find HIGH confidence OU pairs for backtesting.

Runs all candidates through:
  1. Engle-Granger cointegration
  2. Full 5-test OU validation battery

Prints a ranked summary and saves results to CSV.

Usage:
    python screening/screen_high_confidence_pairs.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.validate_real_pairs import FinancialPairsFetcher
from validation.validation_framework import validate_series

# ── Candidate pairs ────────────────────────────────────────────────────────────
# Chosen from sectors with strong historical pairs-trading literature.
# Each entry: (ticker1, ticker2, sector, note)

CANDIDATES = [
    # Utilities — regulated, correlated capex cycles
    ("D",   "SO",  "Utilities",        "Dominion / Southern Company"),
    ("XEL", "WEC", "Utilities",        "Xcel / WEC Energy"),
    ("ED",  "EIX", "Utilities",        "ConEd / Edison International"),
    ("AEP", "ETR", "Utilities",        "AEP / Entergy"),
    ("DTE", "PPL", "Utilities",        "DTE Energy / PPL"),

    # Energy majors — commodity-driven
    ("CVX", "XOM", "Energy",           "Chevron / ExxonMobil"),
    ("COP", "EOG", "Energy",           "ConocoPhillips / EOG"),
    ("OXY", "DVN", "Energy",           "Occidental / Devon"),

    # Consumer staples — non-cyclical, brand-driven
    ("KO",  "PEP", "Consumer Staples", "Coca-Cola / PepsiCo"),
    ("PG",  "CL",  "Consumer Staples", "P&G / Colgate"),
    ("KMB", "CLX", "Consumer Staples", "Kimberly-Clark / Clorox"),
    ("COST","WMT", "Consumer Staples", "Costco / Walmart"),

    # Gold miners — same commodity driver
    ("NEM", "GOLD","Materials",        "Newmont / Barrick Gold"),
    ("AEM", "WPM", "Materials",        "Agnico Eagle / Wheaton"),

    # Telecom — regulated duopoly
    ("VZ",  "T",   "Telecom",          "Verizon / AT&T"),

    # Banking — already tried JPM/GS, trying other combos
    ("BAC", "WFC", "Banking",          "Bank of America / Wells Fargo"),
    ("C",   "BAC", "Banking",          "Citigroup / BofA"),
    ("MS",  "GS",  "Banking",          "Morgan Stanley / Goldman Sachs"),

    # Insurance — correlated underwriting cycles
    ("ALL", "TRV", "Insurance",        "Allstate / Travelers"),
    ("MET", "PRU", "Insurance",        "MetLife / Prudential"),

    # REITs — similar property types
    ("O",   "NNN", "REIT",             "Realty Income / NNN"),
    ("SPG", "MAC", "REIT",             "Simon Property / Macerich"),

    # Healthcare — correlated pharma/device cycles
    ("JNJ", "ABT", "Healthcare",       "J&J / Abbott"),
    ("MDT", "BSX", "Healthcare",       "Medtronic / Boston Scientific"),
]


def screen_pair(fetcher, ticker1, ticker2, sector, note):
    """Run one pair through cointegration + validation. Returns a result dict."""
    spread, name, coint_result, _ = fetcher.fetch_pair_spread(ticker1, ticker2, name=note)

    if spread is None:
        return {
            'pair': note, 'sector': sector,
            'cointegrated': False,
            'confidence': 'NOT_COINTEGRATED',
            'tests_passed': 0,
            'stationarity': False, 'drift': False,
            'volatility': False, 'acf': False,
            'acf_r2': None, 'acf_theta': None,
            'n_obs': 0,
        }

    report = validate_series(spread, name=note)
    confidence, _ = report.get_confidence_level()

    return {
        'pair': note,
        'sector': sector,
        'cointegrated': True,
        'confidence': confidence,
        'tests_passed': report.count_passing_tests(),
        'stationarity': report.stationarity.passed,
        'drift': report.linear_drift.passed,
        'volatility': report.constant_volatility.passed,
        'acf': report.autocorrelation.passed,
        'acf_r2': report.autocorrelation.r_squared,
        'acf_theta': report.autocorrelation.theta,
        'n_obs': report.n_observations,
    }


def print_summary(results: list[dict]):
    confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2,
                        'NOT_OU': 3, 'NOT_COINTEGRATED': 4}
    results_sorted = sorted(results, key=lambda r: confidence_order.get(r['confidence'], 5))

    W = 100
    print("\n" + "=" * W)
    print("SCREENING RESULTS — Ranked by OU confidence")
    print("=" * W)
    print(f"{'Pair':<35} {'Sector':<18} {'Conf':<8} {'S':>2} {'D':>2} {'V':>2} {'A':>2} "
          f"{'ACF R²':>8} {'ACF θ':>8} {'n':>5}")
    print("-" * W)

    for r in results_sorted:
        s = '✓' if r['stationarity'] else '✗'
        d = '✓' if r['drift']        else '✗'
        v = '✓' if r['volatility']   else '✗'
        a = '✓' if r['acf']          else '✗'
        r2  = f"{r['acf_r2']:.3f}"  if r['acf_r2']    is not None else "  —  "
        th  = f"{r['acf_theta']:.3f}" if r['acf_theta'] is not None else "  —  "
        print(f"{r['pair']:<35} {r['sector']:<18} {r['confidence']:<8} "
              f"{s:>2} {d:>2} {v:>2} {a:>2} {r2:>8} {th:>8} {r['n_obs']:>5}")

    print("=" * W)
    high   = [r for r in results if r['confidence'] == 'HIGH']
    medium = [r for r in results if r['confidence'] == 'MEDIUM']
    print(f"\nHIGH confidence:   {len(high)}")
    print(f"MEDIUM confidence: {len(medium)}")
    print(f"\nColumns: S=Stationarity  D=Absence of drift  V=Volatility stability  A=ACF decay")

    if high:
        print(f"\n{'='*W}")
        print("HIGH CONFIDENCE PAIRS — suitable for backtesting:")
        for r in high:
            print(f"  {r['pair']}  (θ̂≈{r['acf_theta']:.3f}, ACF R²={r['acf_r2']:.3f}, n={r['n_obs']})")
    print("=" * W)


def main():
    # Use 5-year window: longer samples give cointegration tests more power
    from datetime import datetime, timedelta
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    fetcher = FinancialPairsFetcher(start_date=start_date, end_date=end_date)

    print("=" * 70)
    print(f"PAIR SCREENING — {len(CANDIDATES)} candidates")
    print(f"Date range: {fetcher.start_date} to {fetcher.end_date}")
    print("=" * 70)

    results = []
    for ticker1, ticker2, sector, note in CANDIDATES:
        result = screen_pair(fetcher, ticker1, ticker2, sector, note)
        results.append(result)

    print_summary(results)

    # Save to CSV
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'screening_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
