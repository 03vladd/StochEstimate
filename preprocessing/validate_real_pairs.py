"""
Fetch Real Financial Pairs Data and Test Engle-Granger Cointegration

Downloads historical price data for financial pairs from Yahoo Finance
and tests for cointegration using Engle-Granger method.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
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