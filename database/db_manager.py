"""
Database manager for StochEstimate
Handles all connections and CRUD operations for the PostgreSQL database
"""

from psycopg2 import pool
import psycopg2
from typing import Optional, List, Dict, Tuple
from datetime import datetime


class DatabaseManager:
    """Manages PostgreSQL connections and database operations"""

    def __init__(self, host: str = 'localhost', port: int = 5432,
                 database: str = 'stochestimate', user: str = 'stochestimate',
                 password: str = 'stochestimate_dev', min_connections: int = 2,
                 max_connections: int = 5):
        """
        Initialize database manager with connection pool

        Args:
            host: PostgreSQL server hostname
            port: PostgreSQL server port
            database: Database name
            user: Database username
            password: Database password
            min_connections: Minimum connections to maintain in pool
            max_connections: Maximum connections allowed in pool
        """
        try:
            # Create a connection pool (reusable connections)
            self.connection_pool = pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            print(f"✓ Database connection pool created ({min_connections}-{max_connections} connections)")
        except Exception as e:
            print(f"✗ Failed to create connection pool: {e}")
            raise

    def get_connection(self):
        """
        Get a connection from the pool

        Returns:
            psycopg2 connection object
        """
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """
        Return a connection to the pool for reuse

        Args:
            conn: Connection to return
        """
        if conn:
            self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute a SELECT query and return results as list of dicts

        Args:
            query: SQL SELECT query
            params: Query parameters (for safe parameterized queries)

        Returns:
            List of dictionaries (one dict per row)
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Execute with parameters to prevent SQL injection
            cursor.execute(query, params or ())

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Convert rows to list of dicts
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            cursor.close()
            return results

        except Exception as e:
            print(f"✗ Query execution failed: {e}")
            return []

        finally:
            if conn:
                self.return_connection(conn)

    def execute_insert(self, query: str, params: tuple = None) -> Optional[int]:
        """
        Execute an INSERT query and return the inserted row's ID

        Args:
            query: SQL INSERT query (should include RETURNING id)
            params: Query parameters

        Returns:
            The ID of the inserted row, or None if failed
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params or ())

            # If query returns an ID, fetch it
            result = cursor.fetchone()
            inserted_id = result[0] if result else None

            conn.commit()
            cursor.close()
            return inserted_id

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"✗ Insert failed: {e}")
            return None

        finally:
            if conn:
                self.return_connection(conn)

    def execute_update(self, query: str, params: tuple = None) -> bool:
        """
        Execute an UPDATE or DELETE query

        Args:
            query: SQL UPDATE/DELETE query
            params: Query parameters

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params or ())
            conn.commit()
            cursor.close()
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"✗ Update failed: {e}")
            return False

        finally:
            if conn:
                self.return_connection(conn)

    def close_pool(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✓ Connection pool closed")

    def insert_pair(self, ticker1: str, ticker2: str, pair_name: str, sector: str = None)-> Optional[int]:
        """
            Insert a new financial pair into the database

            Args:
                ticker1: First stock ticker (e.g., 'JPM')
                ticker2: Second stock ticker (e.g., 'GS')
                pair_name: Human-readable name (e.g., 'JPMorgan vs Goldman Sachs')
                sector: Optional sector category (e.g., 'Financial')

            Returns:
                The pair_id if successful, None otherwise
            """
        query = """
            INSERT INTO pairs (ticker1, ticker2, pair_name, sector)
            VALUES (%s, %s, %s, %s)
            RETURNING pair_id
        """
        params = (ticker1, ticker2, pair_name, sector)
        return self.execute_insert(query, params)

    def insert_cointegration_result(self, pair_id: int, interval: str, hedge_ratio: float,
                                    intercept: float, adf_statistic: float, adf_pvalue: float,
                                    cointegrated: bool, observations_used: int) -> Optional[int]:
        """
        Insert cointegration test results

        Args:
            pair_id: ID of the pair being tested
            interval: Time interval ('1d', '1h', '5m', etc.)
            hedge_ratio: Beta coefficient from Engle-Granger regression
            intercept: Alpha constant from regression
            adf_statistic: ADF test statistic value
            adf_pvalue: P-value from ADF test
            cointegrated: Boolean - did they pass the test?
            observations_used: How many data points were in the test

        Returns:
            The coint_id if successful, None otherwise
        """
        query = """
            INSERT INTO cointegration_results 
            (pair_id, interval, hedge_ratio, intercept, adf_statistic, adf_pvalue, cointegrated, observations_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING coint_id
        """
        params = (pair_id, interval, hedge_ratio, intercept, adf_statistic, adf_pvalue, cointegrated, observations_used)
        return self.execute_insert(query, params)

    def insert_validation_result(self, pair_id: int, interval: str, coint_id: int,
                                 stationarity_passed: bool, stationarity_pvalue: float, stationarity_statistic: float,
                                 drift_passed: bool, drift_pvalue: float, drift_statistic: float,
                                 volatility_passed: bool, volatility_pvalue: float, volatility_statistic: float,
                                 autocorr_passed: bool, autocorr_pvalue: float, autocorr_statistic: float,
                                 normality_passed: bool, normality_pvalue: float, normality_statistic: float,
                                 confidence_level: str, tests_passed_count: int) -> Optional[int]:
        """
        Insert validation framework results for a pair

        Args:
            pair_id: ID of the pair
            interval: Time interval ('1d', '1h', etc.)
            coint_id: ID of the cointegration result this validates
            stationarity_passed: Boolean - did stationarity test pass?
            stationarity_pvalue: P-value from ADF test
            stationarity_statistic: Test statistic
            drift_passed: Boolean - did drift test pass?
            drift_pvalue: P-value
            drift_statistic: Test statistic
            volatility_passed: Boolean - did volatility test pass?
            volatility_pvalue: P-value from Levene's test
            volatility_statistic: Test statistic
            autocorr_passed: Boolean - did autocorrelation test pass?
            autocorr_pvalue: P-value
            autocorr_statistic: Test statistic
            normality_passed: Boolean - did normality test pass?
            normality_pvalue: P-value from normality test
            normality_statistic: Test statistic
            confidence_level: 'HIGH', 'MEDIUM', 'LOW', or 'NOT_OU'
            tests_passed_count: How many of the 5 tests passed (0-5)

        Returns:
            The val_id if successful, None otherwise
        """
        query = """
            INSERT INTO validation_results 
            (pair_id, interval, coint_id, 
             stationarity_passed, stationarity_pvalue, stationarity_statistic,
             drift_passed, drift_pvalue, drift_statistic,
             volatility_passed, volatility_pvalue, volatility_statistic,
             autocorr_passed, autocorr_pvalue, autocorr_statistic,
             normality_passed, normality_pvalue, normality_statistic,
             confidence_level, tests_passed_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING val_id
        """
        params = (
            pair_id, interval, coint_id,
            stationarity_passed, stationarity_pvalue, stationarity_statistic,
            drift_passed, drift_pvalue, drift_statistic,
            volatility_passed, volatility_pvalue, volatility_statistic,
            autocorr_passed, autocorr_pvalue, autocorr_statistic,
            normality_passed, normality_pvalue, normality_statistic,
            confidence_level, tests_passed_count
        )
        return self.execute_insert(query, params)

    def insert_price_data(self, pair_id: int, interval: str, timestamp: str,
                          close: float, open: float = None, high: float = None,
                          low: float = None, volume: int = None) -> Optional[int]:
        """
        Insert price data for a pair

        Args:
            pair_id: ID of the pair
            interval: Time interval ('1d', '1h', '5m', etc.)
            timestamp: When this price occurred (e.g., '2024-12-04 16:00:00')
            close: Closing price (required)
            open: Opening price (optional)
            high: High price (optional)
            low: Low price (optional)
            volume: Trading volume (optional)

        Returns:
            The price_id if successful, None otherwise
        """
        query = """
            INSERT INTO price_data (pair_id, interval, timestamp, close, open, high, low, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING price_id
        """
        params = (pair_id, interval, timestamp, close, open, high, low, volume)
        return self.execute_insert(query, params)

    def get_validation_results(self, pair_id: int = None, interval: str = None,
                               confidence_level: str = None) -> List[Dict]:
        """
        Query validation results with optional filters

        Args:
            pair_id: Filter by specific pair (optional)
            interval: Filter by time interval like '1d' (optional)
            confidence_level: Filter by confidence level: 'HIGH', 'MEDIUM', 'LOW', 'NOT_OU' (optional)

        Returns:
            List of validation result dictionaries
        """
        query = "SELECT * FROM validation_results WHERE 1=1"
        params = []

        # Build WHERE clause dynamically based on provided filters
        if pair_id is not None:
            query += " AND pair_id = %s"
            params.append(pair_id)

        if interval is not None:
            query += " AND interval = %s"
            params.append(interval)

        if confidence_level is not None:
            query += " AND confidence_level = %s"
            params.append(confidence_level)

        return self.execute_query(query, tuple(params))