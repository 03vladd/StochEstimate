"""
Initialize PostgreSQL schema for StochEstimate

Creates all 5 tables with proper relationships and constraints
"""

import psycopg2
from psycopg2 import sql


def create_schema():
    """Create all tables in the database"""

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="stochestimate",
        user="stochestimate",
        password="stochestimate_dev"
    )

    cursor = conn.cursor()

    print("Creating schema...")

    # 1. pairs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            pair_id SERIAL PRIMARY KEY,
            ticker1 VARCHAR(10) NOT NULL,
            ticker2 VARCHAR(10) NOT NULL,
            pair_name VARCHAR(100) NOT NULL,
            sector VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("✓ Created pairs table")

    # 2. price_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            price_id SERIAL PRIMARY KEY,
            pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            close NUMERIC NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            volume BIGINT,
            fetch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("✓ Created price_data table")

    # 3. cointegration_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cointegration_results (
            coint_id SERIAL PRIMARY KEY,
            pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
            interval VARCHAR(10) NOT NULL,
            hedge_ratio NUMERIC NOT NULL,
            intercept NUMERIC NOT NULL,
            adf_statistic NUMERIC NOT NULL,
            adf_pvalue NUMERIC NOT NULL,
            cointegrated BOOLEAN NOT NULL,
            observations_used INTEGER NOT NULL,
            tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair_id, interval)
        );
    """)
    print("✓ Created cointegration_results table")

    # 4. validation_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            val_id SERIAL PRIMARY KEY,
            pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
            interval VARCHAR(10) NOT NULL,
            coint_id INTEGER REFERENCES cointegration_results(coint_id),
            stationarity_passed BOOLEAN,
            stationarity_pvalue NUMERIC,
            stationarity_statistic NUMERIC,
            drift_passed BOOLEAN,
            drift_pvalue NUMERIC,
            drift_statistic NUMERIC,
            volatility_passed BOOLEAN,
            volatility_pvalue NUMERIC,
            volatility_statistic NUMERIC,
            autocorr_passed BOOLEAN,
            autocorr_pvalue NUMERIC,
            autocorr_statistic NUMERIC,
            normality_passed BOOLEAN,
            normality_pvalue NUMERIC,
            normality_statistic NUMERIC,
            confidence_level VARCHAR(20) NOT NULL,
            tests_passed_count INTEGER NOT NULL,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair_id, interval)
        );
    """)
    print("✓ Created validation_results table")

    # 5. lstm_models table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lstm_models (
            model_id SERIAL PRIMARY KEY,
            interval VARCHAR(10) NOT NULL,
            validation_criteria VARCHAR(50) NOT NULL,
            model_filename VARCHAR(255) NOT NULL,
            training_pairs_count INTEGER NOT NULL,
            training_data_points_used INTEGER NOT NULL,
            mae_validation_loss NUMERIC,
            rmse_validation_loss NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version INTEGER NOT NULL,
            UNIQUE(interval, validation_criteria, version)
        );
    """)
    print("✓ Created lstm_models table")

    conn.commit()
    cursor.close()
    conn.close()

    print("\n✓ Schema created successfully!")


if __name__ == "__main__":
    try:
        create_schema()
    except Exception as e:
        print(f"✗ Error: {e}")