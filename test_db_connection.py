"""
Test connection to PostgreSQL database running in Docker
"""

import psycopg2
from psycopg2 import sql

# Connection parameters (must match docker-compose.yml)
conn_params = {
    'host': 'localhost',  # PostgreSQL is on your local machine
    'port': 5432,  # Default PostgreSQL port
    'database': 'stochestimate',  # Database name from docker-compose.yml
    'user': 'stochestimate',  # Username from docker-compose.yml
    'password': 'stochestimate_dev'  # Password from docker-compose.yml
}

try:
    # Attempt connection
    conn = psycopg2.connect(**conn_params)
    print("Successfully connected to PostgreSQL!")

    # Create a cursor (like a "pen" that executes queries)
    cursor = conn.cursor()

    # Run a simple test query
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f" PostgreSQL version: {version[0]}")

    # Close cursor and connection
    cursor.close()
    conn.close()
    print("Connection closed cleanly")

except Exception as e:
    print(f"Connection failed: {e}")