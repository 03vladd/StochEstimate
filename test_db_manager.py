"""
Test the DatabaseManager connection and basic operations
"""

from database.db_manager import DatabaseManager

# Initialize the database manager
print("Initializing database manager...")
db = DatabaseManager()

# Test 1: Insert a pair
print("\nTest 1: Inserting a pair...")
pair_id = db.execute_insert(
    "INSERT INTO pairs (ticker1, ticker2, pair_name, sector) VALUES (%s, %s, %s, %s) RETURNING pair_id",
    ('JPM', 'GS', 'JPMorgan vs Goldman Sachs', 'Financial')
)
print(f"✓ Inserted pair with ID: {pair_id}")

# Test 2: Query the pair back
print("\nTest 2: Querying the pair...")
result = db.execute_query(
    "SELECT * FROM pairs WHERE pair_id = %s",
    (pair_id,)
)
print(f"✓ Retrieved pair: {result}")

# Test 3: Insert price data
print("\nTest 3: Inserting price data...")
price_id = db.execute_insert(
    "INSERT INTO price_data (pair_id, interval, timestamp, close, open, high, low, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING price_id",
    (pair_id, '1d', '2024-12-04 16:00:00', 150.25, 149.50, 151.00, 149.00, 1000000)
)
print(f"✓ Inserted price data with ID: {price_id}")

# Test 4: Query all pairs
print("\nTest 4: Querying all pairs...")
all_pairs = db.execute_query("SELECT * FROM pairs")
print(f"✓ Total pairs in database: {len(all_pairs)}")
for pair in all_pairs:
    print(f"  - {pair['pair_name']}")

# Close the connection pool
print("\nClosing database connection pool...")
db.close_pool()
print("✓ Test complete!")