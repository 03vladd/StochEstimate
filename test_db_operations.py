"""
Test the DatabaseManager CRUD operations
"""

from database.db_manager import DatabaseManager
from datetime import datetime

# Initialize database manager
print("Initializing database manager...")
db = DatabaseManager()

# Test 1: Insert a pair
print("\n" + "="*70)
print("TEST 1: Insert a pair")
print("="*70)
pair_id = db.insert_pair('JPM', 'GS', 'JPMorgan vs Goldman Sachs', 'Financial')
print(f"✓ Inserted pair with ID: {pair_id}")

# Test 2: Insert cointegration results
print("\n" + "="*70)
print("TEST 2: Insert cointegration results")
print("="*70)
coint_id = db.insert_cointegration_result(
    pair_id=pair_id,
    interval='1d',
    hedge_ratio=0.8234,
    intercept=5.1234,
    adf_statistic=-3.456,
    adf_pvalue=0.0012,
    cointegrated=True,
    observations_used=1500
)
print(f"✓ Inserted cointegration result with ID: {coint_id}")

# Test 3: Insert validation results
print("\n" + "="*70)
print("TEST 3: Insert validation results")
print("="*70)
val_id = db.insert_validation_result(
    pair_id=pair_id,
    interval='1d',
    coint_id=coint_id,
    stationarity_passed=True,
    stationarity_pvalue=0.001,
    stationarity_statistic=-3.456,
    drift_passed=True,
    drift_pvalue=0.045,
    drift_statistic=1.234,
    volatility_passed=True,
    volatility_pvalue=0.123,
    volatility_statistic=0.987,
    autocorr_passed=True,
    autocorr_pvalue=0.089,
    autocorr_statistic=0.654,
    normality_passed=False,
    normality_pvalue=0.156,
    normality_statistic=2.345,
    confidence_level='HIGH',
    tests_passed_count=4
)
print(f"✓ Inserted validation result with ID: {val_id}")

# Test 4: Insert price data
print("\n" + "="*70)
print("TEST 4: Insert price data")
print("="*70)
price_id = db.insert_price_data(
    pair_id=pair_id,
    interval='1d',
    timestamp='2024-12-04 16:00:00',
    close=150.25,
    open=149.50,
    high=151.00,
    low=149.00,
    volume=1000000
)
print(f"✓ Inserted price data with ID: {price_id}")

# Test 5: Query validation results
print("\n" + "="*70)
print("TEST 5: Query validation results")
print("="*70)
results = db.get_validation_results(confidence_level='HIGH')
print(f"✓ Found {len(results)} HIGH confidence results")
for result in results:
    print(f"  Pair ID: {result['pair_id']}")
    print(f"  Confidence: {result['confidence_level']}")
    print(f"  Tests Passed: {result['tests_passed_count']}/5")

# Test 6: Query specific pair
print("\n" + "="*70)
print("TEST 6: Query validation results for specific pair")
print("="*70)
results = db.get_validation_results(pair_id=pair_id, interval='1d')
print(f"✓ Found {len(results)} results for pair_id={pair_id}, interval='1d'")
if results:
    result = results[0]
    print(f"  Stationarity: {'✓' if result['stationarity_passed'] else '✗'}")
    print(f"  Drift: {'✓' if result['drift_passed'] else '✗'}")
    print(f"  Volatility: {'✓' if result['volatility_passed'] else '✗'}")
    print(f"  Autocorr: {'✓' if result['autocorr_passed'] else '✗'}")
    print(f"  Normality: {'✓' if result['normality_passed'] else '✗'}")

# Close connection pool
print("\n" + "="*70)
db.close_pool()
print("✓ All tests passed!")
print("="*70)