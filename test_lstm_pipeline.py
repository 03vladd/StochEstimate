"""
Full pipeline smoke test — LSTM integration
Run this before committing to verify everything works end-to-end.

What it does:
  1. Quick-trains the LSTM (500 samples, 3 epochs — rough but fast)
  2. Saves model to estimation/saved_models/ou_lstm_smoke.pt
  3. Loads the cointegrated pairs already cached in the DB
  4. Drives Phase 2 (Validation → MLE → LSTM → Backtest) on those pairs
  5. Prints the comparison table and summary
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimation.lstm_estimator import OULSTMEstimator
from pipeline import HybridPipeline
from database.db_manager import DatabaseManager

FULL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'estimation', 'saved_models', 'ou_lstm_v1.pt'
)
SMOKE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'estimation', 'saved_models', 'ou_lstm_smoke.pt'
)


def get_model_path() -> str:
    """Use the full trained model if available, otherwise quick-train a smoke model."""
    if os.path.exists(FULL_MODEL_PATH):
        print("=" * 70)
        print(f"STEP 1: Using existing trained model: {FULL_MODEL_PATH}")
        print("=" * 70)
        return FULL_MODEL_PATH

    print("=" * 70)
    print("STEP 1: No trained model found — quick-training (500 samples, 3 epochs)")
    print("=" * 70)
    estimator = OULSTMEstimator(window_size=126, hidden_size=64, num_layers=2, dropout=0.2)
    estimator.train(
        n_samples=500, epochs=3, batch_size=64,
        theta_range=(0.005, 0.5), sigma_range=(0.1, 2.0),
        lr=1e-3, val_split=0.1, seed=42,
    )
    os.makedirs(os.path.dirname(SMOKE_MODEL_PATH), exist_ok=True)
    estimator.save(SMOKE_MODEL_PATH)
    return SMOKE_MODEL_PATH


def load_cointegrated_pairs_from_db(db: DatabaseManager):
    """
    Fetch HIGH and MEDIUM confidence pairs from the DB.
    Returns list of (ticker1, ticker2, pair_name).
    """
    rows = db.execute_query("""
        SELECT p.ticker1, p.ticker2, p.pair_name
        FROM pairs p
        JOIN validation_results v ON p.pair_id = v.pair_id
        WHERE v.confidence_level IN ('HIGH', 'MEDIUM')
        AND v.interval = '1d'
        ORDER BY v.tests_passed_count DESC
    """)
    return [(r['ticker1'], r['ticker2'], r['pair_name']) for r in rows]


def main():
    # Step 1: Get model (existing or quick-trained)
    model_path = get_model_path()

    # Step 2: Build pipeline with LSTM model
    print("\n" + "=" * 70)
    print("STEP 2: Initialising pipeline with LSTM estimator")
    print("=" * 70)

    # Use same date window as cached data (2023-12-05 → 2025-12-03)
    pipeline = HybridPipeline(
        start_date='2023-12-05',
        end_date='2025-12-03',
        lstm_model_path=model_path,
    )

    # Step 3: Load cointegrated pairs from DB (skip Phase 1 brute force)
    print("\n" + "=" * 70)
    print("STEP 3: Loading cointegrated pairs from DB (skipping Phase 1)")
    print("=" * 70)

    pairs = load_cointegrated_pairs_from_db(pipeline.db)
    if not pairs:
        print("No HIGH/MEDIUM confidence pairs found in DB. Run pipeline.py Phase 1 first.")
        pipeline.db.close_pool()
        return

    pipeline.cointegrated_pairs = pairs
    print(f"Found {len(pairs)} pairs to analyse:")
    for t1, t2, name in pairs:
        print(f"  {name} ({t1} / {t2})")

    # Step 4: Phase 2 — full analysis (Validation + MLE + LSTM + Backtest)
    pipeline.phase_2_full_analysis()

    # Step 5: Summary table
    pipeline.print_phase_2_summary()

    pipeline.db.close_pool()
    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE — no errors above means pipeline is ready to commit")
    print("=" * 70)


if __name__ == "__main__":
    main()
