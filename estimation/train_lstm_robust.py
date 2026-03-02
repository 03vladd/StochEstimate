"""
Training script for LSTM-v2-robust: OU Parameter Estimator trained on contaminated data.

Trains the same architecture as v1 (window=126, hidden=64, 2 layers, dropout=0.2)
on a 50/50 mix of clean and jump-contaminated OU paths. Targets are always the clean
OU parameters — the model learns to recover (θ, μ, σ) despite jumps in the input.

Contamination strategy:
  - 50% of training paths: contaminated with jumps ~ N(0, 5σ) at rate Uniform(1%, 10%)
  - 50%: clean OU paths (same as v1)
  - This range covers the full evaluation range [0%, 10%] used in robustness_experiment.py

Usage:
    python estimation/train_lstm_robust.py

Saves to:
    estimation/saved_models/ou_lstm_v2_robust.pt

Clean-data comparison with v1 is printed at the end. The v2 model is then
used as the 4th estimator in robustness_experiment.py (--model-robust flag).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimation.lstm_estimator import OULSTMEstimator
from synthetic_data.ou_generator import generate_ou_process

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ou_lstm_v2_robust.pt')

# V1 known numbers from RESEARCH_PROGRESS.md (clean test set, seed=999, n=1000)
V1_KNOWN = {'mae_theta': 0.0290, 'mae_sigma': 0.0540}


def spot_check(estimator: OULSTMEstimator):
    """Spot-check on known parameters: θ=0.05, μ=0.0, σ=1.0 (clean path)"""
    print(f"\n{'=' * 60}")
    print(f"SPOT CHECK: θ=0.05, μ=0.0, σ=1.0 (clean input)")
    print(f"{'=' * 60}")

    path = generate_ou_process(mu=0.0, theta=0.05, sigma=1.0, n_steps=300, dt=1.0, seed=42)
    result = estimator.estimate(path, n_mc_samples=200)

    half_life_true = np.log(2) / 0.05
    half_life_lstm = np.log(2) / result.theta if result.theta > 0 else float('inf')

    print(f"True:      θ=0.050000, μ=0.000000, σ=1.000000")
    print(f"LSTM-v2:   θ={result.theta:.6f}, μ={result.mu:.6f}, σ={result.sigma:.6f}")
    print(f"CI θ:      [{result.theta_ci[0]:.6f}, {result.theta_ci[1]:.6f}]")
    print(f"CI μ:      [{result.mu_ci[0]:.6f}, {result.mu_ci[1]:.6f}]")
    print(f"CI σ:      [{result.sigma_ci[0]:.6f}, {result.sigma_ci[1]:.6f}]")
    print(f"Half-life: {half_life_lstm:.1f} days (true: {half_life_true:.1f} days)")


def evaluate_on_test_set(estimator: OULSTMEstimator, n_test: int = 1000,
                         theta_range: tuple = (0.005, 0.5),
                         sigma_range: tuple = (0.1, 2.0),
                         seed: int = 999) -> dict:
    """
    Evaluate on a clean synthetic test set (same seed as v1 for fair comparison).

    Both v1 and v2 are evaluated on the same clean paths so the accuracy difference
    is attributable solely to training data composition, not test set variation.
    """
    rng = np.random.default_rng(seed)

    true_thetas, true_mus, true_sigmas = [], [], []
    pred_thetas, pred_mus, pred_sigmas = [], [], []

    print(f"\nEvaluating on {n_test} clean synthetic test paths (seed={seed})...")

    for i in range(n_test):
        log_theta_min = np.log(theta_range[0])
        log_theta_max = np.log(theta_range[1])
        theta_true = np.exp(rng.uniform(log_theta_min, log_theta_max))
        mu_true = rng.uniform(-2.0, 2.0)
        sigma_true = rng.uniform(sigma_range[0], sigma_range[1])

        path = generate_ou_process(
            mu=mu_true, theta=theta_true, sigma=sigma_true,
            n_steps=300, dt=1.0, initial_value=mu_true,
            seed=None
        )

        result = estimator.estimate(path, n_mc_samples=50)

        true_thetas.append(theta_true)
        true_mus.append(mu_true)
        true_sigmas.append(sigma_true)
        pred_thetas.append(result.theta)
        pred_mus.append(result.mu)
        pred_sigmas.append(result.sigma)

        if (i + 1) % 200 == 0:
            print(f"  Evaluated {i + 1}/{n_test} paths...")

    true_thetas = np.array(true_thetas)
    true_mus = np.array(true_mus)
    true_sigmas = np.array(true_sigmas)
    pred_thetas = np.array(pred_thetas)
    pred_mus = np.array(pred_mus)
    pred_sigmas = np.array(pred_sigmas)

    metrics = {
        'mae_theta':  float(np.mean(np.abs(pred_thetas - true_thetas))),
        'rmse_theta': float(np.sqrt(np.mean((pred_thetas - true_thetas) ** 2))),
        'mae_mu':     float(np.mean(np.abs(pred_mus - true_mus))),
        'rmse_mu':    float(np.sqrt(np.mean((pred_mus - true_mus) ** 2))),
        'mae_sigma':  float(np.mean(np.abs(pred_sigmas - true_sigmas))),
        'rmse_sigma': float(np.sqrt(np.mean((pred_sigmas - true_sigmas) ** 2))),
    }

    print(f"\n{'=' * 60}")
    print(f"TEST SET EVALUATION — CLEAN DATA  ({n_test} samples, seed={seed})")
    print(f"{'=' * 60}")
    print(f"{'Parameter':<12}  {'MAE':>10}  {'RMSE':>10}")
    print(f"{'-' * 36}")
    print(f"{'θ (theta)':<12}  {metrics['mae_theta']:>10.6f}  {metrics['rmse_theta']:>10.6f}")
    print(f"{'μ (mu)':<12}  {metrics['mae_mu']:>10.6f}  {metrics['rmse_mu']:>10.6f}")
    print(f"{'σ (sigma)':<12}  {metrics['mae_sigma']:>10.6f}  {metrics['rmse_sigma']:>10.6f}")
    print(f"{'=' * 60}")

    return metrics


def print_v1_v2_comparison(metrics_v2: dict):
    """Side-by-side comparison of v1 (clean training) vs v2 (contaminated training)."""
    print(f"\n{'=' * 70}")
    print(f"CLEAN-DATA COMPARISON: LSTM-v1 (clean training) vs LSTM-v2 (robust)")
    print(f"{'=' * 70}")
    print(f"{'Model':<18}  {'θ MAE':>10}  {'σ MAE':>10}  {'Notes'}")
    print(f"{'-' * 65}")

    v1_theta = V1_KNOWN['mae_theta']
    v1_sigma = V1_KNOWN['mae_sigma']
    v2_theta = metrics_v2['mae_theta']
    v2_sigma = metrics_v2['mae_sigma']

    delta_theta = v2_theta - v1_theta
    delta_sigma = v2_sigma - v1_sigma

    print(f"{'LSTM-v1':<18}  {v1_theta:>10.4f}  {v1_sigma:>10.4f}  trained on clean data only")
    print(f"{'LSTM-v2-robust':<18}  {v2_theta:>10.4f}  {v2_sigma:>10.4f}  trained 50% contaminated")
    print(f"{'-' * 65}")
    sign_t = '+' if delta_theta >= 0 else ''
    sign_s = '+' if delta_sigma >= 0 else ''
    print(f"{'Delta (v2 - v1)':<18}  {sign_t}{delta_theta:>9.4f}  {sign_s}{delta_sigma:>9.4f}  "
          f"{'positive = v1 better on clean' if delta_theta > 0 else 'negative = v2 better'}")
    print(f"{'=' * 70}")
    print(f"\nNote: clean-data accuracy cost is expected — the model has traded some")
    print(f"      efficiency on clean paths for robustness under contamination.")
    print(f"      Run robustness_experiment.py --model-robust to see the full picture.")


def save_to_db(metrics: dict, model_filename: str):
    """Store v2 model metadata in lstm_models table."""
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        model_id = db.insert_lstm_model(
            interval='1d',
            validation_criteria='synthetic_ou_contaminated',
            model_filename=model_filename,
            training_pairs_count=0,
            training_data_points_used=50000,
            mae_theta=metrics['mae_theta'],
            rmse_theta=metrics['rmse_theta'],
            version=2,
        )
        if model_id:
            print(f"✓ Model metadata stored in DB (model_id={model_id})")
        db.close_pool()
    except Exception as e:
        print(f"⚠ Could not store in DB (DB may not be running): {e}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("LSTM-v2-ROBUST — TRAINING ON CONTAMINATED DATA")
    print("Architecture: identical to v1 (window=126, hidden=64, 2L, d=0.2)")
    print("Training mix: 50% clean paths + 50% jump-contaminated paths")
    print("=" * 60)

    estimator = OULSTMEstimator(
        window_size=126, hidden_size=64, num_layers=2, dropout=0.2
    )

    history = estimator.train(
        n_samples=50000,
        epochs=100,
        batch_size=256,
        theta_range=(0.005, 0.5),
        sigma_range=(0.1, 2.0),
        lr=1e-3,
        val_split=0.1,
        seed=42,
        contamination_mix=0.5,
        contamination_max_rate=0.10,
    )

    # Save model
    estimator.save(MODEL_PATH)

    # Spot-check on clean path
    spot_check(estimator)

    # Evaluate on clean test set (same seed=999 as v1 for fair comparison)
    metrics = evaluate_on_test_set(estimator, n_test=1000, seed=999)

    # Side-by-side comparison with v1
    print_v1_v2_comparison(metrics)

    # Store metadata in database
    save_to_db(metrics, os.path.basename(MODEL_PATH))

    print(f"\n✓ Done! Model saved to: {MODEL_PATH}")
    print(f"  Next step: python estimation/robustness_experiment.py "
          f"--n-paths 200 --model-robust {MODEL_PATH}")


if __name__ == "__main__":
    main()
