"""
Robustness Experiment — Scenario A: Jump Outlier Contamination

Compares MLE vs LSTM OU parameter estimation under increasing levels of
jump contamination. For each contamination level, generates N synthetic OU
paths, adds jumps at that rate, then estimates (θ, σ) with both methods.

Hypothesis:
  - MLE is theoretically optimal on clean OU data (Cramér-Rao) but uses a
    Gaussian likelihood — large residuals from jumps inflate σ̂ and bias θ̂.
  - LSTM may degrade more gracefully: its LSTM gating can suppress isolated
    spikes; it was not trained on jumps so it will degrade too, but possibly
    more slowly.

Output:
  A table of θ MAE / σ MAE for each method at each contamination level,
  plus a brief narrative conclusion for the thesis.

Usage:
  python estimation/robustness_experiment.py [--n-paths N] [--model PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.ou_generator import generate_ou_process, add_jump_contamination
from estimation.mle import estimate_ou_mle
from estimation.lstm_estimator import OULSTMEstimator

# ── Experiment parameters ────────────────────────────────────────────────────

TRUE_THETA = 0.05       # Mean-reversion speed  (half-life ≈ 13.9 days)
TRUE_MU    = 0.0        # Long-term mean
TRUE_SIGMA = 1.0        # Volatility

JUMP_RATES  = [0.00, 0.02, 0.05, 0.10]  # 0%, 2%, 5%, 10%
JUMP_SCALE  = 5.0       # Jumps drawn from N(0, JUMP_SCALE * sigma_est)

PATH_LENGTH = 200       # Observations per path (≈ 9 months of daily data)
SEED_BASE   = 2025      # Base RNG seed; each path gets SEED_BASE + path_idx

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'saved_models', 'ou_lstm_v1.pt'
)

# ── Core estimation helpers ──────────────────────────────────────────────────

def run_mle(series: pd.Series) -> tuple[float, float]:
    """Return (theta_hat, sigma_hat) from MLE, or (nan, nan) on failure."""
    try:
        result = estimate_ou_mle(series, verbose=False)
        if result.theta > 0 and result.sigma > 0:
            return result.theta, result.sigma
        return float('nan'), float('nan')
    except Exception:
        return float('nan'), float('nan')


def run_lstm(estimator: OULSTMEstimator, series: pd.Series) -> tuple[float, float]:
    """Return (theta_hat, sigma_hat) from LSTM MC-mean, or (nan, nan) on failure."""
    try:
        result = estimator.estimate(series, n_mc_samples=50)  # 50 draws → fast
        return result.theta, result.sigma
    except Exception:
        return float('nan'), float('nan')


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(n_paths: int, model_path: str) -> pd.DataFrame:
    """
    For each (contamination level, method), estimate on n_paths synthetic paths
    and compute θ MAE, θ RMSE, σ MAE, σ RMSE.

    Returns a DataFrame with columns:
        jump_rate | method | theta_mae | theta_rmse | sigma_mae | sigma_rmse
    """
    # Load LSTM model
    print(f"Loading LSTM model from: {model_path}", flush=True)
    lstm = OULSTMEstimator()
    lstm.load(model_path)

    records = []

    for jump_rate in JUMP_RATES:
        label = f"{int(jump_rate * 100):2d}%"
        print(f"\nJump rate {label} — estimating {n_paths} paths ...", flush=True)

        mle_theta_errs, mle_sigma_errs = [], []
        lstm_theta_errs, lstm_sigma_errs = [], []

        for i in range(n_paths):
            seed = SEED_BASE + i
            path = generate_ou_process(
                mu=TRUE_MU, theta=TRUE_THETA, sigma=TRUE_SIGMA,
                n_steps=PATH_LENGTH, dt=1.0, initial_value=0.0,
                seed=seed
            )

            contaminated = add_jump_contamination(
                path, jump_rate=jump_rate, jump_scale=JUMP_SCALE,
                seed=seed + 10000
            )

            # MLE
            t_mle, s_mle = run_mle(contaminated)
            if not (np.isnan(t_mle) or np.isnan(s_mle)):
                mle_theta_errs.append(abs(t_mle - TRUE_THETA))
                mle_sigma_errs.append(abs(s_mle - TRUE_SIGMA))

            # LSTM
            t_lstm, s_lstm = run_lstm(lstm, contaminated)
            if not (np.isnan(t_lstm) or np.isnan(s_lstm)):
                lstm_theta_errs.append(abs(t_lstm - TRUE_THETA))
                lstm_sigma_errs.append(abs(s_lstm - TRUE_SIGMA))

        def _stats(errs):
            if not errs:
                return float('nan'), float('nan')
            a = np.array(errs)
            return float(np.mean(a)), float(np.sqrt(np.mean(a ** 2)))

        mle_t_mae,  mle_t_rmse  = _stats(mle_theta_errs)
        mle_s_mae,  mle_s_rmse  = _stats(mle_sigma_errs)
        lstm_t_mae, lstm_t_rmse = _stats(lstm_theta_errs)
        lstm_s_mae, lstm_s_rmse = _stats(lstm_sigma_errs)

        print(
            f"  MLE  — θ MAE={mle_t_mae:.4f}  θ RMSE={mle_t_rmse:.4f}  "
            f"σ MAE={mle_s_mae:.4f}  σ RMSE={mle_s_rmse:.4f}  "
            f"(n={len(mle_theta_errs)})",
            flush=True
        )
        print(
            f"  LSTM — θ MAE={lstm_t_mae:.4f}  θ RMSE={lstm_t_rmse:.4f}  "
            f"σ MAE={lstm_s_mae:.4f}  σ RMSE={lstm_s_rmse:.4f}  "
            f"(n={len(lstm_theta_errs)})",
            flush=True
        )

        records.append(dict(
            jump_rate=jump_rate, method='MLE',
            theta_mae=mle_t_mae,  theta_rmse=mle_t_rmse,
            sigma_mae=mle_s_mae,  sigma_rmse=mle_s_rmse,
            n_valid=len(mle_theta_errs),
        ))
        records.append(dict(
            jump_rate=jump_rate, method='LSTM',
            theta_mae=lstm_t_mae,  theta_rmse=lstm_t_rmse,
            sigma_mae=lstm_s_mae,  sigma_rmse=lstm_s_rmse,
            n_valid=len(lstm_theta_errs),
        ))

    return pd.DataFrame(records)


def print_summary_table(df: pd.DataFrame):
    """Print a formatted comparison table suitable for the thesis."""
    print("\n")
    print("=" * 78)
    print("ROBUSTNESS EXPERIMENT — Scenario A: Jump Contamination")
    print(f"True: θ={TRUE_THETA}, μ={TRUE_MU}, σ={TRUE_SIGMA}  |  "
          f"Jump scale={JUMP_SCALE}×σ  |  Path length={PATH_LENGTH}")
    print("=" * 78)
    print(f"{'Jump %':>8}  {'Method':>6}  {'θ MAE':>9}  {'θ RMSE':>9}  "
          f"{'σ MAE':>9}  {'σ RMSE':>9}  {'n':>5}")
    print("-" * 78)

    for _, row in df.iterrows():
        pct = f"{row['jump_rate'] * 100:.0f}%"
        print(
            f"{pct:>8}  {row['method']:>6}  "
            f"{row['theta_mae']:>9.4f}  {row['theta_rmse']:>9.4f}  "
            f"{row['sigma_mae']:>9.4f}  {row['sigma_rmse']:>9.4f}  "
            f"{int(row['n_valid']):>5}"
        )
        if row['method'] == 'LSTM':
            print()  # blank line between contamination levels

    print("=" * 78)

    # Narrative
    clean_mle  = df[(df.jump_rate == 0.00) & (df.method == 'MLE')].iloc[0]
    clean_lstm = df[(df.jump_rate == 0.00) & (df.method == 'LSTM')].iloc[0]
    high_mle   = df[(df.jump_rate == JUMP_RATES[-1]) & (df.method == 'MLE')].iloc[0]
    high_lstm  = df[(df.jump_rate == JUMP_RATES[-1]) & (df.method == 'LSTM')].iloc[0]

    mle_theta_degradation  = (high_mle['theta_mae']  - clean_mle['theta_mae'])  / (clean_mle['theta_mae']  + 1e-9)
    lstm_theta_degradation = (high_lstm['theta_mae'] - clean_lstm['theta_mae']) / (clean_lstm['theta_mae'] + 1e-9)
    mle_sigma_degradation  = (high_mle['sigma_mae']  - clean_mle['sigma_mae'])  / (clean_mle['sigma_mae']  + 1e-9)
    lstm_sigma_degradation = (high_lstm['sigma_mae'] - clean_lstm['sigma_mae']) / (clean_lstm['sigma_mae'] + 1e-9)

    print("\nNARRATIVE CONCLUSION")
    print("-" * 78)
    print(f"  At 0% contamination:")
    print(f"    MLE  θ MAE={clean_mle['theta_mae']:.4f}  σ MAE={clean_mle['sigma_mae']:.4f}")
    print(f"    LSTM θ MAE={clean_lstm['theta_mae']:.4f}  σ MAE={clean_lstm['sigma_mae']:.4f}")
    print(f"  At {JUMP_RATES[-1]*100:.0f}% contamination:")
    print(f"    MLE  θ MAE={high_mle['theta_mae']:.4f} (+{mle_theta_degradation*100:.0f}%)"
          f"  σ MAE={high_mle['sigma_mae']:.4f} (+{mle_sigma_degradation*100:.0f}%)")
    print(f"    LSTM θ MAE={high_lstm['theta_mae']:.4f} (+{lstm_theta_degradation*100:.0f}%)"
          f"  σ MAE={high_lstm['sigma_mae']:.4f} (+{lstm_sigma_degradation*100:.0f}%)")

    if lstm_sigma_degradation < mle_sigma_degradation:
        advantage = (mle_sigma_degradation - lstm_sigma_degradation) * 100
        print(f"\n  RESULT: LSTM degrades {advantage:.0f}% less than MLE on σ under jump contamination.")
        print(f"  Conclusion: LSTM shows robustness advantage when the Gaussian OU assumption is violated.")
    else:
        print(f"\n  RESULT: MLE and LSTM degrade similarly under jump contamination.")
        print(f"  Conclusion: No robustness advantage observed for LSTM on this contamination type.")
    print("=" * 78)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OU estimator robustness experiment — jump contamination")
    parser.add_argument('--n-paths', type=int, default=200,
                        help='Number of synthetic paths per contamination level (default: 200)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to trained LSTM .pt file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: LSTM model not found at {args.model}")
        print("Run 'python estimation/train_lstm.py' first.")
        sys.exit(1)

    print("=" * 78)
    print(f"ROBUSTNESS EXPERIMENT — Scenario A: Jump Contamination")
    print(f"Paths per level: {args.n_paths}  |  Jump rates: {JUMP_RATES}")
    print("=" * 78)

    df = run_experiment(n_paths=args.n_paths, model_path=args.model)
    print_summary_table(df)

    # Save results to CSV next to the script
    out_path = os.path.join(os.path.dirname(__file__), 'robustness_results_jumps.csv')
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
