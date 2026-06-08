"""
Calibration Coverage Check

Tests whether the 95% confidence intervals from MLE and LSTM (MC Dropout)
actually achieve ~95% empirical coverage on held-out synthetic OU paths.

Setup:
  - 500 OU paths: θ=0.05, μ=0.0, σ=1.0
  - Seeds 9000–9499 (disjoint from training seed=42 and eval seed=999/2025+)
  - Path length: 200 (matches robustness_experiment.py)

For each path: checks whether the true parameter falls inside the reported 95% CI.

Usage:
    python estimation/calibration_check.py [--n-paths N] [--mc-samples N]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.ou_generator import generate_ou_process, add_jump_contamination
from estimation.mle import estimate_ou_mle
from estimation.mle_robust import estimate_ou_t_mle, estimate_ou_t_mle_adaptive
from estimation.lstm_estimator import OULSTMEstimator
from estimation.conformal_estimator import ConformalOUEstimator

JUMP_RATES = [0.00, 0.02, 0.05, 0.10]
JUMP_SCALE = 5.0
ROBUSTNESS_SEED_BASE = 2025   # same seeds as robustness_experiment.py

TRUE_THETA = 0.05
TRUE_MU    = 0.0
TRUE_SIGMA = 1.0
PATH_LENGTH = 200
SEED_START  = 9000   # disjoint from train seed=42, eval seeds 999 and 2025+

DEFAULT_V1_PATH     = os.path.join(os.path.dirname(__file__), 'saved_models', 'ou_lstm_v1.pt')
DEFAULT_ROBUST_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'ou_lstm_v2_robust.pt')


def run_coverage_check(n_paths: int = 500, mc_samples: int = 50) -> None:
    print("=" * 65)
    print(f"CALIBRATION COVERAGE CHECK")
    print(f"  n_paths={n_paths}, θ={TRUE_THETA}, μ={TRUE_MU}, σ={TRUE_SIGMA}")
    print(f"  path_length={PATH_LENGTH}, seeds {SEED_START}–{SEED_START + n_paths - 1}")
    print(f"  MC samples per LSTM call: {mc_samples}")
    print("=" * 65)

    # ── Load LSTM models ──────────────────────────────────────────────────────
    lstm_v1 = None
    lstm_rob = None

    if os.path.exists(DEFAULT_V1_PATH):
        lstm_v1 = OULSTMEstimator(window_size=126, hidden_size=64, num_layers=2, dropout=0.2)
        lstm_v1.load(DEFAULT_V1_PATH)
        print(f"✓ Loaded LSTM-v1 from {DEFAULT_V1_PATH}")
    else:
        print(f"⚠ LSTM-v1 not found at {DEFAULT_V1_PATH} — skipping")

    if os.path.exists(DEFAULT_ROBUST_PATH):
        lstm_rob = OULSTMEstimator(window_size=126, hidden_size=64, num_layers=2, dropout=0.2)
        lstm_rob.load(DEFAULT_ROBUST_PATH)
        print(f"✓ Loaded LSTM-robust from {DEFAULT_ROBUST_PATH}")
    else:
        print(f"⚠ LSTM-robust not found at {DEFAULT_ROBUST_PATH} — skipping")

    # ── Counters ──────────────────────────────────────────────────────────────
    mle_covered  = {'theta': 0, 'mu': 0, 'sigma': 0}
    v1_covered   = {'theta': 0, 'mu': 0, 'sigma': 0}
    rob_covered  = {'theta': 0, 'mu': 0, 'sigma': 0}

    mle_valid = 0   # paths where MLE converged

    print(f"\nRunning {n_paths} paths...")

    for i in range(n_paths):
        seed = SEED_START + i
        path = generate_ou_process(
            mu=TRUE_MU, theta=TRUE_THETA, sigma=TRUE_SIGMA,
            n_steps=PATH_LENGTH, dt=1.0,
            initial_value=TRUE_MU, seed=seed
        )

        # ── MLE ──────────────────────────────────────────────────────────────
        try:
            r = estimate_ou_mle(path, verbose=False)
            if r.success and r.theta > 0 and r.sigma > 0:
                mle_valid += 1
                if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                    mle_covered['theta'] += 1
                if r.mu_ci[0] <= TRUE_MU <= r.mu_ci[1]:
                    mle_covered['mu'] += 1
                if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                    mle_covered['sigma'] += 1
        except Exception:
            pass

        # ── LSTM-v1 ───────────────────────────────────────────────────────────
        if lstm_v1 is not None:
            try:
                r = lstm_v1.estimate(path, n_mc_samples=mc_samples)
                if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                    v1_covered['theta'] += 1
                if r.mu_ci[0] <= TRUE_MU <= r.mu_ci[1]:
                    v1_covered['mu'] += 1
                if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                    v1_covered['sigma'] += 1
            except Exception:
                pass

        # ── LSTM-robust ───────────────────────────────────────────────────────
        if lstm_rob is not None:
            try:
                r = lstm_rob.estimate(path, n_mc_samples=mc_samples)
                if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                    rob_covered['theta'] += 1
                if r.mu_ci[0] <= TRUE_MU <= r.mu_ci[1]:
                    rob_covered['mu'] += 1
                if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                    rob_covered['sigma'] += 1
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_paths} paths complete...")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"RESULTS — empirical 95% CI coverage (nominal target: 95.0%)")
    print(f"{'=' * 65}")
    print(f"{'Estimator':<18}  {'θ coverage':>12}  {'μ coverage':>12}  {'σ coverage':>12}")
    print(f"{'-' * 60}")

    pct = lambda hits, total: f"{100 * hits / total:.1f}%" if total > 0 else "N/A"

    print(f"{'MLE':<18}  {pct(mle_covered['theta'], mle_valid):>12}  "
          f"{pct(mle_covered['mu'], mle_valid):>12}  "
          f"{pct(mle_covered['sigma'], mle_valid):>12}  "
          f"(n_valid={mle_valid}/{n_paths})")

    if lstm_v1 is not None:
        print(f"{'LSTM-v1':<18}  {pct(v1_covered['theta'], n_paths):>12}  "
              f"{pct(v1_covered['mu'], n_paths):>12}  "
              f"{pct(v1_covered['sigma'], n_paths):>12}")

    if lstm_rob is not None:
        print(f"{'LSTM-robust':<18}  {pct(rob_covered['theta'], n_paths):>12}  "
              f"{pct(rob_covered['mu'], n_paths):>12}  "
              f"{pct(rob_covered['sigma'], n_paths):>12}")

    print(f"{'=' * 65}")
    print(f"\nInterpretation guide:")
    print(f"  90–100%  Well-calibrated — CIs are reliable")
    print(f"  80–90%   Slightly overconfident — report as limitation")
    print(f"  < 80%    Overconfident — revise UQ claim in thesis")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), 'calibration_coverage.csv')
    rows = []
    rows.append({'estimator': 'MLE', 'n': mle_valid,
                 'theta_coverage': mle_covered['theta'] / mle_valid if mle_valid else None,
                 'mu_coverage':    mle_covered['mu']    / mle_valid if mle_valid else None,
                 'sigma_coverage': mle_covered['sigma'] / mle_valid if mle_valid else None})
    if lstm_v1 is not None:
        rows.append({'estimator': 'LSTM-v1', 'n': n_paths,
                     'theta_coverage': v1_covered['theta'] / n_paths,
                     'mu_coverage':    v1_covered['mu']    / n_paths,
                     'sigma_coverage': v1_covered['sigma'] / n_paths})
    if lstm_rob is not None:
        rows.append({'estimator': 'LSTM-robust', 'n': n_paths,
                     'theta_coverage': rob_covered['theta'] / n_paths,
                     'mu_coverage':    rob_covered['mu']    / n_paths,
                     'sigma_coverage': rob_covered['sigma'] / n_paths})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n✓ Results saved to {out_path}")


def run_contamination_coverage_table(n_paths: int = 200, mc_samples: int = 50) -> None:
    """
    Coverage table across all 4 contamination levels, all 5 estimators.
    Uses seeds ROBUSTNESS_SEED_BASE+i and the same contamination protocol as
    robustness_experiment.py so results are directly comparable to the MAE tables.
    """
    print("=" * 72)
    print("CI COVERAGE TABLE — across contamination levels (all 5 estimators)")
    print(f"  n_paths={n_paths} per level, seeds {ROBUSTNESS_SEED_BASE}–{ROBUSTNESS_SEED_BASE+n_paths-1}")
    print(f"  θ*={TRUE_THETA}, σ*={TRUE_SIGMA}, path_length={PATH_LENGTH}")
    print(f"  MC samples per LSTM call: {mc_samples}")
    print("=" * 72)

    # Load models
    lstm_v1, lstm_rob = None, None
    if os.path.exists(DEFAULT_V1_PATH):
        lstm_v1 = OULSTMEstimator(window_size=126, hidden_size=64, num_layers=2, dropout=0.2)
        lstm_v1.load(DEFAULT_V1_PATH)
        print(f"✓ LSTM-v1 loaded")
    if os.path.exists(DEFAULT_ROBUST_PATH):
        lstm_rob = OULSTMEstimator(window_size=126, hidden_size=64, num_layers=2, dropout=0.2)
        lstm_rob.load(DEFAULT_ROBUST_PATH)
        print(f"✓ LSTM-robust loaded")

    METHOD_NAMES = ['MLE', 't-MLE', 't-MLE(ν*)', 'LSTM-v1', 'LSTM-robust']
    records = []

    for jump_rate in JUMP_RATES:
        covered = {m: {'theta': 0, 'sigma': 0} for m in METHOD_NAMES}
        valid   = {m: 0 for m in METHOD_NAMES}

        print(f"\nJump rate {int(jump_rate*100):2d}% — running {n_paths} paths ...", flush=True)

        for i in range(n_paths):
            seed = ROBUSTNESS_SEED_BASE + i
            path = generate_ou_process(
                mu=TRUE_MU, theta=TRUE_THETA, sigma=TRUE_SIGMA,
                n_steps=PATH_LENGTH, dt=1.0, initial_value=TRUE_MU, seed=seed
            )
            contaminated = add_jump_contamination(
                path, jump_rate=jump_rate, jump_scale=JUMP_SCALE, seed=seed + 10000
            )

            # MLE
            try:
                r = estimate_ou_mle(contaminated, verbose=False)
                if r.success and r.theta > 0 and r.sigma > 0:
                    valid['MLE'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['MLE']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['MLE']['sigma'] += 1
            except Exception:
                pass

            # t-MLE (ν=4)
            try:
                r = estimate_ou_t_mle(contaminated, df=4.0, verbose=False)
                if r.theta > 0 and r.sigma > 0:
                    valid['t-MLE'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['t-MLE']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['t-MLE']['sigma'] += 1
            except Exception:
                pass

            # t-MLE (ν*)
            try:
                r = estimate_ou_t_mle_adaptive(contaminated, verbose=False)
                if r.theta > 0 and r.sigma > 0:
                    valid['t-MLE(ν*)'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['t-MLE(ν*)']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['t-MLE(ν*)']['sigma'] += 1
            except Exception:
                pass

            # LSTM-v1
            if lstm_v1 is not None:
                try:
                    r = lstm_v1.estimate(contaminated, n_mc_samples=mc_samples)
                    valid['LSTM-v1'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['LSTM-v1']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['LSTM-v1']['sigma'] += 1
                except Exception:
                    pass

            # LSTM-robust
            if lstm_rob is not None:
                try:
                    r = lstm_rob.estimate(contaminated, n_mc_samples=mc_samples)
                    valid['LSTM-robust'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['LSTM-robust']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['LSTM-robust']['sigma'] += 1
                except Exception:
                    pass

        # Print row
        pct = lambda hits, n: f"{100*hits/n:.1f}" if n > 0 else "N/A"
        print(f"  {'Method':<14}  {'θ cov%':>7}  {'σ cov%':>7}  {'n_valid':>7}")
        for m in METHOD_NAMES:
            if valid[m] == 0:
                continue
            print(f"  {m:<14}  {pct(covered[m]['theta'], valid[m]):>7}  "
                  f"{pct(covered[m]['sigma'], valid[m]):>7}  {valid[m]:>7}")

        for m in METHOD_NAMES:
            if valid[m] == 0:
                continue
            records.append({
                'jump_rate': jump_rate,
                'method': m,
                'theta_coverage': covered[m]['theta'] / valid[m],
                'sigma_coverage': covered[m]['sigma'] / valid[m],
                'n_valid': valid[m],
            })

    # Summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY — empirical θ coverage % (nominal: 95%)")
    print(f"{'=' * 72}")
    df = pd.DataFrame(records)
    methods_present = [m for m in METHOD_NAMES if m in df['method'].values]
    header = f"{'ε':>5}  " + "  ".join(f"{m:>14}" for m in methods_present)
    print(header)
    print("-" * 72)
    for jr in JUMP_RATES:
        row_str = f"{int(jr*100):>4}%  "
        for m in methods_present:
            sub = df[(df.jump_rate == jr) & (df.method == m)]
            if len(sub):
                row_str += f"  {sub.iloc[0]['theta_coverage']*100:>13.1f}%"
            else:
                row_str += f"  {'N/A':>13}"
        print(row_str)
    print(f"\n{'=' * 72}")
    print("SUMMARY — empirical σ coverage % (nominal: 95%)")
    print(f"{'=' * 72}")
    print(header)
    print("-" * 72)
    for jr in JUMP_RATES:
        row_str = f"{int(jr*100):>4}%  "
        for m in methods_present:
            sub = df[(df.jump_rate == jr) & (df.method == m)]
            if len(sub):
                row_str += f"  {sub.iloc[0]['sigma_coverage']*100:>13.1f}%"
            else:
                row_str += f"  {'N/A':>13}"
        print(row_str)
    print(f"{'=' * 72}")

    out_path = os.path.join(os.path.dirname(__file__), 'calibration_coverage_contaminated.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Results saved to {out_path}")


def run_conformal_coverage_table(n_cal: int = 500, n_test: int = 200,
                                  mc_samples: int = 50) -> None:
    """
    Coverage comparison: MLE Wald vs LSTM-robust MC Dropout vs LSTM-robust Conformal
    across all 4 contamination levels.

    Calibration set : n_cal clean paths, seeds SEED_START .. SEED_START+n_cal-1
                      (disjoint from training seed=42 and test seeds ROBUSTNESS_SEED_BASE+)
    Test set        : n_test paths per contamination level, seeds ROBUSTNESS_SEED_BASE+i
                      (same paths as robustness_experiment.py for direct comparability)
    """
    print("=" * 72)
    print("CONFORMAL COVERAGE TABLE")
    print(f"  Calibration: {n_cal} clean paths, seeds {SEED_START}–{SEED_START+n_cal-1}")
    print(f"  Test:        {n_test} paths per level, seeds "
          f"{ROBUSTNESS_SEED_BASE}–{ROBUSTNESS_SEED_BASE+n_test-1}")
    print(f"  θ*={TRUE_THETA}, σ*={TRUE_SIGMA}, path_length={PATH_LENGTH}")
    print("=" * 72)

    # ── Load models ───────────────────────────────────────────────────────────
    if not os.path.exists(DEFAULT_ROBUST_PATH):
        print(f"✗ LSTM-robust model not found at {DEFAULT_ROBUST_PATH}")
        return

    lstm_rob_base = OULSTMEstimator(window_size=126, hidden_size=64,
                                    num_layers=2, dropout=0.2)
    lstm_rob_base.load(DEFAULT_ROBUST_PATH)
    print(f"✓ LSTM-robust loaded")

    # ── Build calibration set (clean paths, disjoint seeds) ───────────────────
    print(f"\nBuilding calibration set ({n_cal} clean paths)...")
    cal_series = []
    true_thetas = [TRUE_THETA] * n_cal
    true_sigmas = [TRUE_SIGMA] * n_cal

    for i in range(n_cal):
        path = generate_ou_process(
            mu=TRUE_MU, theta=TRUE_THETA, sigma=TRUE_SIGMA,
            n_steps=PATH_LENGTH, dt=1.0, initial_value=TRUE_MU,
            seed=SEED_START + i
        )
        cal_series.append(path)

    # ── Calibrate conformal estimator ─────────────────────────────────────────
    cp = ConformalOUEstimator(lstm_rob_base, alpha=0.05)
    cp.calibrate(cal_series, true_thetas, true_sigmas, n_mc_samples=mc_samples)
    print(f"  {cp.summary()}")

    # ── Run coverage across contamination levels ──────────────────────────────
    METHOD_NAMES = ['MLE', 'LSTM-rob (MC)', 'LSTM-rob (Conf)']
    records = []

    for jump_rate in JUMP_RATES:
        covered = {m: {'theta': 0, 'sigma': 0} for m in METHOD_NAMES}
        valid   = {m: 0 for m in METHOD_NAMES}

        print(f"\nJump rate {int(jump_rate*100):2d}% — {n_test} test paths...", flush=True)

        for i in range(n_test):
            seed = ROBUSTNESS_SEED_BASE + i
            path = generate_ou_process(
                mu=TRUE_MU, theta=TRUE_THETA, sigma=TRUE_SIGMA,
                n_steps=PATH_LENGTH, dt=1.0, initial_value=TRUE_MU, seed=seed
            )
            contaminated = add_jump_contamination(
                path, jump_rate=jump_rate, jump_scale=JUMP_SCALE, seed=seed + 10000
            )

            # MLE Wald CIs
            try:
                r = estimate_ou_mle(contaminated, verbose=False)
                if r.success and r.theta > 0 and r.sigma > 0:
                    valid['MLE'] += 1
                    if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                        covered['MLE']['theta'] += 1
                    if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                        covered['MLE']['sigma'] += 1
            except Exception:
                pass

            # LSTM-robust MC Dropout CIs
            try:
                r = lstm_rob_base.estimate(contaminated, n_mc_samples=mc_samples)
                valid['LSTM-rob (MC)'] += 1
                if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                    covered['LSTM-rob (MC)']['theta'] += 1
                if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                    covered['LSTM-rob (MC)']['sigma'] += 1
            except Exception:
                pass

            # LSTM-robust Conformal CIs
            try:
                r = cp.estimate(contaminated, n_mc_samples=mc_samples)
                valid['LSTM-rob (Conf)'] += 1
                if r.theta_ci[0] <= TRUE_THETA <= r.theta_ci[1]:
                    covered['LSTM-rob (Conf)']['theta'] += 1
                if r.sigma_ci[0] <= TRUE_SIGMA <= r.sigma_ci[1]:
                    covered['LSTM-rob (Conf)']['sigma'] += 1
            except Exception:
                pass

        pct = lambda h, n: f"{100*h/n:.1f}%" if n > 0 else "N/A"
        print(f"  {'Method':<22}  {'θ cov':>7}  {'σ cov':>7}  {'n':>5}")
        for m in METHOD_NAMES:
            if valid[m] == 0:
                continue
            print(f"  {m:<22}  {pct(covered[m]['theta'], valid[m]):>7}  "
                  f"{pct(covered[m]['sigma'], valid[m]):>7}  {valid[m]:>5}")

        for m in METHOD_NAMES:
            if valid[m] == 0:
                continue
            records.append({
                'jump_rate': jump_rate, 'method': m,
                'theta_coverage': covered[m]['theta'] / valid[m],
                'sigma_coverage': covered[m]['sigma'] / valid[m],
                'n_valid': valid[m],
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    print(f"\n{'=' * 72}")
    print("SUMMARY — θ coverage % across contamination levels (nominal: 95%)")
    print(f"{'ε':>5}  {'MLE':>10}  {'LSTM-rob MC':>12}  {'LSTM-rob Conf':>14}")
    print("-" * 48)
    for jr in JUMP_RATES:
        row = f"{int(jr*100):>4}%"
        for m in METHOD_NAMES:
            sub = df[(df.jump_rate == jr) & (df.method == m)]
            row += f"  {sub.iloc[0]['theta_coverage']*100:>12.1f}%" if len(sub) else f"  {'N/A':>12}"
        print(row)

    print(f"\n{'=' * 72}")
    print("SUMMARY — σ coverage % across contamination levels (nominal: 95%)")
    print(f"{'ε':>5}  {'MLE':>10}  {'LSTM-rob MC':>12}  {'LSTM-rob Conf':>14}")
    print("-" * 48)
    for jr in JUMP_RATES:
        row = f"{int(jr*100):>4}%"
        for m in METHOD_NAMES:
            sub = df[(df.jump_rate == jr) & (df.method == m)]
            row += f"  {sub.iloc[0]['sigma_coverage']*100:>12.1f}%" if len(sub) else f"  {'N/A':>12}"
        print(row)
    print(f"{'=' * 72}")

    out_path = os.path.join(os.path.dirname(__file__), 'calibration_coverage_conformal.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CI calibration coverage check')
    parser.add_argument('--n-paths',    type=int, default=500,
                        help='Number of OU paths (default: 500 for clean, 200 for contaminated)')
    parser.add_argument('--mc-samples', type=int, default=50,
                        help='MC Dropout samples per LSTM call (default: 50)')
    parser.add_argument('--contaminated', action='store_true',
                        help='Run coverage table across all contamination levels (N=200 per level)')
    parser.add_argument('--conformal', action='store_true',
                        help='Run conformal prediction coverage comparison across contamination levels')
    args = parser.parse_args()

    if args.conformal:
        run_conformal_coverage_table(n_cal=500, n_test=200, mc_samples=args.mc_samples)
    elif args.contaminated:
        n = args.n_paths if args.n_paths != 500 else 200
        run_contamination_coverage_table(n_paths=n, mc_samples=args.mc_samples)
    else:
        run_coverage_check(n_paths=args.n_paths, mc_samples=args.mc_samples)
