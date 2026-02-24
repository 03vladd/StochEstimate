"""
Robustness Experiment — Scenario A: Jump Outlier Contamination

Three-way comparison: Gaussian MLE vs Student-t MLE (df=4) vs LSTM.

Contaminates synthetic OU paths with jumps drawn from N(0, jump_scale×σ)
at rates [0%, 2%, 5%, 10%] and measures θ/σ MAE and RMSE for each estimator.

Hypothesis:
  - Gaussian MLE breaks fast: quadratic penalty makes each jump dominate loss.
  - Student-t MLE (df=4) should be substantially more robust: bounded penalty
    on large residuals (Muler, Peña & Yohai 2009; Barndorff-Nielsen & Shephard 2001).
  - LSTM may be competitive with t-MLE despite no explicit robustness design,
    providing amortized inference speed as an additional benefit.

Stratified experiment:
  Repeats the analysis using OU parameter profiles matching the three
  validation confidence levels observed in real pairs:

    HIGH   profile — θ≈0.034, σ≈1.0  (Morgan Stanley / Goldman Sachs)
    MEDIUM profile — θ≈0.035, σ≈2.0  (average of EOG/SLB, AMZN/WMT, EXC/AEP)
    LOW    profile — θ≈0.054, σ≈0.3  (PPL / Ameren)

Usage:
  python estimation/robustness_experiment.py [--n-paths N] [--model PATH]
  python estimation/robustness_experiment.py --stratified [--n-paths N] [--model PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.ou_generator import generate_ou_process, add_jump_contamination
from estimation.mle import estimate_ou_mle
from estimation.mle_robust import estimate_ou_t_mle
from estimation.lstm_estimator import OULSTMEstimator

# ── Experiment parameters ─────────────────────────────────────────────────────

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

# Method display order (consistent across all tables)
METHOD_ORDER = ['MLE', 't-MLE', 'LSTM']

# ── Core estimation helpers ───────────────────────────────────────────────────

def run_mle(series: pd.Series) -> tuple[float, float]:
    """Return (theta_hat, sigma_hat) from Gaussian MLE, or (nan, nan) on failure."""
    try:
        result = estimate_ou_mle(series, verbose=False)
        if result.theta > 0 and result.sigma > 0:
            return result.theta, result.sigma
        return float('nan'), float('nan')
    except Exception:
        return float('nan'), float('nan')


def run_t_mle(series: pd.Series, df: float = 4.0) -> tuple[float, float]:
    """Return (theta_hat, sigma_hat) from Student-t MLE (df=4), or (nan, nan) on failure."""
    try:
        result = estimate_ou_t_mle(series, df=df, verbose=False)
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


def _stats(errs: list) -> tuple[float, float]:
    """Return (MAE, RMSE) from a list of absolute errors."""
    if not errs:
        return float('nan'), float('nan')
    a = np.array(errs)
    return float(np.mean(a)), float(np.sqrt(np.mean(a ** 2)))


# ── Generic inner loop (shared by both experiments) ──────────────────────────

def _run_paths(methods: dict, theta_true: float, sigma_true: float,
               jump_rate: float, n_paths: int) -> list[dict]:
    """
    For one (jump_rate, set of methods), run n_paths paths and return records.
    methods: {name: callable(series) -> (theta_hat, sigma_hat)}
    """
    errs = {m: {'theta': [], 'sigma': []} for m in methods}

    for i in range(n_paths):
        seed = SEED_BASE + i
        path = generate_ou_process(
            mu=0.0, theta=theta_true, sigma=sigma_true,
            n_steps=PATH_LENGTH, dt=1.0, initial_value=0.0, seed=seed
        )
        contaminated = add_jump_contamination(
            path, jump_rate=jump_rate, jump_scale=JUMP_SCALE, seed=seed + 10000
        )
        for name, fn in methods.items():
            t_hat, s_hat = fn(contaminated)
            if not (np.isnan(t_hat) or np.isnan(s_hat)):
                errs[name]['theta'].append(abs(t_hat - theta_true))
                errs[name]['sigma'].append(abs(s_hat - sigma_true))

    records = []
    for name in METHOD_ORDER:
        if name not in methods:
            continue
        t_mae, t_rmse = _stats(errs[name]['theta'])
        s_mae, s_rmse = _stats(errs[name]['sigma'])
        n_v = len(errs[name]['theta'])
        print(f"  {name:>5} — θ MAE={t_mae:.4f}  σ MAE={s_mae:.4f}  (n={n_v})", flush=True)
        records.append(dict(
            jump_rate=jump_rate, method=name,
            theta_mae=t_mae, theta_rmse=t_rmse,
            sigma_mae=s_mae, sigma_rmse=s_rmse,
            n_valid=n_v,
        ))
    return records


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(n_paths: int, model_path: str) -> pd.DataFrame:
    """
    Three-way comparison (MLE / t-MLE / LSTM) at four jump contamination levels.
    Returns DataFrame with columns: jump_rate | method | theta_mae | theta_rmse | sigma_mae | sigma_rmse
    """
    print(f"Loading LSTM model from: {model_path}", flush=True)
    lstm = OULSTMEstimator()
    lstm.load(model_path)

    methods = {
        'MLE':   run_mle,
        't-MLE': run_t_mle,
        'LSTM':  lambda s: run_lstm(lstm, s),
    }

    records = []
    for jump_rate in JUMP_RATES:
        print(f"\nJump rate {int(jump_rate*100):2d}% — estimating {n_paths} paths ...", flush=True)
        records.extend(_run_paths(methods, TRUE_THETA, TRUE_SIGMA, jump_rate, n_paths))

    return pd.DataFrame(records)


def print_summary_table(df: pd.DataFrame):
    """Print a formatted three-way comparison table for the thesis."""
    print("\n")
    print("=" * 84)
    print("ROBUSTNESS EXPERIMENT — Scenario A: Jump Contamination (3-way)")
    print(f"True: θ={TRUE_THETA}, μ={TRUE_MU}, σ={TRUE_SIGMA}  |  "
          f"Jump scale={JUMP_SCALE}×σ  |  Path length={PATH_LENGTH}")
    print("=" * 84)
    print(f"{'Jump %':>8}  {'Method':>6}  {'θ MAE':>9}  {'θ RMSE':>9}  "
          f"{'σ MAE':>9}  {'σ RMSE':>9}  {'n':>5}")
    print("-" * 84)

    methods_in_df = [m for m in METHOD_ORDER if m in df['method'].values]
    last_method = methods_in_df[-1]

    for _, row in df.iterrows():
        pct = f"{row['jump_rate'] * 100:.0f}%"
        print(
            f"{pct:>8}  {row['method']:>6}  "
            f"{row['theta_mae']:>9.4f}  {row['theta_rmse']:>9.4f}  "
            f"{row['sigma_mae']:>9.4f}  {row['sigma_rmse']:>9.4f}  "
            f"{int(row['n_valid']):>5}"
        )
        if row['method'] == last_method:
            print()

    print("=" * 84)

    # Narrative: compare all methods against Gaussian MLE at highest jump rate
    top_jr = JUMP_RATES[-1]
    clean_mle = df[(df.jump_rate == 0.00) & (df.method == 'MLE')].iloc[0]
    high_mle  = df[(df.jump_rate == top_jr) & (df.method == 'MLE')].iloc[0]
    mle_s_deg = (high_mle['sigma_mae'] - clean_mle['sigma_mae']) / (clean_mle['sigma_mae'] + 1e-9)

    print("\nNARRATIVE CONCLUSION")
    print("-" * 84)
    print(f"  Baseline (0% contamination) — MLE σ MAE={clean_mle['sigma_mae']:.4f}")
    print(f"  At {top_jr*100:.0f}% contamination:")
    for method in methods_in_df:
        row_high  = df[(df.jump_rate == top_jr) & (df.method == method)].iloc[0]
        row_clean = df[(df.jump_rate == 0.00)  & (df.method == method)].iloc[0]
        deg = (row_high['sigma_mae'] - row_clean['sigma_mae']) / (row_clean['sigma_mae'] + 1e-9)
        rel = row_high['sigma_mae'] / high_mle['sigma_mae']
        print(f"    {method:>5}  σ MAE={row_high['sigma_mae']:.4f}  "
              f"(+{deg*100:.0f}% from clean,  {rel:.2f}× MLE)")
    print("=" * 84)


# ── Stratified experiment (by validation confidence profile) ──────────────────

# Parameter profiles derived from actual DB pairs (MLE estimates on 2023-2025 data)
CONFIDENCE_PROFILES = {
    'HIGH':   {'theta': 0.034, 'sigma': 1.02, 'example': 'Morgan Stanley / Goldman Sachs'},
    'MEDIUM': {'theta': 0.035, 'sigma': 2.00, 'example': 'EOG/SLB, AMZN/WMT, EXC/AEP (avg)'},
    'LOW':    {'theta': 0.054, 'sigma': 0.22, 'example': 'PPL / Ameren'},
}


def run_stratified_experiment(n_paths: int, model_path: str) -> dict[str, pd.DataFrame]:
    """
    Run the three-way jump contamination experiment once per confidence-level profile.
    Returns dict: confidence_level -> results DataFrame.
    """
    print(f"Loading LSTM model from: {model_path}", flush=True)
    lstm = OULSTMEstimator()
    lstm.load(model_path)

    methods = {
        'MLE':   run_mle,
        't-MLE': run_t_mle,
        'LSTM':  lambda s: run_lstm(lstm, s),
    }

    all_results = {}

    for level, profile in CONFIDENCE_PROFILES.items():
        theta_true = profile['theta']
        sigma_true = profile['sigma']

        print(f"\n{'='*78}", flush=True)
        print(f"CONFIDENCE PROFILE: {level}  "
              f"(θ={theta_true}, σ={sigma_true})  —  {profile['example']}", flush=True)
        print(f"{'='*78}", flush=True)

        records = []
        for jump_rate in JUMP_RATES:
            print(f"\n  Jump rate {int(jump_rate*100):2d}% — estimating {n_paths} paths ...", flush=True)
            records.extend(_run_paths(methods, theta_true, sigma_true, jump_rate, n_paths))

        all_results[level] = pd.DataFrame(records)

    return all_results


def print_stratified_summary(all_results: dict[str, pd.DataFrame]):
    """Print a compact cross-profile comparison table for the thesis."""
    methods_present = [m for m in METHOD_ORDER
                       if any(m in df['method'].values for df in all_results.values())]

    col_header = "  ".join(
        f"{'  '.join(methods_present):^{len(methods_present)*6}}" if jr == 0.00
        else f"{'  '.join(m[:5] for m in methods_present)}"
        for jr in JUMP_RATES
    )

    W = 100
    print("\n\n" + "=" * W)
    print("STRATIFIED ROBUSTNESS — Jump Contamination by Validation Confidence Profile (3-way)")
    print(f"Jump scale={JUMP_SCALE}×σ  |  Path length={PATH_LENGTH}  |  σ MAE / σ_true shown")
    print("=" * W)

    # Header row
    header = f"{'Profile':>8}  {'θ':>6}  {'σ':>5}"
    for jr in JUMP_RATES:
        header += f"  {int(jr*100):>2}%:{'  '.join(m[:4] for m in methods_present):>{len(methods_present)*5}}"
    print(header)
    print("-" * W)

    win_counts = {m: 0 for m in methods_present if m != 'MLE'}
    total_comparisons = 0

    for level, df in all_results.items():
        profile = CONFIDENCE_PROFILES[level]
        sigma_true = profile['sigma']
        row = f"{level:>8}  {profile['theta']:>6.3f}  {profile['sigma']:>5.2f}"
        for jr in JUMP_RATES:
            vals = []
            for m in methods_present:
                r = df[(df.jump_rate == jr) & (df.method == m)]
                v = r.iloc[0]['sigma_mae'] / sigma_true if len(r) > 0 else float('nan')
                vals.append((m, v))
            row += "  " + " ".join(f"{v:.3f}" for _, v in vals)

            if jr > 0.0:
                mle_val = next(v for m, v in vals if m == 'MLE')
                for m, v in vals:
                    if m != 'MLE':
                        total_comparisons += 1
                        if v < mle_val:
                            win_counts[m] += 1
        print(row)

    print("=" * W)
    print("Values are σ MAE / σ_true (lower = better).")
    print()
    for m, wins in win_counts.items():
        print(f"  {m} beats Gaussian MLE: {wins}/{total_comparisons} contaminated comparisons.")
    print()
    if all(w == total_comparisons for w in win_counts.values()):
        print("Conclusion: Both t-MLE and LSTM are robustly better than Gaussian MLE across")
        print("           all parameter profiles and contamination levels.")
    print("=" * W)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OU estimator robustness: Gaussian MLE vs Student-t MLE vs LSTM"
    )
    parser.add_argument('--n-paths', type=int, default=200,
                        help='Number of synthetic paths per contamination level (default: 200)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to trained LSTM .pt file')
    parser.add_argument('--stratified', action='store_true',
                        help='Run stratified experiment by validation confidence profile')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: LSTM model not found at {args.model}")
        print("Run 'python estimation/train_lstm.py' first.")
        sys.exit(1)

    if args.stratified:
        print("=" * 78)
        print("ROBUSTNESS EXPERIMENT — Stratified by Validation Confidence Profile (3-way)")
        print(f"Paths per level: {args.n_paths}  |  Jump rates: {JUMP_RATES}")
        print("=" * 78)

        all_results = run_stratified_experiment(n_paths=args.n_paths, model_path=args.model)
        print_stratified_summary(all_results)

        out_path = os.path.join(os.path.dirname(__file__), 'robustness_results_stratified.csv')
        pd.concat(
            [df.assign(confidence=level) for level, df in all_results.items()],
            ignore_index=True
        ).to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")

    else:
        print("=" * 78)
        print("ROBUSTNESS EXPERIMENT — Scenario A: Jump Contamination (3-way)")
        print(f"Paths per level: {args.n_paths}  |  Jump rates: {JUMP_RATES}")
        print("=" * 78)

        df = run_experiment(n_paths=args.n_paths, model_path=args.model)
        print_summary_table(df)

        out_path = os.path.join(os.path.dirname(__file__), 'robustness_results_jumps.csv')
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
