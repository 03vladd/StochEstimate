"""
Bootstrap Confidence Intervals for MAE Differences

Adds statistical significance testing to the robustness experiment results.
For each (jump_rate, method) pair, computes:
  - 95% percentile bootstrap CI on MAE (theta, sigma)
  - Paired 95% CI on Δ_MAE = MAE_method − MAE_LSTM-robust

Uses the same seeds and path-generation logic as robustness_experiment.py,
so point-estimate MAE values here must match robustness_results_jumps.csv.

Usage:
    python estimation/bootstrap_ci.py [--n-paths 200] [--n-boot 2000]
        [--model PATH] [--model-robust PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr so that unicode symbols (✓ ✗) from imported
# modules don't crash on Windows cp1252 terminals.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.ou_generator import generate_ou_process, add_jump_contamination
from estimation.lstm_estimator import OULSTMEstimator
from estimation.robustness_experiment import (
    run_mle, run_t_mle, run_t_mle_adaptive, run_lstm, run_lstm_robust,
    JUMP_RATES, JUMP_SCALE, PATH_LENGTH, SEED_BASE,
    TRUE_THETA, TRUE_SIGMA, METHOD_ORDER,
    DEFAULT_MODEL_PATH, DEFAULT_MODEL_ROBUST_PATH,
)

# ── Per-path error collection ─────────────────────────────────────────────────

def collect_raw_errors(
    methods: dict,
    theta_true: float,
    sigma_true: float,
    jump_rate: float,
    n_paths: int,
) -> list[dict]:
    """
    Run n_paths synthetic paths and return per-path absolute errors.

    Uses identical seeds and contamination logic as _run_paths() in
    robustness_experiment.py — so MAE values aggregated here must match
    robustness_results_jumps.csv (built-in sanity check).

    Returns
    -------
    list of length n_paths.
    Each element: {method_name: {'theta': abs_err, 'sigma': abs_err}}
    Missing key means the method failed on that path.
    """
    path_results = []

    for i in range(n_paths):
        seed = SEED_BASE + i
        path = generate_ou_process(
            mu=0.0, theta=theta_true, sigma=sigma_true,
            n_steps=PATH_LENGTH, dt=1.0, initial_value=0.0, seed=seed,
        )
        contaminated = add_jump_contamination(
            path, jump_rate=jump_rate, jump_scale=JUMP_SCALE, seed=seed + 10000,
        )

        record = {}
        for name, fn in methods.items():
            t_hat, s_hat = fn(contaminated)
            if not (np.isnan(t_hat) or np.isnan(s_hat)):
                record[name] = {
                    'theta': abs(t_hat - theta_true),
                    'sigma': abs(s_hat - sigma_true),
                }
        path_results.append(record)

    return path_results


# ── Bootstrap functions ────────────────────────────────────────────────────────

def bootstrap_mae_ci(
    path_results: list,
    method: str,
    param: str,
    n_boot: int = 2000,
    rng: np.random.Generator = None,
) -> tuple[float, float, float, int]:
    """
    Percentile bootstrap 95% CI on MAE for one (method, param) pair.

    Parameters
    ----------
    path_results : list of per-path dicts from collect_raw_errors
    method       : method name key (e.g. 'MLE', 'LSTM-robust')
    param        : 'theta' or 'sigma'
    n_boot       : number of bootstrap resamples
    rng          : numpy Generator (created internally if None)

    Returns
    -------
    (mae, ci_lo, ci_hi, n_valid)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    errors = np.array([
        r[method][param]
        for r in path_results
        if method in r
    ])
    n = len(errors)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0

    mae = float(np.mean(errors))

    # Vectorised bootstrap: draw n_boot resamples of size n all at once
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = errors[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return mae, ci_lo, ci_hi, n


def bootstrap_diff_ci(
    path_results: list,
    method_a: str,
    method_b: str,
    param: str,
    n_boot: int = 2000,
    rng: np.random.Generator = None,
) -> tuple[float, float, float, int, bool]:
    """
    Paired percentile bootstrap 95% CI on Δ_MAE = MAE_A − MAE_B.

    Only paths where BOTH methods succeeded are used (preserves pairing).
    d_i = |err_A_i| − |err_B_i|

    Parameters
    ----------
    path_results : list of per-path dicts from collect_raw_errors
    method_a     : name of method A
    method_b     : name of method B (reference, e.g. 'LSTM-robust')
    param        : 'theta' or 'sigma'
    n_boot       : number of bootstrap resamples
    rng          : numpy Generator (created internally if None)

    Returns
    -------
    (delta_mae, ci_lo, ci_hi, n_paired, significant)
    significant = True if 0 is outside [ci_lo, ci_hi]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    paired = [
        (r[method_a][param], r[method_b][param])
        for r in path_results
        if method_a in r and method_b in r
    ]
    n = len(paired)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0, False

    err_a = np.array([p[0] for p in paired])
    err_b = np.array([p[1] for p in paired])
    diffs = err_a - err_b   # d_i = |err_A_i| - |err_B_i|

    delta_mae = float(np.mean(diffs))

    # Vectorised paired bootstrap
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_deltas = diffs[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    significant = not (ci_lo <= 0.0 <= ci_hi)

    return delta_mae, ci_lo, ci_hi, n, significant


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_bootstrap_analysis(
    n_paths: int,
    n_boot: int,
    model_path: str,
    model_robust_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load models, generate paths, run bootstrap CIs for all methods × jump rates.

    Returns
    -------
    (df_mae, df_diff)
    df_mae  : bootstrap_mae_ci.csv contents
    df_diff : bootstrap_diff_ci.csv contents  (empty if LSTM-robust absent)
    """
    # ── Load models ────────────────────────────────────────────────────────────
    print(f"Loading LSTM-v1 model from: {model_path}", flush=True)
    lstm = OULSTMEstimator()
    lstm.load(model_path)

    methods = {
        'MLE':        run_mle,
        't-MLE':      run_t_mle,
        't-MLE(df*)': run_t_mle_adaptive,
        'LSTM':       lambda s: run_lstm(lstm, s),
    }

    has_robust = False
    if model_robust_path and os.path.exists(model_robust_path):
        print(f"Loading LSTM-v2-robust model from: {model_robust_path}", flush=True)
        lstm_robust = OULSTMEstimator()
        lstm_robust.load(model_robust_path)
        methods['LSTM-robust'] = lambda s: run_lstm_robust(lstm_robust, s)
        has_robust = True
    else:
        print("⚠ LSTM-robust model not found — computing MAE CIs only (no Δ column).",
              flush=True)

    # Shared RNG (seeded for reproducibility of bootstrap draws)
    rng = np.random.default_rng(1234)

    mae_rows = []
    diff_rows = []

    for jump_rate in JUMP_RATES:
        pct = int(jump_rate * 100)
        print(f"\nJump rate {pct:2d}% — collecting {n_paths} path errors ...", flush=True)

        path_results = collect_raw_errors(
            methods=methods,
            theta_true=TRUE_THETA,
            sigma_true=TRUE_SIGMA,
            jump_rate=jump_rate,
            n_paths=n_paths,
        )

        for method in METHOD_ORDER:
            if method not in methods:
                continue
            for param in ('theta', 'sigma'):
                mae, ci_lo, ci_hi, n_valid = bootstrap_mae_ci(
                    path_results, method, param, n_boot=n_boot, rng=rng,
                )
                mae_rows.append(dict(
                    jump_rate=jump_rate, method=method, param=param,
                    mae=mae, ci_lo=ci_lo, ci_hi=ci_hi, n_valid=n_valid,
                ))

        if has_robust:
            ref = 'LSTM-robust'
            for method_a in METHOD_ORDER:
                if method_a == ref or method_a not in methods:
                    continue
                for param in ('theta', 'sigma'):
                    delta, ci_lo, ci_hi, n_paired, sig = bootstrap_diff_ci(
                        path_results, method_a, ref, param,
                        n_boot=n_boot, rng=rng,
                    )
                    diff_rows.append(dict(
                        jump_rate=jump_rate,
                        method_a=method_a, method_b=ref, param=param,
                        delta_mae=delta, ci_lo=ci_lo, ci_hi=ci_hi,
                        n_paired=n_paired, significant=sig,
                    ))

    df_mae = pd.DataFrame(mae_rows)
    df_diff = pd.DataFrame(diff_rows) if diff_rows else pd.DataFrame(
        columns=['jump_rate', 'method_a', 'method_b', 'param',
                 'delta_mae', 'ci_lo', 'ci_hi', 'n_paired', 'significant']
    )
    return df_mae, df_diff


# ── Output formatting ──────────────────────────────────────────────────────────

def _sig_star(sig: bool) -> str:
    return '✓' if sig else ' '


def print_bootstrap_table(df_mae: pd.DataFrame, df_diff: pd.DataFrame):
    """
    For each parameter × jump level print one comparison table.

    Example output:
        Jump rate 10% | θ
        Method         MAE     95% CI            vs LSTM-robust Δ   95% CI            Sig?
        ────────────────────────────────────────────────────────────────────────────────────
        MLE          2.0256  [1.623, 2.428]      +1.9923  [1.585, 2.399]              ✓
        ...
        LSTM-robust  0.0233  [0.020, 0.027]      —                                    —
    """
    has_diff = len(df_diff) > 0
    ref_method = 'LSTM-robust'

    PARAMS = [('theta', 'θ'), ('sigma', 'σ')]

    for jump_rate in JUMP_RATES:
        pct = int(jump_rate * 100)
        for param, param_label in PARAMS:
            sub_mae = df_mae[(df_mae.jump_rate == jump_rate) & (df_mae.param == param)]
            if sub_mae.empty:
                continue

            print(f"\nJump rate {pct:2d}% | {param_label}")

            if has_diff:
                sub_diff = df_diff[
                    (df_diff.jump_rate == jump_rate) &
                    (df_diff.param == param) &
                    (df_diff.method_b == ref_method)
                ]
                hdr = (f"{'Method':<14}  {'MAE':>8}  {'95% CI':<20}"
                       f"  {'vs LSTM-robust Δ':>18}  {'95% CI':<20}  {'Sig?':>4}")
                sep = '─' * 84
            else:
                hdr = f"{'Method':<14}  {'MAE':>8}  {'95% CI':<20}  {'n':>5}"
                sep = '─' * 52

            print(hdr)
            print(sep)

            methods_in_table = [m for m in METHOD_ORDER
                                if m in sub_mae['method'].values]

            for method in methods_in_table:
                row_m = sub_mae[sub_mae.method == method].iloc[0]
                mae_str = f"{row_m['mae']:8.4f}"
                ci_str = f"[{row_m['ci_lo']:.4f}, {row_m['ci_hi']:.4f}]"

                if has_diff and method != ref_method:
                    diff_rows = sub_diff[sub_diff.method_a == method]
                    if len(diff_rows) > 0:
                        dr = diff_rows.iloc[0]
                        sign = '+' if dr['delta_mae'] >= 0 else ''
                        d_str = f"{sign}{dr['delta_mae']:8.4f}"
                        d_ci = f"[{dr['ci_lo']:.4f}, {dr['ci_hi']:.4f}]"
                        sig = _sig_star(bool(dr['significant']))
                        print(f"{method:<14}  {mae_str}  {ci_str:<20}  "
                              f"{d_str}  {d_ci:<20}  {sig:>4}")
                    else:
                        print(f"{method:<14}  {mae_str}  {ci_str:<20}  "
                              f"{'n/a':>10}  {'':20}  {'?':>4}")
                elif has_diff and method == ref_method:
                    print(f"{method:<14}  {mae_str}  {ci_str:<20}  "
                          f"{'—':>10}  {'':20}  {'—':>4}")
                else:
                    n_v = int(row_m['n_valid'])
                    print(f"{method:<14}  {mae_str}  {ci_str:<20}  {n_v:>5}")

            print(sep)


def _sanity_check(df_mae: pd.DataFrame, csv_path: str):
    """
    Compare MAE values against robustness_results_jumps.csv.
    Warns if any |diff| > 0.001 (tolerance for floating-point).
    """
    if not os.path.exists(csv_path):
        print(f"⚠ Sanity check skipped: {csv_path} not found.", flush=True)
        return

    ref = pd.read_csv(csv_path)
    print("\nSanity check vs robustness_results_jumps.csv", flush=True)
    all_ok = True
    for _, row in ref.iterrows():
        for param in ('theta', 'sigma'):
            ref_mae = row[f'{param}_mae']
            match = df_mae[
                (df_mae.jump_rate == row['jump_rate']) &
                (df_mae.method == row['method']) &
                (df_mae.param == param)
            ]
            if match.empty:
                continue
            boot_mae = match.iloc[0]['mae']
            diff = abs(boot_mae - ref_mae)
            if diff > 0.001:
                print(f"  ⚠ {row['method']} {param} jump={row['jump_rate']}: "
                      f"ref={ref_mae:.6f}  bootstrap={boot_mae:.6f}  diff={diff:.6f}",
                      flush=True)
                all_ok = False
    if all_ok:
        print("  ✓ All MAE values match robustness_results_jumps.csv within 0.001.",
              flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def save_raw_errors(methods: dict, n_paths: int, out_path: str):
    """
    Collect per-path absolute errors for all jump rates and save to CSV.
    Used by robustness_visualization.py for violin plot Figure 3.
    CSV columns: jump_rate, path_idx, method, theta_err, sigma_err
    Rows where a method failed on a path are omitted.
    """
    rows = []
    for jump_rate in JUMP_RATES:
        pct = int(jump_rate * 100)
        print(f"  Collecting raw errors — jump rate {pct}% ...", flush=True)
        path_results = collect_raw_errors(
            methods=methods,
            theta_true=TRUE_THETA, sigma_true=TRUE_SIGMA,
            jump_rate=jump_rate, n_paths=n_paths,
        )
        for i, record in enumerate(path_results):
            for method, errs in record.items():
                rows.append(dict(
                    jump_rate=jump_rate, path_idx=i, method=method,
                    theta_err=errs['theta'], sigma_err=errs['sigma'],
                ))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap CIs on MAE and MAE differences for OU robustness experiment'
    )
    parser.add_argument('--n-paths', type=int, default=200,
                        help='Paths per contamination level (default: 200)')
    parser.add_argument('--n-boot', type=int, default=2000,
                        help='Bootstrap resamples (default: 2000)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to LSTM-v1 .pt file')
    parser.add_argument('--model-robust', type=str, default=DEFAULT_MODEL_ROBUST_PATH,
                        help='Path to LSTM-v2-robust .pt file')
    parser.add_argument('--save-raw', action='store_true',
                        help='Also save per-path raw errors to bootstrap_raw_errors.csv '
                             '(required for violin plots in robustness_visualization.py)')
    parser.add_argument('--classical-only', action='store_true',
                        help='With --save-raw: skip LSTM inference, collect only MLE / '
                             't-MLE / t-MLE(df*) errors. Runs in ~15 min instead of hours.')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: LSTM model not found at {args.model}")
        print("Run 'python estimation/train_lstm.py' first.")
        sys.exit(1)

    print("=" * 78)
    print(f"BOOTSTRAP CI ANALYSIS — OU Robustness Experiment")
    print(f"Paths: {args.n_paths}  |  Bootstrap resamples: {args.n_boot}  |  "
          f"Jump rates: {JUMP_RATES}")
    print("=" * 78)

    df_mae, df_diff = run_bootstrap_analysis(
        n_paths=args.n_paths,
        n_boot=args.n_boot,
        model_path=args.model,
        model_robust_path=args.model_robust,
    )

    print_bootstrap_table(df_mae, df_diff)

    # Sanity check against previously computed MAE values
    ref_csv = os.path.join(os.path.dirname(__file__), 'robustness_results_jumps.csv')
    _sanity_check(df_mae, ref_csv)

    # Save CSVs
    out_dir = os.path.dirname(__file__)
    mae_path = os.path.join(out_dir, 'bootstrap_mae_ci.csv')
    diff_path = os.path.join(out_dir, 'bootstrap_diff_ci.csv')

    df_mae.to_csv(mae_path, index=False)
    print(f"\nSaved: {mae_path}")

    df_diff.to_csv(diff_path, index=False)
    print(f"Saved: {diff_path}")

    # Optional: save raw per-path errors for violin plots (Fig 3)
    if args.save_raw:
        print("\nCollecting raw per-path errors for violin plots ...", flush=True)
        if args.classical_only:
            print("(--classical-only: skipping LSTM inference)", flush=True)
            methods_raw = {
                'MLE':        run_mle,
                't-MLE':      run_t_mle,
                't-MLE(df*)': run_t_mle_adaptive,
            }
        else:
            from estimation.lstm_estimator import OULSTMEstimator
            lstm_raw = OULSTMEstimator()
            lstm_raw.load(args.model)
            methods_raw = {
                'MLE':        run_mle,
                't-MLE':      run_t_mle,
                't-MLE(df*)': run_t_mle_adaptive,
                'LSTM':       lambda s: run_lstm(lstm_raw, s),
            }
            if args.model_robust and os.path.exists(args.model_robust):
                lstm_robust_raw = OULSTMEstimator()
                lstm_robust_raw.load(args.model_robust)
                methods_raw['LSTM-robust'] = lambda s: run_lstm_robust(lstm_robust_raw, s)
        raw_path = os.path.join(out_dir, 'bootstrap_raw_errors.csv')
        save_raw_errors(methods_raw, n_paths=args.n_paths, out_path=raw_path)


if __name__ == '__main__':
    main()
