"""
Robustness Experiment Visualization

Produces three thesis-quality figures from robustness experiment results:

  1. robustness_degradation.png  — θ and σ MAE vs jump rate, all 3 methods (main figure)
  2. robustness_relative.png     — fold-change from clean baseline (shows collapse scale)
  3. robustness_stratified.png   — σ MAE / σ_true by validation confidence profile

Usage:
  python visualization/robustness_visualization.py
  python visualization/robustness_visualization.py --out-dir path/to/figures
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style constants ────────────────────────────────────────────────────────────

METHOD_STYLES = {
    'MLE':   {'color': '#d62728', 'linestyle': '-',  'marker': 'o', 'label': 'Gaussian MLE'},
    't-MLE': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Student-t MLE (df=4)'},
    'LSTM':  {'color': '#1f77b4', 'linestyle': ':',  'marker': '^', 'label': 'LSTM (MC Dropout)'},
}

METHOD_ORDER = ['MLE', 't-MLE', 'LSTM']
JUMP_PCT     = [0, 2, 5, 10]

CONFIDENCE_PROFILES = {
    'HIGH':   {'theta': 0.034, 'sigma': 1.02},
    'MEDIUM': {'theta': 0.035, 'sigma': 2.00},
    'LOW':    {'theta': 0.054, 'sigma': 0.22},
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['jump_pct'] = (df['jump_rate'] * 100).round().astype(int)
    return df


def _methods_present(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in df['method'].values]


def _log_axis(ax):
    """Apply log y-scale with clean major tick labels, suppress minor labels."""
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


# ── Figure 1: MAE degradation curves ──────────────────────────────────────────

def plot_degradation(df: pd.DataFrame, out_path: str = None) -> plt.Figure:
    """
    Two-panel figure (θ MAE | σ MAE) vs jump contamination rate.
    Log y-scale: MLE collapses to ~8.7 while t-MLE / LSTM stay bounded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        'Robustness to Jump Contamination — Three Estimators\n'
        r'True: $\theta=0.05$,  $\sigma=1.0$  |  jump scale = $5\sigma$  |  200 paths / level',
        fontsize=11,
    )

    panels = [
        ('theta_mae', r'$\hat\theta$ MAE  (log scale)', 'Mean reversion speed'),
        ('sigma_mae', r'$\hat\sigma$ MAE  (log scale)', 'Volatility'),
    ]

    for ax, (col, ylabel, param_name) in zip(axes, panels):
        for method in _methods_present(df):
            sub = df[df['method'] == method].sort_values('jump_pct')
            s = METHOD_STYLES[method]
            ax.plot(
                sub['jump_pct'], sub[col],
                color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markersize=7, linewidth=1.8,
                label=s['label'],
            )

        _log_axis(ax)
        ax.set_xticks(JUMP_PCT)
        ax.set_xlabel('Jump contamination rate (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(param_name)
        ax.legend(loc='upper left')

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"[ok] {out_path}")
    return fig


# ── Figure 2: Fold-change from clean baseline ──────────────────────────────────

def plot_relative_degradation(df: pd.DataFrame, out_path: str = None) -> plt.Figure:
    """
    Each method's MAE divided by its own clean-data (0% contamination) baseline.
    Puts all three methods on a common 'degradation' scale regardless of
    their absolute accuracy at 0% — a fairer measure of robustness.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        'Relative Degradation from Clean Baseline\n'
        r'(MAE at contamination level) / (MAE at 0%)  —  log scale',
        fontsize=11,
    )

    panels = [
        ('theta_mae', r'$\hat\theta$ MAE fold-change', 'Mean reversion speed'),
        ('sigma_mae', r'$\hat\sigma$ MAE fold-change', 'Volatility'),
    ]

    for ax, (col, ylabel, param_name) in zip(axes, panels):
        for method in _methods_present(df):
            sub = df[df['method'] == method].sort_values('jump_pct')
            baseline_row = sub[sub['jump_pct'] == 0]
            if baseline_row.empty:
                continue
            baseline = baseline_row[col].values[0]
            fold = sub[col].values / baseline

            s = METHOD_STYLES[method]
            ax.plot(
                sub['jump_pct'].values, fold,
                color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markersize=7, linewidth=1.8,
                label=s['label'],
            )

        # 1× reference line (= no degradation)
        ax.axhline(1.0, color='black', linewidth=0.8, linestyle='-', alpha=0.4, zorder=0)

        _log_axis(ax)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:.0f}×' if x >= 10 else f'{x:.1f}×')
        )
        ax.set_xticks(JUMP_PCT)
        ax.set_xlabel('Jump contamination rate (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(param_name)
        ax.legend(loc='upper left')

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"[ok] {out_path}")
    return fig


# ── Figure 3: Stratified by validation confidence profile ─────────────────────

def plot_stratified(df_strat: pd.DataFrame, out_path: str = None) -> plt.Figure:
    """
    σ MAE / σ_true vs jump rate, one panel per validation confidence profile.
    Normalising by σ_true makes the HIGH / MEDIUM / LOW profiles comparable
    despite having very different absolute σ values (1.02 / 2.00 / 0.22).
    """
    profiles = [p for p in ['HIGH', 'MEDIUM', 'LOW'] if p in df_strat['confidence'].values]
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        r'Stratified Robustness by Validation Confidence Profile  ($\hat\sigma$ MAE / $\sigma_\mathrm{true}$, log scale)',
        fontsize=11,
    )

    for ax, level in zip(axes, profiles):
        sub_level = df_strat[df_strat['confidence'] == level]
        sigma_true = CONFIDENCE_PROFILES[level]['sigma']
        theta_true = CONFIDENCE_PROFILES[level]['theta']

        for method in _methods_present(df_strat):
            sub = sub_level[sub_level['method'] == method].sort_values('jump_pct')
            if sub.empty:
                continue
            s = METHOD_STYLES[method]
            ax.plot(
                sub['jump_pct'],
                sub['sigma_mae'] / sigma_true,
                color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markersize=7, linewidth=1.8,
                label=s['label'],
            )

        _log_axis(ax)
        ax.set_xticks(JUMP_PCT)
        ax.set_xlabel('Jump contamination rate (%)')
        ax.set_ylabel(r'$\hat\sigma$ MAE / $\sigma_\mathrm{true}$  (log scale)')
        ax.set_title(
            f'{level} confidence\n'
            r'$\theta$=' + f'{theta_true},  ' + r'$\sigma$=' + f'{sigma_true}'
        )
        ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"[ok] {out_path}")
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Visualize robustness experiment results')
    parser.add_argument(
        '--jumps-csv', type=str,
        default=os.path.join(root, 'estimation', 'robustness_results_jumps.csv'),
        help='Path to robustness_results_jumps.csv',
    )
    parser.add_argument(
        '--stratified-csv', type=str,
        default=os.path.join(root, 'estimation', 'robustness_results_stratified.csv'),
        help='Path to robustness_results_stratified.csv',
    )
    parser.add_argument(
        '--out-dir', type=str,
        default=os.path.join(root, 'estimation'),
        help='Directory to save PNG figures (default: estimation/)',
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    produced = []

    if os.path.exists(args.jumps_csv):
        df = load_results(args.jumps_csv)
        methods = _methods_present(df)
        print(f"Loaded jump results: {len(df)} rows, methods: {methods}")

        p1 = os.path.join(args.out_dir, 'robustness_degradation.png')
        p2 = os.path.join(args.out_dir, 'robustness_relative.png')
        plot_degradation(df, out_path=p1)
        plot_relative_degradation(df, out_path=p2)
        produced += [p1, p2]
    else:
        print(f"warn: Jumps CSV not found: {args.jumps_csv}")

    if os.path.exists(args.stratified_csv):
        df_strat = load_results(args.stratified_csv)
        methods = _methods_present(df_strat)
        print(f"Loaded stratified results: {len(df_strat)} rows, methods: {methods}")
        if 't-MLE' not in methods:
            print("note: t-MLE not in stratified CSV (experiment pre-dates t-MLE addition).")

        p3 = os.path.join(args.out_dir, 'robustness_stratified.png')
        plot_stratified(df_strat, out_path=p3)
        produced.append(p3)
    else:
        print(f"warn: Stratified CSV not found: {args.stratified_csv}")

    print(f"\n{len(produced)} figure(s) saved.")
    plt.show()


if __name__ == '__main__':
    main()
