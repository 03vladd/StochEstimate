"""
Robustness Experiment Visualization — Phase 7

Four thesis-quality figures for the 5-way comparison:

  Fig 1. robustness_degradation_ci.png
       θ and σ MAE vs jump rate, all 5 methods, with 95% bootstrap CI shading.
       Protocol: Muler, Peña & Yohai (2009) degradation curve structure;
                 Efron & Tibshirani (1993) Ch. 13 percentile bootstrap CI bands.

  Fig 2. robustness_frontier.png
       Efficiency-robustness frontier: 0% MAE (x) vs 10% MAE (y), one labelled
       point per method, error bars from bootstrap CIs.
       Protocol: Huber (1964) efficiency-robustness trade-off;
                 Hampel, Ronchetti, Rousseeuw & Stahel (1986) Ch. 1 Pareto framing;
                 Maronna, Martin & Yohai (2006) Ch. 2 scatter convention.

  Fig 3. robustness_violin_theta.png / robustness_violin_sigma.png
       Per-path absolute error distributions (violin plots), log y-scale.
       Requires bootstrap_raw_errors.csv — run bootstrap_ci.py --save-raw first.
       Protocol: Hintze & Nelson (1998) violin plot structure (KDE + median);
                 Box & Cox (1964) log transform for right-skewed data;
                 Rousseeuw & Leroy (1987) Ch. 1 on showing full error distributions.

  Fig 4. training_loss_curves.png
       Train and validation MSE loss per epoch for LSTM-v1 and LSTM-robust.
       Requires lstm_v1_losses.csv and lstm_v2_losses.csv.
       Re-run train_lstm.py / train_lstm_robust.py to generate them.
       Protocol: Goodfellow, Bengio & Courville (2016) Ch. 8.2 on monitoring
                 training dynamics to diagnose convergence failure.

Usage:
  python visualization/robustness_visualization.py
  python visualization/robustness_visualization.py --out-dir figures/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Style constants ────────────────────────────────────────────────────────────

# Colour convention: red → catastrophic failure, orange → moderate robustness,
# brown → best classical, blue → neural clean, green → neural robust (winner).
METHOD_STYLES = {
    'MLE':         {'color': '#d62728', 'linestyle': '-',  'marker': 'o', 'lw': 1.8,
                    'label': 'Gaussian MLE',        'alpha_fill': 0.13},
    't-MLE':       {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'lw': 1.8,
                    'label': 'Student-t MLE (df=4)', 'alpha_fill': 0.13},
    't-MLE(df*)':  {'color': '#8c564b', 'linestyle': '-.', 'marker': 'D', 'lw': 1.8,
                    'label': 't-MLE (adaptive df)', 'alpha_fill': 0.13},
    'LSTM':        {'color': '#1f77b4', 'linestyle': ':',  'marker': '^', 'lw': 1.8,
                    'label': 'LSTM (MC Dropout)',   'alpha_fill': 0.13},
    'LSTM-robust': {'color': '#2ca02c', 'linestyle': '-',  'marker': '*', 'lw': 2.4,
                    'label': 'LSTM-robust (v2)',    'alpha_fill': 0.20},
}

METHOD_ORDER = ['MLE', 't-MLE', 't-MLE(df*)', 'LSTM', 'LSTM-robust']
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
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def _methods_present(df: pd.DataFrame, col: str = 'method') -> list[str]:
    return [m for m in METHOD_ORDER if m in df[col].values]


def _log_axis(ax):
    """Log y-scale with clean major tick labels, no minor tick clutter."""
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _save(fig: plt.Figure, out_path):
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"[ok] {out_path}")


def load_jumps(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['jump_pct'] = (df['jump_rate'] * 100).round().astype(int)
    return df


def load_bootstrap_ci(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['jump_pct'] = (df['jump_rate'] * 100).round().astype(int)
    return df


# ── Figure 1: Degradation curves with CI bands ────────────────────────────────

def plot_degradation_ci(df_jumps: pd.DataFrame, df_ci: pd.DataFrame,
                        out_path: str = None) -> plt.Figure:
    """
    Two-panel figure (θ MAE | σ MAE) vs jump contamination rate, 5 methods.
    Each method: solid line (point estimate) + shaded 95% percentile bootstrap CI.
    Log y-scale allows MLE's three-order-of-magnitude collapse to appear alongside
    LSTM-robust's near-flat sub-0.1 trajectory on the same axes.

    Protocol: Muler, Peña & Yohai (2009) use this degradation curve structure
    to compare M-estimators for AR models (their Fig. 1 is the direct precedent).
    Shaded CI bands: Efron & Tibshirani (1993) Ch. 13.
    """
    methods = _methods_present(df_jumps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        '5-way MAE Degradation under Jump Contamination\n'
        r'True: $\theta=0.05$, $\sigma=1.0$  |  jump scale $=5\sigma$  |  '
        r'$n=200$ paths/level  |  shaded = 95% bootstrap CI',
        fontsize=10.5,
    )

    panels = [
        ('theta_mae', 'theta', r'$\hat\theta$ MAE  (log scale)', r'Mean reversion speed $\theta$'),
        ('sigma_mae', 'sigma', r'$\hat\sigma$ MAE  (log scale)', r'Volatility $\sigma$'),
    ]

    for ax, (col, param, ylabel, title) in zip(axes, panels):
        for method in methods:
            sub_j  = df_jumps[df_jumps['method'] == method].sort_values('jump_pct')
            sub_ci = df_ci[
                (df_ci['method'] == method) & (df_ci['param'] == param)
            ].sort_values('jump_pct')
            s = METHOD_STYLES[method]
            ms = 9 if method == 'LSTM-robust' else 7

            ax.plot(
                sub_j['jump_pct'], sub_j[col],
                color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markersize=ms,
                linewidth=s['lw'], label=s['label'], zorder=3,
            )
            if len(sub_ci) == len(sub_j):
                ax.fill_between(
                    sub_ci['jump_pct'],
                    sub_ci['ci_lo'], sub_ci['ci_hi'],
                    color=s['color'], alpha=s['alpha_fill'], zorder=2,
                )

        _log_axis(ax)
        ax.set_xticks(JUMP_PCT)
        ax.set_xlabel('Jump contamination rate (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper left')

    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ── Figure 2: Efficiency-robustness frontier ──────────────────────────────────

def plot_efficiency_robustness_frontier(df_ci: pd.DataFrame,
                                        out_path: str = None) -> plt.Figure:
    """
    Scatter: clean-data MAE (x, efficiency) vs 10% contamination MAE (y, robustness).
    One labelled point per method; error bars = 95% bootstrap CI half-widths.
    Lower-left = Pareto-dominant (simultaneously more efficient AND more robust).
    Log-log axes: required because the methods span three orders of magnitude.

    Protocol: Huber (1964) introduced the efficiency-robustness trade-off;
    Hampel et al. (1986) Ch. 1 formalised the Pareto frontier framing;
    Maronna et al. (2006) Ch. 2 use 2D scatter plots throughout to position estimators.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        'Efficiency–Robustness Frontier\n'
        'x = MAE on clean data (↓ more efficient)  |  '
        'y = MAE at 10% contamination (↓ more robust)\n'
        'Lower-left = Pareto-dominant. Error bars = 95% bootstrap CI.',
        fontsize=10.5,
    )

    panels = [
        ('theta', r'$\hat\theta$ MAE', r'Mean reversion speed $\theta$'),
        ('sigma', r'$\hat\sigma$ MAE', r'Volatility $\sigma$'),
    ]

    for ax, (param, axlabel, title) in zip(axes, panels):
        sub   = df_ci[df_ci['param'] == param]
        clean = sub[sub['jump_pct'] == 0].set_index('method')
        dirty = sub[sub['jump_pct'] == 10].set_index('method')
        methods = _methods_present(sub, col='method')

        all_vals = []
        for method in methods:
            if method not in clean.index or method not in dirty.index:
                continue
            s  = METHOD_STYLES[method]
            cx = float(clean.loc[method, 'mae'])
            cy = float(dirty.loc[method, 'mae'])
            xerr = np.array([[cx - clean.loc[method, 'ci_lo']],
                             [clean.loc[method, 'ci_hi'] - cx]])
            yerr = np.array([[cy - dirty.loc[method, 'ci_lo']],
                             [dirty.loc[method, 'ci_hi'] - cy]])

            ax.errorbar(
                cx, cy, xerr=xerr, yerr=yerr,
                fmt=s['marker'], color=s['color'],
                markersize=11 if method == 'LSTM-robust' else 8,
                markeredgecolor='white', markeredgewidth=0.8,
                elinewidth=1.2, capsize=4, capthick=1.2,
                label=s['label'], zorder=4,
            )
            short_label = (s['label']
                           .replace(' (MC Dropout)', '')
                           .replace(' (v2)', '')
                           .replace(' (adaptive df)', ''))
            ax.annotate(
                short_label, (cx, cy),
                textcoords='offset points', xytext=(7, 4),
                fontsize=8, color=s['color'],
            )
            all_vals += [cx, cy]

        # y=x reference: a method on this line degrades perfectly—no worse at 10%
        # than it was on clean data.  In practice every method lies above it.
        lo = min(all_vals) * 0.4
        hi = max(all_vals) * 2.5
        ax.plot([lo, hi], [lo, hi], color='grey', linewidth=0.8,
                linestyle='--', alpha=0.5, zorder=1, label='y = x (no degradation)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Clean-data {axlabel}  (↓ more efficient)')
        ax.set_ylabel(f'10% contamination {axlabel}  (↓ more robust)')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ── Figure 3: Per-path error distributions (violin plots) ─────────────────────

def plot_violin_errors(df_raw: pd.DataFrame, param: str,
                       out_path: str = None) -> plt.Figure:
    """
    One figure per parameter, four panels (one per jump rate).
    Each panel: violin plots of log10 per-path absolute errors for each method.

    Log10 transform applied before KDE estimation (Box & Cox 1964) because
    raw errors span three orders of magnitude (LSTM-robust ~0.02 to MLE ~9
    at 10% contamination) — a linear-scale violin for MLE would make all other
    violins invisible.  Y-axis ticks are re-labelled with actual error values
    (10^y) for readability.

    Protocol: Hintze & Nelson (1998) — violin = KDE body + median line;
              Box & Cox (1964) log transform for right-skewed data;
              Rousseeuw & Leroy (1987) Ch. 1 — show full distributions, not just mean.
    """
    err_col     = f'{param}_err'
    param_label = r'$\theta$' if param == 'theta' else r'$\sigma$'
    methods     = [m for m in METHOD_ORDER if m in df_raw['method'].values]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f'Per-path Absolute Error Distributions — {param_label}\n'
        r'$\log_{10}$ scale. Violin shape = KDE; white line = median.  '
        r'$n \leq 200$ paths/level.',
        fontsize=10.5,
    )

    for ax, jump_pct in zip(axes, JUMP_PCT):
        sub = df_raw[df_raw['jump_pct'] == jump_pct]
        data_per_method = []
        valid_methods   = []

        for method in methods:
            errs = sub[sub['method'] == method][err_col].dropna().values
            if len(errs) > 1:
                # log10 transform; 1e-8 floor prevents log(0) on perfect estimates
                data_per_method.append(np.log10(errs + 1e-8))
                valid_methods.append(method)

        if not data_per_method:
            ax.set_visible(False)
            continue

        positions = np.arange(1, len(valid_methods) + 1)
        parts = ax.violinplot(
            data_per_method, positions=positions,
            showmedians=True, showextrema=False,
        )

        for body, method in zip(parts['bodies'], valid_methods):
            body.set_facecolor(METHOD_STYLES[method]['color'])
            body.set_edgecolor('none')
            body.set_alpha(0.65)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2.0)

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [m.replace('LSTM-robust', 'LSTM\nrobust')
              .replace('t-MLE(df*)', 't-MLE\n(df*)')
             for m in valid_methods],
            fontsize=8,
        )
        ax.set_title(f'Jump rate {jump_pct}%')
        if jump_pct == 0:
            ax.set_ylabel(
                r'$\log_{10}|\hat{p} - p_\mathrm{true}|$  '
                f'({param_label})'
            )

        # Relabel y-ticks with actual error values (inverse of log10 transform)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'{10 ** y:.3g}')
        )

    # Shared legend at figure bottom
    handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=METHOD_STYLES[m]['color'],
                   markersize=9, label=METHOD_STYLES[m]['label'])
        for m in methods
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(methods),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, out_path)
    return fig


# ── Figure 4: Training loss curves ────────────────────────────────────────────

def plot_training_curves(df_v1: pd.DataFrame, df_v2: pd.DataFrame,
                         out_path: str = None) -> plt.Figure:
    """
    Two-panel figure: training loss (left) and validation loss (right) per epoch,
    overlaid for LSTM-v1 (clean training) and LSTM-robust (50% contaminated).

    If the v2 validation loss converges to a comparable level as v1, this
    establishes that contamination-aware training did not destabilise optimisation
    or prevent the model from learning a useful representation of clean OU dynamics.

    Protocol: Goodfellow, Bengio & Courville (2016) Ch. 8.2 — plot train and
    validation loss curves to diagnose overfitting, underfitting, and convergence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        'Training Convergence — LSTM-v1 (clean) vs LSTM-robust (50% contaminated)\n'
        'MSE loss on log-space targets  (log θ, μ_norm, log σ_norm)',
        fontsize=10.5,
    )

    v1_color = METHOD_STYLES['LSTM']['color']
    v2_color = METHOD_STYLES['LSTM-robust']['color']

    for ax, (col, panel_title) in zip(axes, [('train_loss', 'Training loss'),
                                              ('val_loss',   'Validation loss')]):
        ax.plot(df_v1['epoch'], df_v1[col], color=v1_color, linewidth=1.8,
                label='LSTM-v1 (clean training)')
        ax.plot(df_v2['epoch'], df_v2[col], color=v2_color, linewidth=1.8,
                linestyle='--', label='LSTM-robust (50% contaminated)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss')
        ax.set_title(panel_title)
        ax.legend()
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))

    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ── Legacy: stratified by validation confidence profile ───────────────────────

def plot_stratified(df_strat: pd.DataFrame, out_path: str = None) -> plt.Figure:
    """
    σ MAE / σ_true vs jump rate, one panel per validation confidence profile.
    Normalising by σ_true makes HIGH / MEDIUM / LOW profiles comparable
    despite very different absolute σ values (1.02 / 2.00 / 0.22).
    """
    profiles = [p for p in ['HIGH', 'MEDIUM', 'LOW'] if p in df_strat['confidence'].values]
    methods  = _methods_present(df_strat)
    n        = len(profiles)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        r'Stratified Robustness by Validation Confidence Profile'
        r'  ($\hat\sigma$ MAE / $\sigma_\mathrm{true}$, log scale)',
        fontsize=11,
    )

    for ax, level in zip(axes, profiles):
        sub_level  = df_strat[df_strat['confidence'] == level]
        sigma_true = CONFIDENCE_PROFILES[level]['sigma']
        theta_true = CONFIDENCE_PROFILES[level]['theta']

        for method in methods:
            sub = sub_level[sub_level['method'] == method].sort_values('jump_pct')
            if sub.empty:
                continue
            s = METHOD_STYLES[method]
            ax.plot(
                sub['jump_pct'], sub['sigma_mae'] / sigma_true,
                color=s['color'], linestyle=s['linestyle'],
                marker=s['marker'], markersize=7, linewidth=s['lw'],
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
        ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    est  = os.path.join(root, 'estimation')

    parser = argparse.ArgumentParser(description='Produce Phase 7 thesis visualizations')
    parser.add_argument('--jumps-csv',
                        default=os.path.join(est, 'robustness_results_jumps.csv'))
    parser.add_argument('--ci-csv',
                        default=os.path.join(est, 'bootstrap_mae_ci.csv'))
    parser.add_argument('--stratified-csv',
                        default=os.path.join(est, 'robustness_results_stratified.csv'))
    parser.add_argument('--raw-csv',
                        default=os.path.join(est, 'bootstrap_raw_errors.csv'))
    parser.add_argument('--losses-v1-csv',
                        default=os.path.join(est, 'lstm_v1_losses.csv'))
    parser.add_argument('--losses-v2-csv',
                        default=os.path.join(est, 'lstm_v2_losses.csv'))
    parser.add_argument('--out-dir', default=est)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    produced = []

    # ── Figures 1 & 2 (require jumps CSV + CI CSV) ────────────────────────────
    if os.path.exists(args.jumps_csv) and os.path.exists(args.ci_csv):
        df_j  = load_jumps(args.jumps_csv)
        df_ci = load_bootstrap_ci(args.ci_csv)
        print(f"Loaded: {len(df_j)} jump rows, methods: {_methods_present(df_j)}")

        p1 = os.path.join(args.out_dir, 'robustness_degradation_ci.png')
        p2 = os.path.join(args.out_dir, 'robustness_frontier.png')
        plot_degradation_ci(df_j, df_ci, out_path=p1)
        plot_efficiency_robustness_frontier(df_ci, out_path=p2)
        produced += [p1, p2]
    else:
        print('warn: jumps CSV or CI CSV missing — skipping Figs 1 & 2.')

    # ── Figure 3 (requires raw per-path errors CSV) ───────────────────────────
    if os.path.exists(args.raw_csv):
        df_raw = pd.read_csv(args.raw_csv)
        df_raw['jump_pct'] = (df_raw['jump_rate'] * 100).round().astype(int)
        print(f"Loaded: {len(df_raw)} raw error rows")

        p3a = os.path.join(args.out_dir, 'robustness_violin_theta.png')
        p3b = os.path.join(args.out_dir, 'robustness_violin_sigma.png')
        plot_violin_errors(df_raw, 'theta', out_path=p3a)
        plot_violin_errors(df_raw, 'sigma', out_path=p3b)
        produced += [p3a, p3b]
    else:
        print(f'warn: {args.raw_csv} not found — skipping Fig 3 (violin plots).')
        print('      Run: python estimation/bootstrap_ci.py --save-raw')

    # ── Figure 4 (requires loss CSVs from training scripts) ───────────────────
    if os.path.exists(args.losses_v1_csv) and os.path.exists(args.losses_v2_csv):
        df_v1 = pd.read_csv(args.losses_v1_csv)
        df_v2 = pd.read_csv(args.losses_v2_csv)
        print(f"Loaded: v1 losses ({len(df_v1)} epochs), v2 losses ({len(df_v2)} epochs)")

        p4 = os.path.join(args.out_dir, 'training_loss_curves.png')
        plot_training_curves(df_v1, df_v2, out_path=p4)
        produced.append(p4)
    else:
        print('warn: loss CSVs not found — skipping Fig 4 (training curves).')
        print('      Re-run: python estimation/train_lstm.py')
        print('              python estimation/train_lstm_robust.py')

    # ── Legacy stratified plot ─────────────────────────────────────────────────
    if os.path.exists(args.stratified_csv):
        df_s = load_jumps(args.stratified_csv)
        if 'confidence' not in df_s.columns:
            print('warn: stratified CSV has no confidence column — skipping.')
        else:
            p5 = os.path.join(args.out_dir, 'robustness_stratified.png')
            plot_stratified(df_s, out_path=p5)
            produced.append(p5)

    print(f'\n{len(produced)} figure(s) saved.')
    plt.show()


if __name__ == '__main__':
    main()
