"""
Validation Framework Visualization

Creates diagnostic plots for each validation test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import curve_fit


def plot_validation_report(series: pd.Series, report, save_path: str = None):
    """
    Create comprehensive diagnostic plots for validation report

    Args:
        series: pandas Series with time series data
        report: ValidationReport object from validate_series()
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    fig.suptitle(f'OU Validation Diagnostics: {report.series_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Time series
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(series.index, series.values, linewidth=1)
    ax1.set_title('Time Series', fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation summary (text box)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = f"""CORE TESTS:
Stationarity: {'✓' if report.stationarity.passed else '✗'}
Linear Drift: {'✓' if report.linear_drift.passed else '✗'}
Volatility: {'✓' if report.constant_volatility.passed else '✗'}
ACF Structure: {'✓' if report.autocorrelation.passed else '✗'}

Overall: {'PASS' if report.overall_pass() else 'FAIL'}

AUXILIARY:
Normality: {'✓' if report.normality.passed else '⚠'}"""
    ax2.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Linear drift (regression line)
    ax3 = fig.add_subplot(gs[1, 0])
    t = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = linregress(t, series.values)

    ax3.scatter(t, series.values, alpha=0.5, s=10)
    ax3.plot(t, intercept + slope*t, 'r-', linewidth=2, label=f'Trend (slope={slope:.4f})')
    ax3.set_title('Linear Drift Test', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Constant volatility (first vs second half)
    ax4 = fig.add_subplot(gs[1, 1])
    mid = len(series) // 2
    first_half = series.iloc[:mid].values
    second_half = series.iloc[mid:].values

    ax4.hist(first_half, bins=30, alpha=0.6, label=f'First half (σ={np.std(first_half):.3f})')
    ax4.hist(second_half, bins=30, alpha=0.6, label=f'Second half (σ={np.std(second_half):.3f})')
    ax4.set_title('Volatility Comparison', fontweight='bold')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Returns distribution (normality)
    ax5 = fig.add_subplot(gs[1, 2])
    returns = series.diff().dropna()
    ax5.hist(returns, bins=50, density=True, alpha=0.7, label='Observed')

    # Overlay normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax5.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax5.set_title('Normality Test (Returns)', fontweight='bold')
    ax5.set_xlabel('Returns')
    ax5.set_ylabel('Density')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: ACF with exponential fit
    ax6 = fig.add_subplot(gs[2, 0])
    from statsmodels.graphics.tsaplots import acf
    acf_values = acf(series, nlags=20, fft=False)
    lags = np.arange(len(acf_values))

    ax6.stem(lags, acf_values, basefmt=' ')

    # Overlay exponential fit if available
    if report.autocorrelation.theta is not None:
        theta = report.autocorrelation.theta
        fitted_acf = np.exp(-theta * lags)
        ax6.plot(lags, fitted_acf, 'r-', linewidth=2,
                label=f'Fit: exp(-{theta:.4f}k), R²={report.autocorrelation.r_squared:.3f}')

    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.set_title('ACF & Exponential Fit', fontweight='bold')
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('ACF')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Stationarity indicator (ADF statistic)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')

    adf_text = f"""ADF TEST (Stationarity)
Statistic: {report.stationarity.adf_statistic:.4f}
P-value: {report.stationarity.p_value:.4f}
Critical (5%): {report.stationarity.critical_value_5pct:.4f}

Result: {'STATIONARY' if report.stationarity.passed else 'NON-STATIONARY'}"""

    ax7.text(0.05, 0.5, adf_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax7.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 8: Test parameters summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    theta_str = f"{report.autocorrelation.theta:.6f}" if report.autocorrelation.theta else 'N/A'
    params_text = f"""TEST PARAMETERS

Linear Drift Slope: {report.linear_drift.slope:.6f}
Vol Ratio (max/min): {report.constant_volatility.volatility_ratio:.4f}
ACF Theta: {theta_str}
Return Kurtosis: {report.normality.kurtosis:.4f}"""

    ax8.text(0.05, 0.5, params_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax8.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Diagnostic plot saved to {save_path}")

    return fig


def plot_comparison(reports_dict: dict, save_path: str = None):
    """
    Create comparison plots for multiple validation reports

    Args:
        reports_dict: Dictionary of {series_name: ValidationReport}
        save_path: Path to save figure (optional)
    """
    n_series = len(reports_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OU Validation Framework - Multiple Series Comparison',
                 fontsize=14, fontweight='bold')

    series_names = list(reports_dict.keys())
    colors = ['green' if reports_dict[name].overall_pass() else 'red' for name in series_names]
    passes = [reports_dict[name].overall_pass() for name in series_names]

    # Plot 1: Overall results
    ax = axes[0, 0]
    y_pos = np.arange(len(series_names))
    pass_counts = [sum([getattr(reports_dict[name], test).passed
                       for test in ['stationarity', 'linear_drift', 'constant_volatility', 'autocorrelation']])
                   for name in series_names]
    ax.barh(y_pos, pass_counts, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(series_names)
    ax.set_xlabel('Tests Passed (out of 4)')
    ax.set_title('Core Tests Passed', fontweight='bold')
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Stationarity results
    ax = axes[0, 1]
    stationarity_results = [reports_dict[name].stationarity.passed for name in series_names]
    colors_stat = ['green' if x else 'red' for x in stationarity_results]
    ax.bar(range(len(series_names)), [1]*len(series_names), color=colors_stat, alpha=0.7)
    ax.set_xticks(range(len(series_names)))
    ax.set_xticklabels(series_names, rotation=45, ha='right')
    ax.set_title('Stationarity Test', fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    # Plot 3: Volatility ratio
    ax = axes[1, 0]
    vol_ratios = [reports_dict[name].constant_volatility.volatility_ratio for name in series_names]
    bars = ax.bar(range(len(series_names)), vol_ratios, alpha=0.7)
    for i, bar in enumerate(bars):
        bar.set_color('green' if reports_dict[series_names[i]].constant_volatility.passed else 'red')
    ax.axhline(y=1.2, color='orange', linestyle='--', linewidth=1, label='Threshold region')
    ax.set_xticks(range(len(series_names)))
    ax.set_xticklabels(series_names, rotation=45, ha='right')
    ax.set_ylabel('Volatility Ratio (max/min)')
    ax.set_title('Volatility Constancy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8)

    # Plot 4: ACF theta (mean reversion speed)
    ax = axes[1, 1]
    acf_thetas = [reports_dict[name].autocorrelation.theta if reports_dict[name].autocorrelation.theta else 0
                  for name in series_names]
    bars = ax.bar(range(len(series_names)), acf_thetas, alpha=0.7)
    for i, bar in enumerate(bars):
        bar.set_color('green' if reports_dict[series_names[i]].autocorrelation.passed else 'red')
    ax.set_xticks(range(len(series_names)))
    ax.set_xticklabels(series_names, rotation=45, ha='right')
    ax.set_ylabel('Theta (Mean Reversion Speed)')
    ax.set_title('ACF Exponential Fit', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")

    return fig