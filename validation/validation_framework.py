"""
OU Validation Framework

Master module that combines all validation tests:
- Stationarity (ADF test)
- Linear drift
- Constant volatility
- Autocorrelation structure (exponential fit)

Plus auxiliary normality test for informational purposes.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict

from .stationarity_test import test_stationarity_adf, StationarityTestResult
from .linear_drift_test import test_linear_drift, LinearDriftTestResult
from .constant_volatility import test_constant_volatility, ConstantVolatilityTestResult
from .acf_exponential import test_autocorrelation_exponential, ACFExponentialFitResult
from .normality_test import test_normality, NormalityTestResult
from visualization.validation_visualization import plot_validation_report, plot_comparison


@dataclass
class ValidationReport:
    """Complete validation report for a time series"""
    series_name: str
    n_observations: int

    # Core test results
    stationarity: StationarityTestResult
    linear_drift: LinearDriftTestResult
    constant_volatility: ConstantVolatilityTestResult
    autocorrelation: ACFExponentialFitResult

    # Auxiliary test results
    normality: NormalityTestResult

    def overall_pass(self) -> bool:
        """Returns True if all core tests pass"""
        return (
            self.stationarity.passed and
            self.linear_drift.passed and
            self.constant_volatility.passed and
            self.autocorrelation.passed
        )

    def count_passing_tests(self) -> int:
        """Count how many core tests pass"""
        count = 0
        if self.stationarity.passed:
            count += 1
        if self.linear_drift.passed:
            count += 1
        if self.constant_volatility.passed:
            count += 1
        if self.autocorrelation.passed:
            count += 1
        return count

    def get_confidence_level(self) -> tuple:
        """
        Determine confidence level for OU process classification

        Returns:
            (confidence_level, description)
            confidence_level: "HIGH", "MEDIUM", "LOW", or "NOT_OU"
        """
        passing_tests = self.count_passing_tests()

        # Stationarity is non-negotiable
        if not self.stationarity.passed:
            return ("NOT_OU", "Series is non-stationary (unit root detected)")

        # ACF structure is critical for OU identification
        if not self.autocorrelation.passed:
            return ("NOT_OU", "No exponential autocorrelation decay (not mean-reverting)")

        # If stationarity and ACF pass, classify by how many total tests pass
        if passing_tests == 4:
            return ("HIGH", "All core tests pass - strong OU characteristics")
        elif passing_tests == 3:
            return ("MEDIUM", "3/4 core tests pass - OU with some marginal characteristics")
        elif passing_tests == 2:
            return ("LOW", "2/4 core tests pass - weak OU signal but mean-reverting")
        else:
            # This shouldn't happen given stationarity and ACF checks above
            return ("LOW", "Minimal OU characteristics detected")

    def get_recommendation(self) -> str:
        """Get recommendation for parameter estimation based on confidence level"""
        confidence, _ = self.get_confidence_level()

        if confidence == "HIGH":
            return (
                "✓ Suitable for standard parameter estimation (MLE or LSTM)\n"
                "  Use point estimates with standard confidence intervals"
            )
        elif confidence == "MEDIUM":
            return (
                "⚠ Suitable for estimation with caution\n"
                "  Recommend: Ensemble methods (MLE + LSTM), widen confidence intervals by 15-20%"
            )
        elif confidence == "LOW":
            return (
                "⚠ Use only if combined with other signals\n"
                "  Recommend: Multiple validation sources, high uncertainty tolerance,\n"
                "  consider as part of ensemble strategy"
            )
        else:
            return (
                "✗ Not suitable for OU-based estimation\n"
                "  Consider alternative modeling approaches"
            )

    def summary(self) -> str:
        """Returns a human-readable summary"""
        confidence_level, confidence_desc = self.get_confidence_level()
        recommendation = self.get_recommendation()

        lines = [
            f"\n{'='*70}",
            f"OU VALIDATION REPORT: {self.series_name}",
            f"Observations: {self.n_observations}",
            f"{'='*70}",
            f"\nCORE TESTS (Required for OU):",
            f"{'-'*70}"
        ]

        # Core tests
        tests = [
            ("Stationarity (ADF)", self.stationarity),
            ("Linear Drift", self.linear_drift),
            ("Constant Volatility", self.constant_volatility),
            ("Autocorrelation (Exponential Fit)", self.autocorrelation)
        ]

        for test_name, result in tests:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"\n{status} | {test_name}")

        # Confidence level
        lines.append(f"\n{'-'*70}")
        lines.append(f"Tests Passed: {self.count_passing_tests()}/4")
        lines.append(f"Confidence Level: {confidence_level}")
        lines.append(f"Description: {confidence_desc}")

        # Auxiliary test
        lines.append(f"\n{'='*70}")
        lines.append(f"AUXILIARY TEST (Informational):")
        lines.append(f"{'-'*70}")

        normality_status = "✓ PASS" if self.normality.passed else "⚠ WARNING"
        lines.append(f"\n{normality_status} | Normality of Returns")

        if self.normality.passed:
            lines.append(f"  Returns appear normally distributed")
        else:
            lines.append(f"  ⚠ Returns deviate from normality")
            lines.append(f"    Skewness: {self.normality.skewness:.4f}")
            lines.append(f"    Kurtosis (excess): {self.normality.kurtosis:.4f}")
            if self.normality.kurtosis > 1:
                lines.append(f"    → Fat tails detected (more extreme events than normal)")

        # Recommendation
        lines.append(f"\n{'='*70}")
        lines.append(f"RECOMMENDATION FOR PARAMETER ESTIMATION:")
        lines.append(f"{'-'*70}")
        lines.append(recommendation)

        lines.append(f"\n{'='*70}\n")

        return "\n".join(lines)

    def detailed_results(self) -> str:
        """Returns detailed results for each test"""
        lines = [
            f"\n{'='*70}",
            f"DETAILED RESULTS: {self.series_name}",
            f"{'='*70}"
        ]

        # Core tests
        tests = [
            ("STATIONARITY TEST (ADF)", self.stationarity),
            ("LINEAR DRIFT TEST", self.linear_drift),
            ("CONSTANT VOLATILITY TEST", self.constant_volatility),
            ("AUTOCORRELATION TEST (Exponential Fit)", self.autocorrelation)
        ]

        for test_name, result in tests:
            lines.append(f"\n{test_name}")
            lines.append("-" * 70)
            lines.append(result.detailed_result)

        # Normality
        lines.append(f"\nNORMALITY TEST (Auxiliary)")
        lines.append("-" * 70)
        lines.append(self.normality.detailed_result)

        return "\n".join(lines)

    def plot_diagnostics(self, series: pd.Series, save_path: str = None):
        """Create diagnostic plots for this report"""
        return plot_validation_report(series, self, save_path)


def validate_series(series: pd.Series, name: str = "Series") -> ValidationReport:
    """
    Run complete OU validation on a time series

    Args:
        series: pandas Series with time series data
        name: Name for reporting

    Returns:
        ValidationReport with all test results
    """
    # Run all tests
    stationarity_result = test_stationarity_adf(series)
    linear_drift_result = test_linear_drift(series)
    constant_vol_result = test_constant_volatility(series)
    autocorr_result = test_autocorrelation_exponential(series)
    normality_result = test_normality(series)

    # Create report
    report = ValidationReport(
        series_name=name,
        n_observations=len(series),
        stationarity=stationarity_result,
        linear_drift=linear_drift_result,
        constant_volatility=constant_vol_result,
        autocorrelation=autocorr_result,
        normality=normality_result
    )

    return report