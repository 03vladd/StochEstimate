"""
Master Validation Test

Tests synthetic OU processes through the complete validation framework.
Demonstrates that the framework correctly identifies OU processes.
"""

import numpy as np
import pandas as pd
from synthetic_data.ou_generator import generate_ou_process, generate_random_walk
from validation.validation_framework import validate_series
from visualization.validation_visualization import plot_comparison


def main():
    print("=" * 70)
    print("MASTER VALIDATION FRAMEWORK TEST")
    print("Testing synthetic OU processes")
    print("=" * 70)

    # Test 1: True OU process with strong mean reversion
    print("\n" + "=" * 70)
    print("TEST 1: True OU Process (θ=0.5)")
    print("=" * 70)

    ou_strong = generate_ou_process(mu=0.0, theta=0.5, sigma=0.5, n_steps=1000, seed=42)
    report1 = validate_series(ou_strong, "OU (θ=0.5)")

    print(report1.summary())
    print(f"Result: {'✓ PASS' if report1.overall_pass() else '✗ FAIL'}")

    # Test 2: True OU process with weak mean reversion
    print("\n" + "=" * 70)
    print("TEST 2: True OU Process (θ=0.1)")
    print("=" * 70)

    ou_weak = generate_ou_process(mu=0.0, theta=0.1, sigma=0.5, n_steps=1000, seed=42)
    report2 = validate_series(ou_weak, "OU (θ=0.1)")

    print(report2.summary())
    print(f"Result: {'✓ PASS' if report2.overall_pass() else '✗ FAIL'}")

    # Test 3: Non-stationary random walk
    print("\n" + "=" * 70)
    print("TEST 3: Non-Stationary Random Walk")
    print("=" * 70)

    rw = generate_random_walk(n_steps=1000, drift=0.0, volatility=1.0, seed=42)
    report3 = validate_series(rw, "Random Walk")

    print(report3.summary())
    print(f"Result: {'✓ PASS' if report3.overall_pass() else '✗ FAIL'}")

    # Test 4: Very weak mean reversion (near random walk)
    print("\n" + "=" * 70)
    print("TEST 4: Very Weak OU (θ=0.01)")
    print("=" * 70)

    ou_very_weak = generate_ou_process(mu=0.0, theta=0.01, sigma=0.5, n_steps=1000, seed=42)
    report4 = validate_series(ou_very_weak, "OU (θ=0.01)")

    print(report4.summary())
    print(f"Result: {'✓ PASS' if report4.overall_pass() else '✗ FAIL'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL TESTS")
    print("=" * 70)

    tests = [
        ("OU (θ=0.5, strong)", report1, True),
        ("OU (θ=0.1, medium)", report2, True),
        ("Random Walk", report3, False),
        ("OU (θ=0.01, weak)", report4, False),
    ]

    print(f"\n{'Series':<25} {'Expected':<10} {'Actual':<10} {'Correct':<10}")
    print("-" * 60)

    correct_count = 0
    for name, report, expected_pass in tests:
        actual = "PASS" if report.overall_pass() else "FAIL"
        expected = "PASS" if expected_pass else "FAIL"
        correct = report.overall_pass() == expected_pass

        if correct:
            correct_count += 1

        symbol = "✓" if correct else "✗"
        print(f"{name:<25} {expected:<10} {actual:<10} {symbol:<10}")

    accuracy = correct_count / len(tests)
    print("-" * 60)
    print(f"Framework Accuracy: {accuracy:.0%} ({correct_count}/{len(tests)})")

    # Detailed results for first test
    print("\n" + "=" * 70)
    print("DETAILED RESULTS FOR TEST 1 (OU θ=0.5)")
    print("=" * 70)
    print(report1.detailed_results())

    # Generate diagnostic plots
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    report1.plot_diagnostics(ou_strong, save_path="diagnostic_ou_strong.png")
    report2.plot_diagnostics(ou_weak, save_path="diagnostic_ou_weak.png")
    report3.plot_diagnostics(rw, save_path="diagnostic_random_walk.png")
    report4.plot_diagnostics(ou_very_weak, save_path="diagnostic_ou_very_weak.png")

    # Create comparison plot
    reports_dict = {
        "OU (θ=0.5)": report1,
        "OU (θ=0.1)": report2,
        "Random Walk": report3,
        "OU (θ=0.01)": report4
    }
    plot_comparison(reports_dict, save_path="validation_comparison.png")

    print("\n✓ All plots saved")


if __name__ == "__main__":
    main()