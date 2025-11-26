StochEstimate
A validation-first framework for estimating Ornstein-Uhlenbeck process parameters in financial pairs trading. Before estimating, we verify the assumptions.
The Problem
Most research on Ornstein-Uhlenbeck (OU) parameter estimation makes a dangerous assumption: that financial time series actually follow OU dynamics. They don't validate this first—they just assume it and proceed directly to estimation.
This is why sophisticated methods sometimes perform worse than naive approaches during market regime shifts. If your data doesn't satisfy OU assumptions, your parameters are meaningless.
StochEstimate's approach: Validate that OU assumptions hold before attempting estimation. This catches misspecified models early and prevents wasted computation on data that violates the fundamental assumptions.
What We're Building
A Python-based platform that lets users:

Validate: Test whether a financial time series actually follows OU dynamics
Estimate: Extract parameters using Maximum Likelihood Estimation
Compare: Evaluate multiple estimation approaches (with neural network alternatives coming)
Backtest: Assess trading performance on real financial data

Currently implemented: full validation framework + MLE parameter estimation via command-line interface.
