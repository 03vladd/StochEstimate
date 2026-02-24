"""
Maximum Likelihood Estimation for Ornstein-Uhlenbeck Process

Estimates three OU parameters (theta, mu, sigma) from discrete observations
using exact discrete likelihood formulation.

Theory:
For OU process: dX_t = theta(mu - X_t)dt + sigma dW_t
Discrete observations at times t_0, t_1, ..., t_n with spacing dt:

X_{t+dt} | X_t ~ N(m_t, v_t)

where:
  m_t = mu + (X_t - mu)*exp(-theta*dt)          [conditional mean]
  v_t = sigma^2/(2*theta) * (1 - exp(-2*theta*dt))  [conditional variance]

Log-likelihood:
  log L = sum_i [ -0.5*log(v_i) - 0.5*(X_{i+1} - m_i)^2 / v_i ]
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class OUEstimationResult:
    """Result of OU parameter estimation"""
    theta: float  # Speed of mean reversion (per day)
    mu: float  # Long-term mean level
    sigma: float  # Volatility (diffusion coefficient)

    log_likelihood: float  # Log-likelihood at optimal parameters
    success: bool  # Whether optimization converged
    message: str  # Optimization message

    # Confidence intervals (95%)
    theta_ci: tuple  # (lower, upper)
    mu_ci: tuple
    sigma_ci: tuple

    @property
    def detailed_result(self) -> str:
        """Human-readable result summary"""
        lines = [
            f"OU PARAMETER ESTIMATION (Exact Discrete Likelihood)",
            f"",
            f"Parameter Estimates:",
            f"  θ (mean reversion speed): {self.theta:.6f} per day",
            f"    95% CI: [{self.theta_ci[0]:.6f}, {self.theta_ci[1]:.6f}]",
            f"    Interpretation: {1 / self.theta:.2f} days for 63% reversion",
            f"",
            f"  μ (long-term mean): {self.mu:.6f}",
            f"    95% CI: [{self.mu_ci[0]:.6f}, {self.mu_ci[1]:.6f}]",
            f"",
            f"  σ (volatility): {self.sigma:.6f}",
            f"    95% CI: [{self.sigma_ci[0]:.6f}, {self.sigma_ci[1]:.6f}]",
            f"",
            f"Optimization Status: {'✓ Converged' if self.success else '✗ Failed'}",
            f"Log-Likelihood: {self.log_likelihood:.4f}",
        ]
        return "\n".join(lines)


def estimate_ou_mle(
        spread: pd.Series,
        times: np.ndarray = None,
        initial_guess: tuple = None,
        verbose: bool = True
) -> OUEstimationResult:
    """
    Estimate OU parameters using exact discrete likelihood.

    Args:
        spread: pd.Series or array of observations (can be positive or negative)
        times: array of observation times in days (default: 0, 1, 2, ...)
        initial_guess: tuple (theta_init, mu_init, sigma_init)
        verbose: print details during optimization

    Returns:
        OUEstimationResult with parameter estimates and confidence intervals
    """

    # Convert to numpy array
    X = np.asarray(spread.values, dtype=float)
    n = len(X)

    # Default: equally spaced by 1 day
    if times is None:
        times = np.arange(n, dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    # Compute time differences (in days)
    dt = np.diff(times)

    # Default initial guess
    if initial_guess is None:
        # theta: estimate from autocorrelation
        mu_init = np.mean(X)
        sigma_init = np.std(X)
        theta_init = 0.1  # Conservative guess
        initial_guess = (theta_init, mu_init, sigma_init)

    # Negative log-likelihood function (we minimize, so use negative)
    def neg_log_likelihood(params):
        theta, mu, sigma = params

        # Parameter constraints
        if theta <= 0 or sigma <= 0:
            return 1e10  # Penalty for invalid parameters

        # Compute conditional means and variances for each transition
        means = np.zeros(n - 1)
        variances = np.zeros(n - 1)

        for i in range(n - 1):
            dt_i = dt[i]
            exp_neg_theta_dt = np.exp(-theta * dt_i)

            # Conditional mean: m_t = mu + (X_t - mu)*exp(-theta*dt)
            means[i] = mu + (X[i] - mu) * exp_neg_theta_dt

            # Conditional variance: v_t = sigma^2/(2*theta) * (1 - exp(-2*theta*dt))
            # Handle numerical issues when theta is very small
            if theta < 1e-4:
                # Use Taylor expansion: 1 - exp(-2*theta*dt) ≈ 2*theta*dt
                variances[i] = sigma ** 2 * dt_i
            else:
                exp_neg_2theta_dt = np.exp(-2 * theta * dt_i)
                variances[i] = (sigma ** 2 / (2 * theta)) * (1 - exp_neg_2theta_dt)

        # Avoid division by zero
        if np.any(variances <= 0):
            return 1e10

        # Compute log-likelihood
        # log L = sum_i [ -0.5*log(v_i) - 0.5*(X_{i+1} - m_i)^2 / v_i ]
        residuals = X[1:] - means
        log_likelihood = -0.5 * np.sum(np.log(variances)) - 0.5 * np.sum((residuals ** 2) / variances)

        return -log_likelihood  # Negative because we minimize

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8}
    )

    theta_opt, mu_opt, sigma_opt = result.x
    log_lik_opt = -result.fun

    if verbose:
        print(f"  Optimization: {'✓ Converged' if result.success else '⚠ Did not converge'}")
        print(f"  Initial: θ={initial_guess[0]:.4f}, μ={initial_guess[1]:.4f}, σ={initial_guess[2]:.4f}")
        print(f"  Final:   θ={theta_opt:.4f}, μ={mu_opt:.4f}, σ={sigma_opt:.4f}")

    # Compute confidence intervals via numerical Hessian (Fisher Information)
    # This is approximate but practical
    eps = 1e-5

    params_opt = np.array([theta_opt, mu_opt, sigma_opt])
    f_center = neg_log_likelihood(params_opt)

    def hessian_element(params, i, j):
        """Compute Hessian element H[i,j] via central finite differences.

        Diagonal (i==j):  (f(p+e) - 2f(p) + f(p-e)) / e²
        Off-diagonal:     (f(p+ei+ej) - f(p+ei-ej) - f(p-ei+ej) + f(p-ei-ej)) / (4e²)
        """
        if i == j:
            p_plus = params.copy(); p_plus[i] += eps
            p_minus = params.copy(); p_minus[i] -= eps
            return (neg_log_likelihood(p_plus) - 2 * f_center + neg_log_likelihood(p_minus)) / eps ** 2
        else:
            p_pp = params.copy(); p_pp[i] += eps; p_pp[j] += eps
            p_pm = params.copy(); p_pm[i] += eps; p_pm[j] -= eps
            p_mp = params.copy(); p_mp[i] -= eps; p_mp[j] += eps
            p_mm = params.copy(); p_mm[i] -= eps; p_mm[j] -= eps
            return (neg_log_likelihood(p_pp) - neg_log_likelihood(p_pm)
                    - neg_log_likelihood(p_mp) + neg_log_likelihood(p_mm)) / (4 * eps ** 2)

    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            H[i, j] = hessian_element(params_opt, i, j)

    # Fisher Information = Hessian (at MLE)
    # Covariance = inverse of Fisher Information
    try:
        cov_matrix = np.linalg.inv(H)
        std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to rough scale estimates
        # mu SE: spread std dev / sqrt(n) (uncertainty in sample mean)
        std_errors = np.array([theta_opt * 0.1, np.std(X) / np.sqrt(n), sigma_opt * 0.1])

    # 95% confidence intervals (approximately ±1.96 * std_error)
    z_95 = 1.96
    theta_ci = (theta_opt - z_95 * std_errors[0], theta_opt + z_95 * std_errors[0])
    mu_ci = (mu_opt - z_95 * std_errors[1], mu_opt + z_95 * std_errors[1])
    sigma_ci = (sigma_opt - z_95 * std_errors[2], sigma_opt + z_95 * std_errors[2])

    # Ensure positive bounds for theta and sigma
    theta_ci = (max(theta_ci[0], 0.001), theta_ci[1])
    sigma_ci = (max(sigma_ci[0], 0.001), sigma_ci[1])

    return OUEstimationResult(
        theta=theta_opt,
        mu=mu_opt,
        sigma=sigma_opt,
        log_likelihood=log_lik_opt,
        success=result.success,
        message=result.message,
        theta_ci=theta_ci,
        mu_ci=mu_ci,
        sigma_ci=sigma_ci
    )