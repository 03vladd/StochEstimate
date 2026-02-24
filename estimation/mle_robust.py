"""
Robust MLE for Ornstein-Uhlenbeck process — Student-t conditional likelihood

Replaces the Gaussian conditional density in standard OU-MLE with a Student-t:

  X_{t+dt} | X_t  ~  t_ν( location = m_t,  scale = sqrt(v_t) )

where m_t and v_t are the exact OU conditional moments (same as Gaussian MLE):

  m_t = μ + (X_t − μ) · exp(−θ·Δt)
  v_t = σ²/(2θ) · (1 − exp(−2θ·Δt))

This downweights large residuals relative to the Gaussian quadratic penalty:
  Gaussian:   penalty ∝  r²/v
  Student-t:  penalty ∝  (ν+1) · log(1 + r²/(ν·v))   [bounded for large r]

Effect: individual jump observations contribute a bounded amount to the total
log-likelihood, rather than dominating it quadratically, making the estimator
robust to outliers (Muler, Peña & Yohai 2009; Barndorff-Nielsen & Shephard 2001).

Degrees-of-freedom ν controls tail weight:
  ν → ∞  : converges to Gaussian MLE (not robust)
  ν = 30 : nearly Gaussian, slight robustness
  ν = 8  : light heavy tails
  ν = 4  : moderate heavy tails — standard in financial econometrics
  ν = 2  : very heavy tails (variance barely defined)

Interface is identical to estimate_ou_mle() for drop-in comparison.
Returns OUEstimationResult so all downstream code is unchanged.

References:
  Barndorff-Nielsen & Shephard (2001), JRSS-B 63(2)
  Muler, Peña & Yohai (2009), Annals of Statistics 37(2)
  Harvey & Luati (2014), JASA 109(507)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as t_dist

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimation.mle import OUEstimationResult


def estimate_ou_t_mle(
        spread: pd.Series,
        df: float = 4.0,
        times: np.ndarray = None,
        initial_guess: tuple = None,
        verbose: bool = True
) -> OUEstimationResult:
    """
    Estimate OU parameters using Student-t conditional likelihood.

    Args:
        spread:        pd.Series of observations
        df:            Degrees of freedom for Student-t noise (default: 4).
                       Lower = heavier tails = more robustness to outliers.
                       Use df=30 to approximate Gaussian MLE as a sanity check.
        times:         Array of observation times (default: 0, 1, 2, ...)
        initial_guess: (theta_init, mu_init, sigma_init)
        verbose:       Print optimization status

    Returns:
        OUEstimationResult — same interface as estimate_ou_mle()
    """
    X = np.asarray(spread.values, dtype=float)
    n = len(X)

    if times is None:
        times = np.arange(n, dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    dt = np.diff(times)

    if initial_guess is None:
        mu_init    = np.mean(X)
        sigma_init = np.std(X)
        theta_init = 0.1
        initial_guess = (theta_init, mu_init, sigma_init)

    def neg_log_likelihood(params):
        theta, mu, sigma = params

        if theta <= 0 or sigma <= 0:
            return 1e10

        nll = 0.0
        for i in range(n - 1):
            dt_i = dt[i]
            exp_neg_theta_dt = np.exp(-theta * dt_i)

            # Exact OU conditional mean
            m_i = mu + (X[i] - mu) * exp_neg_theta_dt

            # Exact OU conditional variance
            if theta < 1e-4:
                v_i = sigma ** 2 * dt_i          # Taylor expansion
            else:
                v_i = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt_i))

            if v_i <= 0:
                return 1e10

            # Student-t log-likelihood: t_ν(location=m_i, scale=sqrt(v_i))
            # scipy.stats.t.logpdf(x, df, loc, scale)
            nll -= t_dist.logpdf(X[i + 1], df=df, loc=m_i, scale=np.sqrt(v_i))

        return nll

    result = minimize(
        neg_log_likelihood,
        x0=initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8}
    )

    theta_opt, mu_opt, sigma_opt = result.x
    log_lik_opt = -result.fun

    if verbose:
        print(f"  t-MLE (df={df}): {'✓ Converged' if result.success else '⚠ Did not converge'}")
        print(f"  Final: θ={theta_opt:.4f}  μ={mu_opt:.4f}  σ={sigma_opt:.4f}")

    # Confidence intervals via numerical Hessian (same approach as Gaussian MLE)
    eps = 1e-5
    params_opt = np.array([theta_opt, mu_opt, sigma_opt])
    f_center = neg_log_likelihood(params_opt)

    def hessian_element(params, i, j):
        if i == j:
            p_plus  = params.copy(); p_plus[i]  += eps
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

    try:
        cov_matrix = np.linalg.inv(H)
        std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    except np.linalg.LinAlgError:
        std_errors = np.array([theta_opt * 0.1, np.std(X) / np.sqrt(n), sigma_opt * 0.1])

    z_95 = 1.96
    theta_ci = (max(theta_opt - z_95 * std_errors[0], 0.001), theta_opt + z_95 * std_errors[0])
    mu_ci    = (mu_opt    - z_95 * std_errors[1], mu_opt    + z_95 * std_errors[1])
    sigma_ci = (max(sigma_opt - z_95 * std_errors[2], 0.001), sigma_opt + z_95 * std_errors[2])

    return OUEstimationResult(
        theta=theta_opt,
        mu=mu_opt,
        sigma=sigma_opt,
        log_likelihood=log_lik_opt,
        success=result.success,
        message=result.message,
        theta_ci=theta_ci,
        mu_ci=mu_ci,
        sigma_ci=sigma_ci,
    )
