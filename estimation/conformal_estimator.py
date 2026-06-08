"""
Split-conformal prediction wrapper for OULSTMEstimator.

How it works:
  1. Calibrate on a held-out set of paths with known true parameters:
       - Collect absolute residuals |theta_hat_i - theta*_i| across N_cal paths
       - Set radius r_theta = (1-alpha) quantile of those residuals
       - Same for sigma
  2. At test time, replace MC Dropout CIs with symmetric conformal CIs:
       theta_ci = (theta_hat - r_theta,  theta_hat + r_theta)

Coverage guarantee (Vovk et al. 2005):
  If calibration and test series are exchangeable, empirical coverage >= 1-alpha
  in finite samples, unconditionally — no distributional assumptions needed.

The point estimates (theta, sigma) from the base LSTM are unchanged.
"""

import numpy as np
import pandas as pd
from dataclasses import replace as dc_replace
from typing import Optional

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimation.lstm_estimator import OULSTMEstimator, OULSTMResult


class ConformalOUEstimator:
    """
    Wraps OULSTMEstimator with split-conformal prediction intervals.

    The MC Dropout CIs in OULSTMResult are replaced with conformal CIs whose
    width is set by empirical residuals on a held-out calibration set, giving
    guaranteed finite-sample coverage >= 1 - alpha.
    """

    def __init__(self, base_estimator: OULSTMEstimator, alpha: float = 0.05):
        self.base = base_estimator
        self.alpha = alpha
        self.radius_theta: Optional[float] = None
        self.radius_sigma: Optional[float] = None
        self._n_cal: int = 0

    def calibrate(self, cal_series: list, true_thetas: list, true_sigmas: list,
                  n_mc_samples: int = 50) -> None:
        """
        Fit conformal radii from a held-out calibration set.

        Uses the ceil((n+1)(1-alpha))/n quantile of absolute residuals,
        which is the standard finite-sample correction for split conformal.

        cal_series  : list of pd.Series, one per calibration path
        true_thetas : ground-truth theta for each path
        true_sigmas : ground-truth sigma for each path
        """
        res_theta, res_sigma = [], []

        for series, theta_true, sigma_true in zip(cal_series, true_thetas, true_sigmas):
            try:
                r = self.base.estimate(series, n_mc_samples=n_mc_samples)
                res_theta.append(abs(r.theta - theta_true))
                res_sigma.append(abs(r.sigma - sigma_true))
            except Exception:
                pass

        n = len(res_theta)
        if n == 0:
            raise ValueError("No valid calibration estimates — check model and input.")

        # Finite-sample correction: (n+1)/n factor, capped at 1.0
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.radius_theta = float(np.quantile(res_theta, q_level))
        self.radius_sigma = float(np.quantile(res_sigma, q_level))
        self._n_cal = n

        print(f"  Conformal calibration complete: n_cal={n}, "
              f"r_theta={self.radius_theta:.4f}, r_sigma={self.radius_sigma:.4f}")

    def estimate(self, series: pd.Series, n_mc_samples: int = 50) -> OULSTMResult:
        """
        Estimate OU parameters. Point estimates are from the base LSTM;
        theta_ci and sigma_ci are replaced with conformal intervals.
        """
        if self.radius_theta is None:
            raise RuntimeError("Call calibrate() before estimate().")

        r = self.base.estimate(series, n_mc_samples=n_mc_samples)

        return dc_replace(r,
            theta_ci=(max(r.theta - self.radius_theta, 1e-6),
                      r.theta + self.radius_theta),
            sigma_ci=(max(r.sigma - self.radius_sigma, 1e-6),
                      r.sigma + self.radius_sigma),
        )

    @property
    def n_cal(self) -> int:
        return self._n_cal

    def summary(self) -> str:
        if self.radius_theta is None:
            return "ConformalOUEstimator (not yet calibrated)"
        return (f"ConformalOUEstimator | n_cal={self._n_cal} | alpha={self.alpha} | "
                f"r_theta={self.radius_theta:.4f} | r_sigma={self.radius_sigma:.4f}")