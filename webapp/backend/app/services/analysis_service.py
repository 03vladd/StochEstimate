"""
Analysis pipeline — the heart of StochEstimate.

Runs entirely in a thread pool (via asyncio.to_thread) so it never blocks
the FastAPI event loop.  Each public entry point accepts an AsyncSession
for DB writes but does all heavy computation synchronously.

Pipeline stages:
  1. Fetch 2 years of daily prices via yfinance
  2. Engle-Granger cointegration test  → CointegrationResult
  3. OU validation battery             → ValidationResult
  4. LSTM-robust estimation            → EstimationResult
  5. Gaussian MLE (comparison panel)   → MLEResult
  6. Z-score signal series             → [Signal]
  7. Persist everything, mark job done
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services import narration_service

# ── Research-code path injection ──────────────────────────────────────────────
# The research modules live at the repo root, not in the backend package.
# We insert the research root once so all imports resolve cleanly.
_RESEARCH_ROOT = Path(settings.RESEARCH_ROOT)
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

# The visualization module is optional (it needs matplotlib which may not be
# installed in the backend venv).  We patch it in before importing
# validation_framework so the top-level import doesn't crash.
import types as _types

def _noop(*args, **kwargs):  # noqa: ANN
    pass

_viz_inner = _types.ModuleType("visualization.validation_visualization")
_viz_inner.plot_validation_report = _noop  # type: ignore[attr-defined]
_viz_inner.plot_comparison = _noop  # type: ignore[attr-defined]

_viz_stub = _types.ModuleType("visualization")
_viz_stub.validation_visualization = _viz_inner  # type: ignore[attr-defined]

sys.modules.setdefault("visualization", _viz_stub)
sys.modules.setdefault("visualization.validation_visualization", _viz_inner)

from preprocessing.engle_granger_cointegration import engle_granger_cointegration  # noqa: E402
from validation.validation_framework import validate_series  # noqa: E402
from estimation.lstm_estimator import OULSTMEstimator  # noqa: E402
from estimation.mle import estimate_ou_mle  # noqa: E402

from app.models.analysis import (  # noqa: E402
    AnalysisJob,
    CointegrationResult,
    EstimationResult,
    MLEResult,
    ValidationResult,
)
from app.models.pair import Pair, UserPair  # noqa: E402
from app.models.signal import Signal  # noqa: E402

# ── LSTM model (loaded once at import time) ────────────────────────────────────

_lstm_estimator: Optional[OULSTMEstimator] = None


def _get_lstm_estimator() -> OULSTMEstimator:
    global _lstm_estimator
    if _lstm_estimator is None:
        model_path = _RESEARCH_ROOT / settings.LSTM_MODEL_PATH
        estimator = OULSTMEstimator()
        estimator.load(str(model_path))
        _lstm_estimator = estimator
    return _lstm_estimator


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def get_or_create_pair(db: AsyncSession, ticker1: str, ticker2: str) -> Pair:
    result = await db.execute(
        select(Pair).where(Pair.ticker1 == ticker1, Pair.ticker2 == ticker2)
    )
    pair = result.scalar_one_or_none()
    if pair:
        return pair
    # Also check reversed order — normalise to alphabetical
    t1, t2 = (ticker1, ticker2) if ticker1 < ticker2 else (ticker2, ticker1)
    result = await db.execute(
        select(Pair).where(Pair.ticker1 == t1, Pair.ticker2 == t2)
    )
    pair = result.scalar_one_or_none()
    if pair:
        return pair

    pair = Pair(ticker1=t1, ticker2=t2)
    db.add(pair)
    await db.flush()
    return pair


async def get_or_create_user_pair(
    db: AsyncSession, user_id: uuid.UUID, pair_id: uuid.UUID
) -> UserPair:
    result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == user_id, UserPair.pair_id == pair_id
        )
    )
    up = result.scalar_one_or_none()
    if up:
        return up
    up = UserPair(user_id=user_id, pair_id=pair_id)
    db.add(up)
    await db.flush()
    return up


async def create_analysis_job(
    db: AsyncSession, pair_id: uuid.UUID, window_months: int = 12
) -> AnalysisJob:
    job = AnalysisJob(pair_id=pair_id, status="pending", window_months=window_months)
    db.add(job)
    await db.flush()
    return job


async def get_latest_job(db: AsyncSession, pair_id: uuid.UUID) -> Optional[AnalysisJob]:
    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.pair_id == pair_id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Signal computation ─────────────────────────────────────────────────────────

def _compute_signal_type(z: float, long_thr: float = 1.5, short_thr: float = 1.5) -> str:
    if z <= -long_thr:
        return "LONG"
    if z >= short_thr:
        return "SHORT"
    if abs(z) <= 0.5:
        return "EXIT"
    return "NONE"


def _compute_signals(
    spread: pd.Series,
    theta: float,
    mu: float,
    sigma: float,
) -> list[dict]:
    """
    Z-score normalisation:  z_t = (X_t - μ) / sqrt(σ² / 2θ)

    Returns list of dicts ready to build Signal rows.
    """
    stationary_std = float(np.sqrt(sigma ** 2 / (2 * theta))) if theta > 0 else float(sigma)
    records = []
    in_position = False  # True after LONG/SHORT, until EXIT fires
    for dt, val in spread.items():
        z = (float(val) - mu) / stationary_std if stationary_std > 0 else 0.0
        raw = _compute_signal_type(z)

        if raw in ("LONG", "SHORT"):
            in_position = True
            signal_type = raw
        elif raw == "EXIT" and in_position:
            in_position = False
            signal_type = "EXIT"
        else:
            signal_type = "NONE"

        records.append(
            {
                "date": dt.date() if hasattr(dt, "date") else dt,
                "z_score": z,
                "signal_type": signal_type,
                "spread_value": float(val),
                "stationary_mean": mu,
                "stationary_std": stationary_std,
            }
        )
    return records


# ── Core pipeline (synchronous, runs in thread pool) ──────────────────────────

def _run_pipeline_sync(
    pair_id: uuid.UUID, ticker1: str, ticker2: str, window_days: int = 365
) -> dict:
    """
    Returns a dict with keys: cointegration, validation, estimation, mle, signals
    Raises RuntimeError with a human-readable message on non-recoverable failure.
    """
    import yfinance as yf

    # 1. Fetch price data — explicit date window so we always get up to today
    import datetime as _dt
    end_date = _dt.date.today()
    start_date = end_date - _dt.timedelta(days=window_days)
    raw = yf.download(
        [ticker1, ticker2],
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker1}/{ticker2}")

    # Handle single-level columns (yfinance sometimes flattens)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][[ticker1, ticker2]].dropna()
    else:
        raise RuntimeError(f"Unexpected yfinance column format for {ticker1}/{ticker2}")

    if len(close) < 60:
        raise RuntimeError(f"Insufficient price history: only {len(close)} observations")

    price_a = close[ticker1]
    price_b = close[ticker2]

    # 2. Cointegration — test both directions, keep the lower ADF p-value
    coint_ab = engle_granger_cointegration(price_a.values, price_b.values)
    coint_ba = engle_granger_cointegration(price_b.values, price_a.values)
    if coint_ab.get("adf_pvalue", 1.0) <= coint_ba.get("adf_pvalue", 1.0):
        coint_result = coint_ab
    else:
        coint_result = coint_ba
    if not coint_result["cointegrated"]:
        raise RuntimeError(
            f"{ticker1}/{ticker2} is not cointegrated in either direction "
            f"(best p={coint_result['adf_pvalue']:.4f})"
        )
    spread_raw = pd.Series(coint_result["spread"], index=price_a.index)

    # 3. Validation
    report = validate_series(spread_raw, name=f"{ticker1}/{ticker2}")
    confidence_level, _ = report.get_confidence_level()

    # 4. LSTM-robust estimation
    estimator = _get_lstm_estimator()
    lstm_result = estimator.estimate(spread_raw, n_mc_samples=200)

    # 5. Gaussian MLE (silent)
    mle_result = estimate_ou_mle(spread_raw, verbose=False)

    # 6. Signals
    signal_records = _compute_signals(
        spread_raw,
        theta=lstm_result.theta,
        mu=lstm_result.mu,
        sigma=lstm_result.sigma,
    )

    return {
        "cointegration": coint_result,
        "validation": report,
        "confidence_level": confidence_level,
        "lstm": lstm_result,
        "mle": mle_result,
        "signals": signal_records,
    }


# ── Async orchestrator ────────────────────────────────────────────────────────

async def run_analysis_job(
    job_id: uuid.UUID,
    pair_id: uuid.UUID,
    ticker1: str,
    ticker2: str,
    db: AsyncSession,
    window_months: int = 12,
) -> None:
    """
    Called as a background task after the HTTP response has been sent.
    Marks the job running → runs pipeline in thread pool → persists results → marks complete.
    """
    # Mark running
    job_result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        return
    job.status = "running"
    await db.commit()

    try:
        _window_days = {6: 182, 12: 365, 18: 548, 24: 730}.get(window_months, 365)
        results = await asyncio.to_thread(
            _run_pipeline_sync, pair_id, ticker1, ticker2, _window_days
        )
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        await db.commit()
        return

    # Persist — upsert pattern: delete old rows then insert fresh ones
    await db.execute(
        delete(CointegrationResult).where(CointegrationResult.pair_id == pair_id)
    )
    coint = results["cointegration"]
    db.add(CointegrationResult(
        pair_id=pair_id,
        hedge_ratio=float(coint["hedge_ratio"]),
        adf_statistic=float(coint["adf_statistic"]),
        adf_pvalue=float(coint["adf_pvalue"]),
        cointegrated=bool(coint["cointegrated"]),
    ))

    report = results["validation"]
    await db.execute(
        delete(ValidationResult).where(ValidationResult.pair_id == pair_id)
    )
    acf_r2: Optional[float] = None
    acf_theta: Optional[float] = None
    if hasattr(report.autocorrelation, "r_squared"):
        acf_r2 = float(report.autocorrelation.r_squared)
    if hasattr(report.autocorrelation, "theta"):
        acf_theta = float(report.autocorrelation.theta)

    db.add(ValidationResult(
        pair_id=pair_id,
        stationarity_passed=report.stationarity.passed,
        drift_passed=report.linear_drift.passed,
        volatility_passed=report.constant_volatility.passed,
        acf_passed=report.autocorrelation.passed,
        normality_passed=report.normality.passed,
        confidence_level=results["confidence_level"],
        tests_passed_count=report.count_passing_tests(),
        acf_r_squared=acf_r2,
        acf_theta=acf_theta,
    ))

    lstm = results["lstm"]
    await db.execute(
        delete(EstimationResult).where(EstimationResult.pair_id == pair_id)
    )
    db.add(EstimationResult(
        pair_id=pair_id,
        theta=float(lstm.theta),
        mu=float(lstm.mu),
        sigma=float(lstm.sigma),
        theta_ci_lower=float(lstm.theta_ci[0]),
        theta_ci_upper=float(lstm.theta_ci[1]),
        mu_ci_lower=float(lstm.mu_ci[0]),
        mu_ci_upper=float(lstm.mu_ci[1]),
        sigma_ci_lower=float(lstm.sigma_ci[0]),
        sigma_ci_upper=float(lstm.sigma_ci[1]),
        theta_std=float(lstm.theta_std),
        sigma_std=float(lstm.sigma_std),
        n_mc_samples=int(lstm.n_mc_samples),
        model_version="v2_robust",
    ))

    mle = results["mle"]
    await db.execute(
        delete(MLEResult).where(MLEResult.pair_id == pair_id)
    )
    db.add(MLEResult(
        pair_id=pair_id,
        theta=float(mle.theta),
        mu=float(mle.mu),
        sigma=float(mle.sigma),
        log_likelihood=float(mle.log_likelihood) if mle.success else None,
        success=bool(mle.success),
    ))

    # Signals — bulk replace
    await db.execute(delete(Signal).where(Signal.pair_id == pair_id))
    for rec in results["signals"]:
        db.add(Signal(pair_id=pair_id, **rec))

    # Narration — generate via Claude Haiku, cache on the pair row
    latest_signal = results["signals"][-1] if results["signals"] else None
    narration_text = await asyncio.to_thread(
        narration_service.generate_narration,
        ticker1,
        ticker2,
        results["confidence_level"],
        results["validation"].count_passing_tests(),
        float(lstm.theta),
        float(lstm.mu),
        float(lstm.sigma),
        (float(lstm.theta_ci[0]), float(lstm.theta_ci[1])),
        (float(lstm.sigma_ci[0]), float(lstm.sigma_ci[1])),
        float(latest_signal["z_score"]) if latest_signal else None,
        latest_signal["signal_type"] if latest_signal else None,
        settings.ANTHROPIC_API_KEY,
    )
    pair_result = await db.execute(select(Pair).where(Pair.id == pair_id))
    pair_row = pair_result.scalar_one_or_none()
    if pair_row is not None:
        pair_row.narration = narration_text

    job.status = "complete"
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()


# ── Watchlist queries ──────────────────────────────────────────────────────────

async def get_user_watchlist(db: AsyncSession, user_id: uuid.UUID):
    """Returns list of (UserPair, Pair, latest Signal | None)."""
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(UserPair)
        .where(UserPair.user_id == user_id)
        .options(
            selectinload(UserPair.pair).selectinload(Pair.validation_result),
            selectinload(UserPair.pair).selectinload(Pair.signals),
        )
    )
    user_pairs = result.scalars().all()
    return user_pairs


async def get_pair_detail(db: AsyncSession, pair_id: uuid.UUID):
    """Load pair with all analysis relations eager-loaded."""
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Pair)
        .where(Pair.id == pair_id)
        .options(
            selectinload(Pair.cointegration_result),
            selectinload(Pair.validation_result),
            selectinload(Pair.estimation_result),
            selectinload(Pair.mle_result),
            selectinload(Pair.signals),
        )
    )
    return result.scalar_one_or_none()
