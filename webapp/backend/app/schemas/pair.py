import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, field_validator


# ── Request ────────────────────────────────────────────────────────────────────

class PairSubmitRequest(BaseModel):
    ticker1: str
    ticker2: str
    window_months: int = 12  # 6 / 12 / 18 / 24

    @field_validator("ticker1", "ticker2")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("window_months")
    @classmethod
    def validate_window(cls, v: int) -> int:
        if v not in (6, 12, 18, 24):
            raise ValueError("window_months must be 6, 12, 18, or 24")
        return v


# ── Nested response shapes ─────────────────────────────────────────────────────

class CointegrationResponse(BaseModel):
    hedge_ratio: float
    adf_statistic: float
    adf_pvalue: float
    cointegrated: bool
    computed_at: datetime

    model_config = {"from_attributes": True}


class ValidationResponse(BaseModel):
    stationarity_passed: bool
    drift_passed: bool
    volatility_passed: bool
    acf_passed: bool
    normality_passed: bool
    confidence_level: str           # HIGH / MEDIUM / LOW / NOT_OU
    tests_passed_count: int
    acf_r_squared: Optional[float]
    acf_theta: Optional[float]
    computed_at: datetime

    model_config = {"from_attributes": True}


class EstimationResponse(BaseModel):
    theta: float
    mu: float
    sigma: float
    theta_ci_lower: float
    theta_ci_upper: float
    mu_ci_lower: float
    mu_ci_upper: float
    sigma_ci_lower: float
    sigma_ci_upper: float
    theta_std: float
    sigma_std: float
    n_mc_samples: int
    model_version: str
    computed_at: datetime

    model_config = {"from_attributes": True}


class MLEResponse(BaseModel):
    theta: float
    mu: float
    sigma: float
    log_likelihood: Optional[float]
    success: bool
    computed_at: datetime

    model_config = {"from_attributes": True}


class SignalPoint(BaseModel):
    date: date
    z_score: float
    signal_type: str
    spread_value: float
    stationary_mean: float
    stationary_std: float

    model_config = {"from_attributes": True}


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str               # pending / running / complete / failed
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    window_months: int = 12

    model_config = {"from_attributes": True}


# ── Top-level pair responses ───────────────────────────────────────────────────

class PairSubmitResponse(BaseModel):
    pair_id: uuid.UUID
    job_id: uuid.UUID
    status: str               # pending (job just created) or cached (already analysed)


class UserPairSettings(BaseModel):
    alert_enabled: bool
    long_threshold: float
    short_threshold: float

    model_config = {"from_attributes": True}


class PairSummaryResponse(BaseModel):
    """Compact list-view — watchlist row."""
    pair_id: uuid.UUID
    ticker1: str
    ticker2: str
    sector: Optional[str]
    confidence_level: Optional[str]
    latest_z_score: Optional[float]
    latest_signal_type: Optional[str]
    latest_signal_date: Optional[date]
    alert_enabled: bool

    model_config = {"from_attributes": True}


class PairDetailResponse(BaseModel):
    """Full analysis — pair detail page."""
    pair_id: uuid.UUID
    ticker1: str
    ticker2: str
    sector: Optional[str]
    narration: Optional[str]
    window_months: Optional[int]
    settings: UserPairSettings
    cointegration: Optional[CointegrationResponse]
    validation: Optional[ValidationResponse]
    estimation: Optional[EstimationResponse]
    mle: Optional[MLEResponse]
    signals: list[SignalPoint]


class PairSettingsUpdateRequest(BaseModel):
    alert_enabled: Optional[bool] = None
    long_threshold: Optional[float] = None
    short_threshold: Optional[float] = None
