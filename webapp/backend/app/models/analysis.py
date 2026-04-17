import uuid
from datetime import datetime

from sqlalchemy import String, Boolean, Float, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


class CointegrationResult(Base):
    __tablename__ = "cointegration_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False, unique=True)
    hedge_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    adf_statistic: Mapped[float] = mapped_column(Float, nullable=False)
    adf_pvalue: Mapped[float] = mapped_column(Float, nullable=False)
    cointegrated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pair: Mapped["Pair"] = relationship(back_populates="cointegration_result")


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False, unique=True)
    stationarity_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drift_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    volatility_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    acf_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    normality_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(10), nullable=False)  # HIGH/MEDIUM/LOW/NOT_OU
    tests_passed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    acf_r_squared: Mapped[float | None] = mapped_column(Float, nullable=True)
    acf_theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pair: Mapped["Pair"] = relationship(back_populates="validation_result")


class EstimationResult(Base):
    """LSTM-robust estimation output."""
    __tablename__ = "estimation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False, unique=True)
    theta: Mapped[float] = mapped_column(Float, nullable=False)
    mu: Mapped[float] = mapped_column(Float, nullable=False)
    sigma: Mapped[float] = mapped_column(Float, nullable=False)
    theta_ci_lower: Mapped[float] = mapped_column(Float, nullable=False)
    theta_ci_upper: Mapped[float] = mapped_column(Float, nullable=False)
    mu_ci_lower: Mapped[float] = mapped_column(Float, nullable=False)
    mu_ci_upper: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_ci_lower: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_ci_upper: Mapped[float] = mapped_column(Float, nullable=False)
    theta_std: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_std: Mapped[float] = mapped_column(Float, nullable=False)
    n_mc_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), default="v2_robust", nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pair: Mapped["Pair"] = relationship(back_populates="estimation_result")


class MLEResult(Base):
    """Gaussian MLE — stored silently for the comparison panel."""
    __tablename__ = "mle_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False, unique=True)
    theta: Mapped[float] = mapped_column(Float, nullable=False)
    mu: Mapped[float] = mapped_column(Float, nullable=False)
    sigma: Mapped[float] = mapped_column(Float, nullable=False)
    log_likelihood: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pair: Mapped["Pair"] = relationship(back_populates="mle_result")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # pending/running/complete/failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    pair: Mapped["Pair"] = relationship(back_populates="analysis_jobs")
