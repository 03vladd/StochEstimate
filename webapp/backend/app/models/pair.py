import uuid
from datetime import datetime

from sqlalchemy import String, Boolean, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


class Pair(Base):
    """Global pair entity — shared across all users."""
    __tablename__ = "pairs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker1: Mapped[str] = mapped_column(String(20), nullable=False)
    ticker2: Mapped[str] = mapped_column(String(20), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (UniqueConstraint("ticker1", "ticker2", name="uq_pairs_tickers"),)

    watchers: Mapped[list["UserPair"]] = relationship(back_populates="pair", cascade="all, delete-orphan")
    cointegration_result: Mapped["CointegrationResult | None"] = relationship(back_populates="pair", uselist=False)
    validation_result: Mapped["ValidationResult | None"] = relationship(back_populates="pair", uselist=False)
    estimation_result: Mapped["EstimationResult | None"] = relationship(back_populates="pair", uselist=False)
    mle_result: Mapped["MLEResult | None"] = relationship(back_populates="pair", uselist=False)
    analysis_jobs: Mapped[list["AnalysisJob"]] = relationship(back_populates="pair", cascade="all, delete-orphan")
    signals: Mapped[list["Signal"]] = relationship(back_populates="pair", cascade="all, delete-orphan")


class UserPair(Base):
    """Watchlist junction — one row per (user, pair)."""
    __tablename__ = "user_pairs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    alert_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    long_threshold: Mapped[float] = mapped_column(Float, default=1.5, nullable=False)
    short_threshold: Mapped[float] = mapped_column(Float, default=1.5, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "pair_id", name="uq_user_pairs"),)

    user: Mapped["User"] = relationship(back_populates="watched_pairs")
    pair: Mapped["Pair"] = relationship(back_populates="watchers")
