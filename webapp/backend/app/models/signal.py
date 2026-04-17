import uuid
from datetime import date, datetime

from sqlalchemy import String, Float, Date, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG/SHORT/EXIT/NONE
    spread_value: Mapped[float] = mapped_column(Float, nullable=False)
    stationary_mean: Mapped[float] = mapped_column(Float, nullable=False)   # mu used for z-score
    stationary_std: Mapped[float] = mapped_column(Float, nullable=False)    # sqrt(sigma^2 / 2*theta)

    __table_args__ = (UniqueConstraint("pair_id", "date", name="uq_signals_pair_date"),)

    pair: Mapped["Pair"] = relationship(back_populates="signals")


class AlertLog(Base):
    """Record of every alert email sent to a user."""
    __tablename__ = "alert_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    pair_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pairs.id", ondelete="CASCADE"), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="alert_logs")
