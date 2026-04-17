"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("plan", sa.String(20), nullable=False, server_default="premium"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "email_verifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token", sa.String(128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )
    op.create_index("ix_email_verifications_token", "email_verifications", ["token"])

    op.create_table(
        "pairs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ticker1", sa.String(20), nullable=False),
        sa.Column("ticker2", sa.String(20), nullable=False),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker1", "ticker2", name="uq_pairs_tickers"),
    )

    op.create_table(
        "user_pairs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("alert_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("long_threshold", sa.Float(), nullable=False, server_default="1.5"),
        sa.Column("short_threshold", sa.Float(), nullable=False, server_default="1.5"),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "pair_id", name="uq_user_pairs"),
    )

    op.create_table(
        "cointegration_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("hedge_ratio", sa.Float(), nullable=False),
        sa.Column("adf_statistic", sa.Float(), nullable=False),
        sa.Column("adf_pvalue", sa.Float(), nullable=False),
        sa.Column("cointegrated", sa.Boolean(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pair_id"),
    )

    op.create_table(
        "validation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stationarity_passed", sa.Boolean(), nullable=False),
        sa.Column("drift_passed", sa.Boolean(), nullable=False),
        sa.Column("volatility_passed", sa.Boolean(), nullable=False),
        sa.Column("acf_passed", sa.Boolean(), nullable=False),
        sa.Column("normality_passed", sa.Boolean(), nullable=False),
        sa.Column("confidence_level", sa.String(10), nullable=False),
        sa.Column("tests_passed_count", sa.Integer(), nullable=False),
        sa.Column("acf_r_squared", sa.Float(), nullable=True),
        sa.Column("acf_theta", sa.Float(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pair_id"),
    )

    op.create_table(
        "estimation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("mu", sa.Float(), nullable=False),
        sa.Column("sigma", sa.Float(), nullable=False),
        sa.Column("theta_ci_lower", sa.Float(), nullable=False),
        sa.Column("theta_ci_upper", sa.Float(), nullable=False),
        sa.Column("mu_ci_lower", sa.Float(), nullable=False),
        sa.Column("mu_ci_upper", sa.Float(), nullable=False),
        sa.Column("sigma_ci_lower", sa.Float(), nullable=False),
        sa.Column("sigma_ci_upper", sa.Float(), nullable=False),
        sa.Column("theta_std", sa.Float(), nullable=False),
        sa.Column("sigma_std", sa.Float(), nullable=False),
        sa.Column("n_mc_samples", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False, server_default="v2_robust"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pair_id"),
    )

    op.create_table(
        "mle_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("mu", sa.Float(), nullable=False),
        sa.Column("sigma", sa.Float(), nullable=False),
        sa.Column("log_likelihood", sa.Float(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pair_id"),
    )

    op.create_table(
        "analysis_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_jobs_pair_status", "analysis_jobs", ["pair_id", "status"])

    op.create_table(
        "signals",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("z_score", sa.Float(), nullable=False),
        sa.Column("signal_type", sa.String(10), nullable=False),
        sa.Column("spread_value", sa.Float(), nullable=False),
        sa.Column("stationary_mean", sa.Float(), nullable=False),
        sa.Column("stationary_std", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pair_id", "date", name="uq_signals_pair_date"),
    )

    op.create_table(
        "alert_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("signal_type", sa.String(10), nullable=False),
        sa.Column("z_score", sa.Float(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("alert_log")
    op.drop_table("signals")
    op.drop_table("analysis_jobs")
    op.drop_table("mle_results")
    op.drop_table("estimation_results")
    op.drop_table("validation_results")
    op.drop_table("cointegration_results")
    op.drop_table("user_pairs")
    op.drop_table("pairs")
    op.drop_table("email_verifications")
    op.drop_table("users")
