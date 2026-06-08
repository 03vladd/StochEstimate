"""add window_months to analysis_jobs

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-11
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("window_months", sa.Integer(), nullable=False, server_default="12"),
    )
    # Existing rows were all run with the old 730-day (24-month) hardcoded window.
    op.execute("UPDATE analysis_jobs SET window_months = 24")


def downgrade() -> None:
    op.drop_column("analysis_jobs", "window_months")