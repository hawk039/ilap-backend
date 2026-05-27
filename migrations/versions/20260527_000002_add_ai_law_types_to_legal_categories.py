"""add ai law type mappings to legal categories"""

from alembic import op
import sqlalchemy as sa

revision = "20260527_000002"
down_revision = "20260505_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "legal_categories",
        sa.Column("ai_law_types", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("legal_categories", "ai_law_types")
