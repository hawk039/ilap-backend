from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LegalCategory


def list_categories(db: Session) -> list[LegalCategory]:
    return list(
        db.scalars(
            select(LegalCategory)
            .where(LegalCategory.is_active.is_(True))
            .order_by(LegalCategory.sort_order.asc(), LegalCategory.name.asc())
        ).all()
    )


def get_category_by_id(db: Session, category_id: str) -> LegalCategory | None:
    return db.get(LegalCategory, category_id)


def get_category_by_law_type(db: Session, law_type: str) -> LegalCategory | None:
    return db.scalar(
        select(LegalCategory).where(LegalCategory.law_type == law_type, LegalCategory.is_active.is_(True))
    )


def get_category_by_ai_law_type(db: Session, ai_law_type: str) -> LegalCategory | None:
    ai_law_type = ai_law_type.strip()
    if not ai_law_type:
        return None
    for category in list_categories(db):
        if ai_law_type in category.ai_law_types:
            return category
    return None
