from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.legal_categories import LegalCategoryResponse
from app.services.categories import get_category_by_id, list_categories
from app.core.errors import not_found

router = APIRouter()


@router.get("", response_model=list[LegalCategoryResponse])
def categories(db: Session = Depends(get_db)) -> list[LegalCategoryResponse]:
    return [
        LegalCategoryResponse(
            id=item.id,
            name=item.name,
            law_type=item.law_type,
            ai_law_types=item.ai_law_types,
            description=item.description,
            icon=item.icon_key,
            is_active=item.is_active,
            sort_order=item.sort_order,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in list_categories(db)
    ]


@router.get("/{category_id}", response_model=LegalCategoryResponse)
def category(category_id: str, db: Session = Depends(get_db)) -> LegalCategoryResponse:
    item = get_category_by_id(db, category_id)
    if item is None:
        raise not_found("Legal category not found.", "category_not_found")
    return LegalCategoryResponse(
        id=item.id,
        name=item.name,
        law_type=item.law_type,
        ai_law_types=item.ai_law_types,
        description=item.description,
        icon=item.icon_key,
        is_active=item.is_active,
        sort_order=item.sort_order,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )
