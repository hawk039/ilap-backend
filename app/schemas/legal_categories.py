from datetime import datetime

from app.schemas.common import APIModel


class LegalCategoryResponse(APIModel):
    id: str
    name: str
    law_type: str
    ai_law_types: list[str]
    description: str
    icon: str
    is_active: bool
    sort_order: int
    created_at: datetime
    updated_at: datetime | None = None
