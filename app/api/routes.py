from fastapi import APIRouter
from schemas.request import AskRequest
from schemas.response import AskResponse
from app.services.answer_service import get_answer

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_law(request: AskRequest):
    return get_answer(request.query)
