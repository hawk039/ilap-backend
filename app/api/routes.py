from fastapi import APIRouter, HTTPException
from schemas.request import AskRequest
from schemas.response import AskResponse
from app.services.answer_service import get_answer
import traceback

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_law(request: AskRequest):
    try:
        return get_answer(request.query)
    except Exception as e:
        print(f"ERROR processing request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
