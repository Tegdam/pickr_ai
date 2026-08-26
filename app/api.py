from fastapi import APIRouter, HTTPException
from . import conversation
from .models import ChatQuery

router = APIRouter()

@router.post("/query")
async def handle_query(input: ChatQuery):
    try:
        return conversation.handle_conversational_query(input.conversation_id, input.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))