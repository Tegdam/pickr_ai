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


@router.get("/sessions")
async def list_sessions():
    return conversation.list_sessions()


@router.get("/sessions/{conversation_id}")
async def get_session(conversation_id: str):
    return conversation.load_full_history(conversation_id)
