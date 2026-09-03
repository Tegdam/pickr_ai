import logging

from fastapi import APIRouter, HTTPException
from . import conversation
from .models import ChatQuery

logger = logging.getLogger(__name__)

router = APIRouter()

# Generic message rather than str(e): the raw exception can carry internal
# details (stack traces, DB connection strings, file paths) that shouldn't
# reach the client. The real exception is still logged server-side.
GENERIC_ERROR_MESSAGE = "Something went wrong processing your request. Please try again."


@router.post("/query")
async def handle_query(input: ChatQuery):
    try:
        return conversation.handle_conversational_query(input.conversation_id, input.query)
    except Exception:
        logger.exception("unhandled error in /api/query")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)


@router.get("/sessions")
async def list_sessions():
    return conversation.list_sessions()


@router.get("/sessions/{conversation_id}")
async def get_session(conversation_id: str):
    return conversation.load_full_history(conversation_id)


@router.delete("/sessions/{conversation_id}")
async def delete_session(conversation_id: str):
    try:
        deleted = conversation.delete_conversation(conversation_id)
        return {"deleted": deleted}
    except Exception:
        logger.exception("unhandled error in DELETE /api/sessions/%s", conversation_id)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)
