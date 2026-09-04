# HTTP layer: thin routing over app/conversation.py, which owns the actual
# orchestration. Handlers here only translate between HTTP and that module.

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
    """Answer one customer query within a conversation."""
    try:
        return conversation.handle_conversational_query(input.conversation_id, input.query)
    except Exception:
        logger.exception("unhandled error in /api/query")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)


# The two reads below need no try/except: their conversation.py counterparts
# already fail open, returning an empty list on a database error, since a
# missing sidebar is a degraded view rather than a failed request.


@router.get("/sessions")
async def list_sessions():
    """Recent conversations, for the sessions sidebar."""
    return conversation.list_sessions()


@router.get("/sessions/{conversation_id}")
async def get_session(conversation_id: str):
    """Every turn of one conversation, for replaying it in the UI."""
    return conversation.load_full_history(conversation_id)


@router.delete("/sessions/{conversation_id}")
async def delete_session(conversation_id: str):
    """Delete one conversation's history, returning the row count removed.

    Unlike the reads above, delete_conversation deliberately does not fail
    open -- so a failure must surface as a 500 here rather than be reported
    to the customer as a deletion that silently didn't happen.
    """
    try:
        deleted = conversation.delete_conversation(conversation_id)
        return {"deleted": deleted}
    except Exception:
        logger.exception("unhandled error in DELETE /api/sessions/%s", conversation_id)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)
