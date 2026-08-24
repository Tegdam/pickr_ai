from fastapi import APIRouter, HTTPException
from .agents import CoordinatorAgent
from .models import UserQuery

router = APIRouter()
agent = CoordinatorAgent()

@router.post("/query")
async def handle_query(input: UserQuery):
    try:
        return agent.handle_query(input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))