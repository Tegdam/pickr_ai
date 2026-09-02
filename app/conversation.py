# Chat history: persists turns to RDS MySQL, condenses follow-up questions
# against prior turns, and orchestrates one conversational query end-to-end.
#
# CoordinatorAgent and the specialized agents in agents.py stay completely
# unaware history exists -- they only ever see a self-contained query string.
# Condensation and persistence both fail open: a DB or LLM hiccup here
# degrades a turn to "no history" rather than failing the request, since
# losing chat history isn't a safety issue the way a guardrail miss would be.

import logging
import os
from pathlib import Path

from langsmith import traceable
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from .agents import CoordinatorAgent
from .models import UserQuery
from .openai_client import client

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# RDS enforces TLS; this is AWS's public global CA bundle (not a secret --
# safe to commit), matching `mysql --ssl-mode=VERIFY_IDENTITY --ssl-ca=...`.
DB_SSL_CA = os.getenv(
    "DB_SSL_CA", str(Path(__file__).resolve().parent.parent / "certs" / "rds-global-bundle.pem")
)

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# pool_pre_ping: RDS can drop idle connections; this checks a connection is
# still alive before handing it out rather than failing on a stale one.
# ssl_verify_cert + ssl_verify_identity: verifies both the server's certificate
# chain and that the hostname matches -- equivalent to the RDS console's
# `--ssl-mode=VERIFY_IDENTITY`, not just "encrypted but unauthenticated" TLS.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={
        "ssl_ca": DB_SSL_CA,
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
    },
)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# Last 3 exchanges (3 user + 3 assistant rows) fed into condensation -- bounds
# prompt/token growth on long conversations while keeping enough recency for
# follow-up resolution.
HISTORY_WINDOW = 6

CONDENSE_SYSTEM_PROMPT = """Given a conversation history and a new follow-up message from the customer, rewrite the follow-up as a standalone question that makes full sense without needing the prior context. Preserve the customer's intent exactly -- don't answer the question, just rewrite it. If the follow-up is already standalone, return it unchanged. Respond with ONLY the rewritten question, no other text."""


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


def init_db():
    """Create chat_turns if it doesn't exist. Called once at app startup;
    failure is logged and swallowed so a DB outage doesn't prevent the app
    itself from starting -- load_history/save_exchange still fail open
    per-request regardless of whether this succeeded."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        logger.warning(
            "could not initialize chat_turns table; chat history will be "
            "unavailable until the database is reachable",
            exc_info=True,
        )


def load_history(conversation_id: str) -> list:
    """Return up to the last HISTORY_WINDOW (role, content) turns for a
    conversation, oldest first. Ordered by id (insertion order), not
    created_at -- a user/assistant pair saved within the same second would
    otherwise tie on MySQL's default (1-second-resolution) DATETIME.
    """
    try:
        with SessionLocal() as session:
            rows = (
                session.query(ChatTurn)
                .filter(ChatTurn.conversation_id == conversation_id)
                .order_by(ChatTurn.id.desc())
                .limit(HISTORY_WINDOW)
                .all()
            )
        return [(row.role, row.content) for row in reversed(rows)]
    except Exception:
        logger.warning("failed to load chat history; continuing without it", exc_info=True)
        return []


def list_sessions(limit: int = 50) -> list:
    """Return the most recently active conversations, each with a preview
    (its first user message) for the sessions sidebar. Unlike load_history,
    this is not scoped to one conversation_id -- it's a cross-conversation
    summary, so it's a separate query rather than reusing load_history.
    """
    try:
        with SessionLocal() as session:
            activity = (
                session.query(ChatTurn.conversation_id, func.max(ChatTurn.created_at).label("last_activity"))
                .group_by(ChatTurn.conversation_id)
                .order_by(func.max(ChatTurn.created_at).desc())
                .limit(limit)
                .all()
            )
            if not activity:
                return []

            conversation_ids = [row.conversation_id for row in activity]
            first_user_ids = (
                session.query(func.min(ChatTurn.id))
                .filter(ChatTurn.role == "user", ChatTurn.conversation_id.in_(conversation_ids))
                .group_by(ChatTurn.conversation_id)
            )
            previews = {
                row.conversation_id: row.content
                for row in session.query(ChatTurn).filter(ChatTurn.id.in_(first_user_ids))
            }

            return [
                {
                    "conversation_id": row.conversation_id,
                    "preview": previews.get(row.conversation_id, ""),
                    "last_activity": row.last_activity.isoformat() if row.last_activity else None,
                }
                for row in activity
            ]
    except Exception:
        logger.warning("failed to list chat sessions", exc_info=True)
        return []


def load_full_history(conversation_id: str) -> list:
    """Return every turn for a conversation, oldest first -- for display in
    the sessions sidebar (unlike load_history, not window-limited, since this
    is for showing a full past conversation rather than feeding condensation).
    Includes created_at so the UI can show the conversation's start time in
    its receipt header.
    """
    try:
        with SessionLocal() as session:
            rows = (
                session.query(ChatTurn)
                .filter(ChatTurn.conversation_id == conversation_id)
                .order_by(ChatTurn.id.asc())
                .all()
            )
        return [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    except Exception:
        logger.warning("failed to load full session history", exc_info=True)
        return []


def save_exchange(conversation_id: str, user_message: str, assistant_message: str) -> None:
    """Persist one user+assistant pair in a single commit."""
    try:
        with SessionLocal() as session:
            session.add(ChatTurn(conversation_id=conversation_id, role="user", content=user_message))
            session.add(ChatTurn(conversation_id=conversation_id, role="assistant", content=assistant_message))
            session.commit()
    except Exception:
        logger.warning("failed to save chat turn; continuing", exc_info=True)


def condense_query(history: list, raw_query: str) -> str:
    """Rewrite a follow-up into a standalone query using prior turns, e.g.
    "what about something cheaper" -> "what's a cheaper alternative to the
    Alpha Laptop". Falls back to the raw query unchanged on any error or when
    there's no history yet -- functionally identical to a fresh conversation.
    """
    if not history:
        return raw_query

    transcript = "\n".join(
        f"{'Customer' if role == 'user' else 'Assistant'}: {content}"
        for role, content in history
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nFollow-up message: {raw_query}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.warning("query condensation failed; using raw query as-is", exc_info=True)
        return raw_query


coordinator = CoordinatorAgent()


# @traceable groups every OpenAI call made during one query -- condensation,
# guardrails, generation -- under a single LangSmith trace for this function
# call, instead of each showing up as its own disconnected trace. A no-op
# when tracing is disabled (LANGSMITH_TRACING unset), same as wrap_openai in
# openai_client.py -- no LangSmith account needed for the app to work.
@traceable(name="handle_conversational_query")
def handle_conversational_query(conversation_id: str, raw_query: str) -> dict:
    """Orchestrates one turn: load history, condense the raw query against it
    if there is any, route the resolved query through CoordinatorAgent
    unchanged, then persist the exchange. CoordinatorAgent and the
    specialized agents never see history -- only a self-contained query.
    """
    history = load_history(conversation_id)
    resolved_query = condense_query(history, raw_query)

    result = coordinator.handle_query(UserQuery(query=resolved_query, raw_query=raw_query))

    save_exchange(conversation_id, raw_query, result["response"])

    return result
