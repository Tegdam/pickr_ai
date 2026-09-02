from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import conversation


def _chat_response(text: str):
    message = MagicMock(content=text)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


@pytest.fixture
def mock_chat(monkeypatch):
    """Replace conversation.client.chat so condense_query makes no real call."""
    mock_create = MagicMock()
    fake_chat = MagicMock()
    fake_chat.completions.create = mock_create
    monkeypatch.setattr(conversation.client, "chat", fake_chat)
    return mock_create


@pytest.fixture
def sqlite_db(monkeypatch):
    """
    Point conversation.py's DB layer at an in-memory SQLite engine instead of
    real MySQL, so load_history/save_exchange are exercised against real SQL
    (schema, filtering, ordering) without needing RDS reachable to run tests.
    """
    test_engine = create_engine("sqlite:///:memory:")
    conversation.Base.metadata.create_all(bind=test_engine)
    monkeypatch.setattr(conversation, "SessionLocal", sessionmaker(bind=test_engine))
    return test_engine


class TestLoadAndSaveHistory:
    def test_save_then_load_round_trip(self, sqlite_db):
        conversation.save_exchange("conv-1", "hello", "hi there")
        assert conversation.load_history("conv-1") == [("user", "hello"), ("assistant", "hi there")]

    def test_scoped_to_conversation_id(self, sqlite_db):
        conversation.save_exchange("conv-1", "q1", "a1")
        conversation.save_exchange("conv-2", "q2", "a2")
        assert conversation.load_history("conv-1") == [("user", "q1"), ("assistant", "a1")]
        assert conversation.load_history("conv-2") == [("user", "q2"), ("assistant", "a2")]

    def test_empty_for_unknown_conversation(self, sqlite_db):
        assert conversation.load_history("nonexistent") == []

    def test_chronological_order_across_multiple_exchanges(self, sqlite_db):
        conversation.save_exchange("conv-1", "q1", "a1")
        conversation.save_exchange("conv-1", "q2", "a2")
        assert conversation.load_history("conv-1") == [
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
        ]

    def test_respects_history_window(self, sqlite_db, monkeypatch):
        monkeypatch.setattr(conversation, "HISTORY_WINDOW", 2)
        conversation.save_exchange("conv-1", "q1", "a1")
        conversation.save_exchange("conv-1", "q2", "a2")
        # window of 2 keeps only the most recent row pair, not the oldest
        assert conversation.load_history("conv-1") == [("user", "q2"), ("assistant", "a2")]

    def test_load_history_fails_open_on_db_error(self, monkeypatch):
        def raise_error():
            raise RuntimeError("db unreachable")
        monkeypatch.setattr(conversation, "SessionLocal", raise_error)
        assert conversation.load_history("conv-1") == []

    def test_save_exchange_fails_open_on_db_error(self, monkeypatch):
        def raise_error():
            raise RuntimeError("db unreachable")
        monkeypatch.setattr(conversation, "SessionLocal", raise_error)
        conversation.save_exchange("conv-1", "q", "a")  # must not raise


class TestCondenseQuery:
    def test_no_history_returns_raw_query_unchanged(self, mock_chat):
        result = conversation.condense_query([], "what about something cheaper")
        assert result == "what about something cheaper"
        mock_chat.assert_not_called()

    def test_rewrites_follow_up_using_history(self, mock_chat):
        mock_chat.return_value = _chat_response("what's a cheaper alternative to the Alpha Laptop")
        history = [("user", "recommend a laptop"), ("assistant", "I recommend the Alpha Laptop.")]

        result = conversation.condense_query(history, "what about something cheaper")

        assert result == "what's a cheaper alternative to the Alpha Laptop"
        mock_chat.assert_called_once()

    def test_error_falls_back_to_raw_query(self, mock_chat):
        mock_chat.side_effect = RuntimeError("boom")
        history = [("user", "q"), ("assistant", "a")]
        result = conversation.condense_query(history, "follow up")
        assert result == "follow up"


class TestHandleConversationalQuery:
    def test_orchestrates_load_condense_route_save(self, monkeypatch):
        monkeypatch.setattr(
            conversation, "load_history",
            MagicMock(return_value=[("user", "q1"), ("assistant", "a1")]),
        )
        monkeypatch.setattr(conversation, "condense_query", MagicMock(return_value="resolved query"))
        monkeypatch.setattr(conversation.coordinator, "handle_query", MagicMock(return_value={"response": "final answer"}))
        mock_save = MagicMock()
        monkeypatch.setattr(conversation, "save_exchange", mock_save)

        result = conversation.handle_conversational_query("conv-1", "raw follow up")

        assert result == {"response": "final answer"}
        conversation.condense_query.assert_called_once_with(
            [("user", "q1"), ("assistant", "a1")], "raw follow up",
        )
        conversation.coordinator.handle_query.assert_called_once()
        routed_query = conversation.coordinator.handle_query.call_args[0][0]
        assert routed_query.query == "resolved query"
        # Guardrail input-checking must see the literal customer text, not the
        # condensed version -- see CoordinatorAgent.handle_query's raw_query fallback.
        assert routed_query.raw_query == "raw follow up"
        mock_save.assert_called_once_with("conv-1", "raw follow up", "final answer")

    def test_first_message_has_no_history_to_condense(self, monkeypatch):
        monkeypatch.setattr(conversation, "load_history", MagicMock(return_value=[]))
        monkeypatch.setattr(conversation.coordinator, "handle_query", MagicMock(return_value={"response": "answer"}))
        monkeypatch.setattr(conversation, "save_exchange", MagicMock())

        conversation.handle_conversational_query("conv-new", "recommend a laptop")

        routed_query = conversation.coordinator.handle_query.call_args[0][0]
        assert routed_query.query == "recommend a laptop"  # condense_query is a no-op with no history
