from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import api
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_query_returns_agent_response(client, monkeypatch):
    monkeypatch.setattr(
        api.conversation, "handle_conversational_query",
        MagicMock(return_value={"response": "mocked answer"}),
    )
    res = client.post("/api/query", json={"query": "what is your return policy?", "conversation_id": "abc-123"})
    assert res.status_code == 200
    assert res.json() == {"response": "mocked answer"}


def test_query_missing_field_returns_422(client):
    res = client.post("/api/query", json={})
    assert res.status_code == 422


def test_query_missing_conversation_id_returns_422(client):
    res = client.post("/api/query", json={"query": "anything"})
    assert res.status_code == 422


def test_query_agent_exception_returns_500(client, monkeypatch):
    monkeypatch.setattr(
        api.conversation, "handle_conversational_query",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    res = client.post("/api/query", json={"query": "anything", "conversation_id": "abc-123"})
    assert res.status_code == 500
    # Generic message, not the raw exception text -- see api.GENERIC_ERROR_MESSAGE.
    assert res.json() == {"detail": api.GENERIC_ERROR_MESSAGE}
    assert "boom" not in res.text


def test_root_serves_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
