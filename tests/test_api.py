from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import api
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_query_returns_agent_response(client, monkeypatch):
    monkeypatch.setattr(api.agent, "handle_query", MagicMock(return_value={"response": "mocked answer"}))
    res = client.post("/api/query", json={"query": "what is your return policy?"})
    assert res.status_code == 200
    assert res.json() == {"response": "mocked answer"}


def test_query_missing_field_returns_422(client):
    res = client.post("/api/query", json={})
    assert res.status_code == 422


def test_query_agent_exception_returns_500(client, monkeypatch):
    monkeypatch.setattr(api.agent, "handle_query", MagicMock(side_effect=RuntimeError("boom")))
    res = client.post("/api/query", json={"query": "anything"})
    assert res.status_code == 500
    assert res.json() == {"detail": "boom"}


def test_root_serves_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
