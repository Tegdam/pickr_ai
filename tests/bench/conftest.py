"""Fixtures shared by bench tests. Nothing here touches the network."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import agents
from app.guardrails import INPUT_CLASSIFIER_SYSTEM_PROMPT, OUTPUT_CLASSIFIER_SYSTEM_PROMPT
from app.conversation import CONDENSE_SYSTEM_PROMPT
from app.agents import _INTENT_CLASSIFIER_SYSTEM_PROMPT
from app.models import Product, Review, StorePolicy


PRODUCTS = [
    Product(id="P1", name="Alpha Laptop", brand="Acme", category="laptop",
            price=500.0, description="Budget laptop.", stock=10, rating=4.5),
    Product(id="P2", name="Beta Laptop", brand="Acme", category="laptop",
            price=1200.0, description="Premium laptop.", stock=5, rating=4.8),
    Product(id="P3", name="Gamma Phone", brand="Zenith", category="smartphone",
            price=300.0, description="Entry phone.", stock=4, rating=4.0),
    Product(id="P4", name="Delta TV", brand="Zenith", category="smart_tv",
            price=900.0, description="4K TV.", stock=3, rating=3.9),
    Product(id="P5", name="Echo Speaker", brand="Acme", category="speaker",
            price=80.0, description="Bluetooth speaker.", stock=7, rating=4.1),
]
REVIEWS = [
    Review(product_id="P1", rating=5.0, text="Great value.", date="01-01-2025"),
    Review(product_id="P1", rating=4.0, text="Battery could be better.", date="02-01-2025"),
    Review(product_id="P3", rating=3.0, text="Fine for the price.", date="03-01-2025"),
]
POLICIES = [
    StorePolicy(policy_type="returns", description="Laptop Return Policy",
                conditions="Unopened.", timeframe="14"),
    StorePolicy(policy_type="warranty", description="Standard Warranty",
                conditions="Defects only.", timeframe="365"),
]


def _response(text, prompt_tokens=100, completion_tokens=20, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletions:
    """Stands in for client.chat.completions. Answers by system prompt so the
    app's control flow (guardrails pass, classifier picks a category) works."""

    def __init__(self):
        self.calls = []
        self.classifier_category = "recommendation"

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if system == INPUT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"is_injection": false, "is_off_topic": false}', 60, 12)
        if system == OUTPUT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"is_hallucination": false}', 300, 8)
        if system == _INTENT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"category": "%s"}' % self.classifier_category, 250, 6)
        if system == CONDENSE_SYSTEM_PROMPT:
            follow_up = kwargs["messages"][1]["content"].rsplit("Follow-up message: ", 1)[1]
            return _response("standalone: " + follow_up, 180, 15)
        return _response("agent answer about products", 400, 90)


@pytest.fixture
def fake_chat(monkeypatch):
    """Swap client.chat for a fake whose .completions.create records calls.
    Same patch point the app's own tests use (tests/test_agents.py::mock_openai)."""
    completions = FakeCompletions()
    fake = SimpleNamespace(completions=completions)
    monkeypatch.setattr(agents.client, "chat", fake)
    # moderation is a separate endpoint; never flag anything in tests
    fake_mod = MagicMock()
    fake_mod.create.return_value = SimpleNamespace(results=[SimpleNamespace(flagged=False)])
    monkeypatch.setattr(agents.client, "moderations", fake_mod)
    # embeddings is a separate endpoint too; stub it so an FAQAgent fallback
    # can never reach the network in bench tests
    fake_emb = MagicMock()
    fake_emb.create.side_effect = lambda model, input: SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.25, 0.25, 0.25, 0.25]) for _ in input])
    monkeypatch.setattr(agents.client, "embeddings", fake_emb)
    return completions


@pytest.fixture
def small_catalog(monkeypatch):
    monkeypatch.setattr(agents, "load_products", lambda: list(PRODUCTS))
    monkeypatch.setattr(agents, "load_reviews", lambda: list(REVIEWS))
    monkeypatch.setattr(agents, "load_store_policies", lambda: list(POLICIES))
    return SimpleNamespace(products=PRODUCTS, reviews=REVIEWS, policies=POLICIES)
