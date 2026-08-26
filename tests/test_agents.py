from unittest.mock import MagicMock

import pytest

from app import agents
from app.models import Product, Review, StorePolicy, UserQuery


PRODUCTS = [
    Product(id="P1", name="Alpha Laptop", brand="Acme", category="laptop",
            price=500.0, description="Budget laptop.", stock=10, rating=4.5),
    Product(id="P2", name="Beta Laptop", brand="Acme", category="laptop",
            price=1200.0, description="Premium laptop.", stock=5, rating=4.8),
    Product(id="P3", name="Gamma Phone", brand="Zenith", category="smartphone",
            price=300.0, description="Entry phone.", stock=0, rating=4.0),
    Product(id="P4", name="Delta TV", brand="Zenith", category="smart_tv",
            price=900.0, description="4K TV.", stock=3, rating=3.9),
    Product(id="P5", name="Mystery Gadget", brand="Acme", category="gadget",
            price=None, description="No price yet.", stock=5, rating=2.0),
]

REVIEWS = [
    Review(product_id="P1", rating=5.0, text="Great value.", date="01-01-2025"),
    Review(product_id="P1", rating=4.0, text="Battery could be better.", date="02-01-2025"),
]

POLICIES = [
    StorePolicy(policy_type="return", description="30-day returns.",
                conditions="Unopened.", timeframe="30"),
    StorePolicy(policy_type="warranty", description="1-year warranty.",
                conditions="Defects only.", timeframe="365"),
    StorePolicy(policy_type="shipping", description="Free shipping over $50.",
                conditions="US only.", timeframe="0"),
    StorePolicy(policy_type="exchange", description="30-day exchange window.",
                conditions="Original packaging.", timeframe="30"),
]

# Hand-picked so a "return" query has an unambiguous nearest-neighbor ranking:
# return < warranty < shipping < exchange. Lets tests assert retrieval actually
# narrows to the top-k, not just that *something* came back.
FAKE_EMBEDDING_VECTORS = {
    "return": [1.0, 0.0, 0.0, 0.0],
    "warranty": [0.9, 0.1, 0.0, 0.0],
    "shipping": [0.5, 0.0, 0.5, 0.0],
    "exchange": [0.0, 0.0, 0.0, 1.0],
}
FAKE_EMBEDDING_FALLBACK = [0.25, 0.25, 0.25, 0.25]


def fake_embed_one(text: str):
    lowered = text.lower()
    for keyword, vector in FAKE_EMBEDDING_VECTORS.items():
        if keyword in lowered:
            return vector
    return FAKE_EMBEDDING_FALLBACK


@pytest.fixture
def patched_data(monkeypatch):
    """Swap the CSV-backed loaders for small, predictable in-memory fixtures."""
    monkeypatch.setattr(agents, "load_products", lambda: list(PRODUCTS))
    monkeypatch.setattr(agents, "load_reviews", lambda: list(REVIEWS))
    monkeypatch.setattr(agents, "load_store_policies", lambda: list(POLICIES))


@pytest.fixture
def mock_openai(monkeypatch):
    """Replace the module-level openai.chat proxy so no real API call is made."""
    fake_message = MagicMock(content="mocked LLM response")
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])
    mock_create = MagicMock(return_value=fake_response)
    fake_chat = MagicMock()
    fake_chat.completions.create = mock_create
    monkeypatch.setattr(agents.openai, "chat", fake_chat)
    return mock_create


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Replace the module-level openai.embeddings proxy with a deterministic stub."""
    def fake_create(model, input):
        data = [MagicMock(embedding=fake_embed_one(text)) for text in input]
        return MagicMock(data=data)

    fake_embeddings_obj = MagicMock()
    fake_embeddings_obj.create = MagicMock(side_effect=fake_create)
    monkeypatch.setattr(agents.openai, "embeddings", fake_embeddings_obj)
    return fake_embeddings_obj.create


@pytest.fixture
def in_memory_chroma(monkeypatch):
    """
    Redirect PolicyIndex's persistent Chroma client to an in-memory one.

    chromadb.Client() caches its underlying system across calls with the same
    (default) settings, so two tests in one process would otherwise share the
    "store_policies" collection. Drop it first so each test starts clean.
    """
    import chromadb as chromadb_module

    def make_client(path):
        client = chromadb_module.Client()
        try:
            client.delete_collection("store_policies")
        except Exception:
            pass  # collection didn't exist yet -- nothing to clean up
        return client

    monkeypatch.setattr(agents.chromadb, "PersistentClient", make_client)


def sent_messages(mock_create):
    """Concatenate the content of every message from the last OpenAI call."""
    messages = mock_create.call_args.kwargs["messages"]
    return "\n".join(m["content"] for m in messages)


# ---------------------------------------------------------------------------
# ProductRecommendationAgent
# ---------------------------------------------------------------------------

class TestProductRecommendationAgent:
    def test_match_category_handles_underscore_categories(self, patched_data):
        agent = agents.ProductRecommendationAgent()
        assert agent._match_category("looking for a smart tv") == "smart_tv"
        assert agent._match_category("looking for a laptop") == "laptop"
        assert agent._match_category("looking for shoes") is None

    def test_match_brand(self, patched_data):
        agent = agents.ProductRecommendationAgent()
        assert agent._match_brand("an acme laptop please") == "acme"
        assert agent._match_brand("anything decent") is None

    def test_match_price_ceiling(self, patched_data):
        agent = agents.ProductRecommendationAgent()
        assert agent._match_price_ceiling("something under $600") == 600.0
        assert agent._match_price_ceiling("less than 300.50 please") == 300.50
        assert agent._match_price_ceiling("no price mentioned") is None

    def test_filters_by_category_and_price(self, patched_data, mock_openai):
        agent = agents.ProductRecommendationAgent()
        result = agent.recommend_product(UserQuery(query="recommend a laptop under $600"))

        mock_openai.assert_called_once()
        sent = sent_messages(mock_openai)
        assert "Alpha Laptop" in sent
        assert "Beta Laptop" not in sent  # over the price ceiling
        assert result == {"response": "mocked LLM response"}

    def test_excludes_out_of_stock(self, patched_data, mock_openai):
        agent = agents.ProductRecommendationAgent()
        result = agent.recommend_product(UserQuery(query="recommend a smartphone"))

        mock_openai.assert_not_called()
        assert "couldn't find" in result["response"].lower()

    def test_no_constraints_falls_back_to_top_rated_in_stock(self, patched_data, mock_openai):
        agent = agents.ProductRecommendationAgent()
        agent.recommend_product(UserQuery(query="recommend something good"))

        mock_openai.assert_called_once()
        sent = sent_messages(mock_openai)
        assert "Gamma Phone" not in sent  # out of stock


# ---------------------------------------------------------------------------
# PriceComparisonAgent
# ---------------------------------------------------------------------------

class TestPriceComparisonAgent:
    def test_requires_at_least_two_products(self, patched_data):
        agent = agents.PriceComparisonAgent()
        result = agent.compare_products(UserQuery(query="price of Alpha Laptop"))
        assert "at least two" in result["response"].lower()

    def test_two_product_diff(self, patched_data):
        agent = agents.PriceComparisonAgent()
        query = UserQuery(query="price difference between Alpha Laptop and Beta Laptop")
        result = agent.compare_products(query)

        assert "Alpha Laptop is cheaper than Beta Laptop by $700.00 (58.3% less)." in result["response"]

    def test_more_than_two_products_reports_spread(self, patched_data):
        agent = agents.PriceComparisonAgent()
        query = UserQuery(query="compare price of Alpha Laptop, Beta Laptop, and Delta TV")
        result = agent.compare_products(query)

        assert "Cheapest: Alpha Laptop ($500.00)" in result["response"]
        assert "Most expensive: Beta Laptop ($1200.00)" in result["response"]
        assert "Spread: $700.00" in result["response"]

    def test_products_without_a_price_are_ignored(self, patched_data):
        agent = agents.PriceComparisonAgent()
        query = UserQuery(query="price of Alpha Laptop vs Mystery Gadget")
        result = agent.compare_products(query)
        assert "at least two" in result["response"].lower()


# ---------------------------------------------------------------------------
# PolicyIndex (RAG staleness detection)
# ---------------------------------------------------------------------------

class TestPolicyIndex:
    def test_rebuilds_only_when_content_changes(self, monkeypatch, mock_embeddings):
        """
        Own client setup rather than the in_memory_chroma fixture: that fixture
        wipes the collection on every PersistentClient() call to keep tests
        isolated from each other, which would also wipe it *between* the two
        PolicyIndex constructions this test needs to share one collection.
        """
        import chromadb as chromadb_module
        shared_client = chromadb_module.Client()
        monkeypatch.setattr(agents.chromadb, "PersistentClient", lambda path: shared_client)
        try:
            shared_client.delete_collection("store_policies")
        except Exception:
            pass  # nothing to clean up on a fresh client

        original = [StorePolicy(policy_type="return", description="30-day returns.",
                                 conditions="Unopened.", timeframe="30")]

        agents.PolicyIndex(original)
        assert mock_embeddings.call_count == 1  # embedded once on first build

        # Same content, reconstructed (simulates a process restart) -- reuse the index.
        agents.PolicyIndex(list(original))
        assert mock_embeddings.call_count == 1  # not re-embedded

        # Same row count, but the description changed in place.
        edited = [StorePolicy(policy_type="return", description="14-day returns now.",
                               conditions="Unopened.", timeframe="30")]
        index = agents.PolicyIndex(edited)
        assert mock_embeddings.call_count == 2  # edit was detected, index rebuilt

        results = index.search("return policy")
        assert "14-day returns now." in results[0]
        assert "30-day returns." not in results[0]


# ---------------------------------------------------------------------------
# StorePolicyAgent / FAQAgent
# ---------------------------------------------------------------------------

class TestStorePolicyAndFAQ:
    def test_keyword_match_skips_llm(self, patched_data, mock_openai):
        agent = agents.StorePolicyAgent()
        result = agent.get_policy_info(UserQuery(query="what is your return policy"))

        mock_openai.assert_not_called()
        assert "30-day returns." in result["response"]

    def test_no_match_returns_constant(self, patched_data):
        agent = agents.StorePolicyAgent()
        result = agent.get_policy_info(UserQuery(query="do you sell gift cards"))
        assert result["response"] == agents.StorePolicyAgent.NO_MATCH_MESSAGE

    def test_expired_warranty_phrasing_does_not_match_warranty_keyword(self, patched_data):
        """
        "no longer under warranty" should not be treated as a warranty question --
        the customer is asking about something else (e.g. repairs) and only
        mentions warranty to say it's lapsed. Regression test for a bug where this
        phrasing spuriously matched "warranty" and returned the wrong policy
        instead of falling through to FAQAgent's semantic search.
        """
        agent = agents.StorePolicyAgent()
        result = agent.get_policy_info(
            UserQuery(query="my laptop is no longer under warranty, can you fix it?")
        )
        assert result["response"] == agents.StorePolicyAgent.NO_MATCH_MESSAGE

    def test_faq_agent_retrieves_relevant_chunks_only(
        self, patched_data, mock_embeddings, in_memory_chroma, mock_openai
    ):
        agent = agents.FAQAgent()
        agent.get_policy_info(UserQuery(query="what is your return policy"))

        mock_openai.assert_called_once()
        sent = sent_messages(mock_openai)
        # top-3 nearest neighbors for "return": return, warranty, shipping.
        # "exchange" is the deliberate odd one out (RAG narrows, doesn't dump everything).
        assert "30-day returns." in sent
        assert "1-year warranty." in sent
        assert "Free shipping over $50." in sent
        assert "exchange window" not in sent

    def test_faq_agent_empty_policies_skips_llm(self, monkeypatch, mock_embeddings, in_memory_chroma, mock_openai):
        monkeypatch.setattr(agents, "load_store_policies", lambda: [])
        agent = agents.FAQAgent()
        result = agent.get_policy_info(UserQuery(query="anything at all"))

        mock_openai.assert_not_called()
        assert result["response"] == agents.FAQAgent.NO_MATCH_MESSAGE


# ---------------------------------------------------------------------------
# CoordinatorAgent routing
# ---------------------------------------------------------------------------

class TestCoordinatorRouting:
    def test_review_routes_to_review_agent(self, patched_data, mock_openai):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="reviews for Alpha Laptop"))

        mock_openai.assert_called_once()
        assert result == {"response": "mocked LLM response"}

    def test_cheaper_routes_to_price_comparison(self, patched_data, mock_openai):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(
            UserQuery(query="which is cheaper, Alpha Laptop or Beta Laptop")
        )

        mock_openai.assert_not_called()
        assert "Alpha Laptop is cheaper than Beta Laptop" in result["response"]

    def test_compare_routes_to_product_comparison(self, patched_data, mock_openai):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(
            UserQuery(query="compare Alpha Laptop and Beta Laptop")
        )

        mock_openai.assert_called_once()
        assert result == {"response": "mocked LLM response"}

    def test_policy_keyword_match_skips_faq(self, patched_data, mock_openai):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="what is your return policy"))

        mock_openai.assert_not_called()
        assert "30-day returns." in result["response"]

    def test_policy_without_keyword_match_falls_back_to_faq(
        self, patched_data, mock_embeddings, in_memory_chroma, mock_openai
    ):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="what is your policy on gift cards"))

        mock_openai.assert_called_once()
        assert result == {"response": "mocked LLM response"}

    def test_expired_warranty_query_falls_back_to_faq(
        self, patched_data, mock_embeddings, in_memory_chroma, mock_openai
    ):
        """Regression test: a repair question phrased around a lapsed warranty
        must not get answered with warranty policy text -- it should fall
        through to FAQAgent's semantic search instead."""
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(
            UserQuery(query="my laptop is no longer under warranty, can you fix it?")
        )

        mock_openai.assert_called_once()
        assert result == {"response": "mocked LLM response"}

    def test_default_falls_back_to_recommendations(self, patched_data, mock_openai):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="I need a laptop under $600"))

        mock_openai.assert_called_once()
        sent = sent_messages(mock_openai)
        assert "Alpha Laptop" in sent
        assert result == {"response": "mocked LLM response"}


# ---------------------------------------------------------------------------
# CoordinatorAgent routing accuracy
#
# The tests above verify routing indirectly, by checking whether an LLM call
# happened and what the final response contains. That conflates two things:
# whether the coordinator picked the right agent, and whether that agent
# behaved correctly. These tests isolate the first question: every
# specialized agent is replaced with a spy, so each case asserts exactly
# which agent handled the query -- no LLM calls, no data lookups, no
# ambiguity from a downstream agent's own logic.
# ---------------------------------------------------------------------------

ROUTES = {
    "review": ("ReviewSummarizationAgent", "analyze_reviews"),
    "price": ("PriceComparisonAgent", "compare_products"),
    "compare": ("ProductComparisonAgent", "compare_products"),
    "policy": ("StorePolicyAgent", "get_policy_info"),
    "faq": ("FAQAgent", "get_policy_info"),
    "recommend": ("ProductRecommendationAgent", "recommend_product"),
}


@pytest.fixture
def routing_spies(monkeypatch):
    """Replace every specialized agent class with a spy that records whether
    it was constructed/called, decoupled from that agent's real behavior."""
    real_no_match_message = agents.StorePolicyAgent.NO_MATCH_MESSAGE

    spies = {}
    for key, (class_name, method_name) in ROUTES.items():
        instance = MagicMock()
        getattr(instance, method_name).return_value = {"response": f"{class_name} handled it"}
        spy_cls = MagicMock(return_value=instance)
        if class_name == "StorePolicyAgent":
            spy_cls.NO_MATCH_MESSAGE = real_no_match_message
        monkeypatch.setattr(agents, class_name, spy_cls)
        spies[key] = spy_cls
    return spies


def assert_routed_to(spies, expected_key):
    """Exactly the expected agent was constructed; every other agent was untouched."""
    for key, spy_cls in spies.items():
        if key == expected_key:
            spy_cls.assert_called_once()
        else:
            spy_cls.assert_not_called()


class TestCoordinatorRoutingAccuracy:
    @pytest.mark.parametrize("query, expected_route", [
        ("reviews for Alpha Laptop", "review"),
        ("what do people think of Beta Laptop", "recommend"),  # no "review" keyword -> default
        ("which is cheaper, Alpha Laptop or Beta Laptop", "price"),
        ("what's the price difference between Alpha Laptop and Beta Laptop", "price"),
        ("what's the price of Alpha Laptop", "recommend"),  # "price" alone, no compare/diff/cost
        ("compare Alpha Laptop and Beta Laptop", "compare"),
        ("what is your return policy", "policy"),
        ("what is your warranty duration", "policy"),  # policy_type keyword without the word "policy"
        ("my laptop is broken, is it still under warranty?", "policy"),
        ("I need a laptop under $600", "recommend"),
        ("recommend a smart tv", "recommend"),
    ])
    def test_routes_to_expected_agent(self, patched_data, routing_spies, query, expected_route):
        coordinator = agents.CoordinatorAgent()
        coordinator.handle_query(UserQuery(query=query))
        assert_routed_to(routing_spies, expected_route)

    def test_cheaper_takes_precedence_over_generic_compare(self, patched_data, routing_spies):
        """Both "cheaper" and "compare" appear; the price-comparison branch is
        checked first in CoordinatorAgent.handle_query, so it should win."""
        coordinator = agents.CoordinatorAgent()
        coordinator.handle_query(
            UserQuery(query="compare these laptops and tell me which is cheaper")
        )
        assert_routed_to(routing_spies, "price")

    def test_review_takes_precedence_over_policy(self, patched_data, routing_spies):
        coordinator = agents.CoordinatorAgent()
        coordinator.handle_query(UserQuery(query="reviews on your return policy"))
        assert_routed_to(routing_spies, "review")

    def test_policy_no_match_falls_through_to_faq(self, patched_data, routing_spies):
        routing_spies["policy"].return_value.get_policy_info.return_value = {
            "response": routing_spies["policy"].NO_MATCH_MESSAGE
        }
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="what is your policy on gift cards"))

        routing_spies["policy"].assert_called_once()
        routing_spies["faq"].assert_called_once()
        assert result == {"response": "FAQAgent handled it"}

    def test_policy_match_does_not_fall_through_to_faq(self, patched_data, routing_spies):
        coordinator = agents.CoordinatorAgent()
        result = coordinator.handle_query(UserQuery(query="what is your return policy"))

        routing_spies["policy"].assert_called_once()
        routing_spies["faq"].assert_not_called()
        assert result == {"response": "StorePolicyAgent handled it"}

    def test_expired_warranty_phrasing_routes_through_policy_to_faq(self, patched_data, routing_spies):
        """Regression coverage at the routing level: an expired-warranty query
        still enters the policy branch (it's about warranty), but StorePolicyAgent
        itself should report no match so the coordinator falls through to FAQAgent."""
        routing_spies["policy"].return_value.get_policy_info.return_value = {
            "response": routing_spies["policy"].NO_MATCH_MESSAGE
        }
        coordinator = agents.CoordinatorAgent()
        coordinator.handle_query(
            UserQuery(query="my laptop is no longer under warranty, can you fix it?")
        )
        routing_spies["policy"].assert_called_once()
        routing_spies["faq"].assert_called_once()

    def test_routing_accuracy_across_labeled_dataset(self, patched_data, routing_spies):
        """
        A small labeled (query -> expected agent) dataset scored as a batch,
        mirroring how routing accuracy would be reported for a real eval: run
        every case, then assert on the aggregate accuracy rather than
        stopping at the first mismatch, so a regression's full blast radius
        is visible in one run.
        """
        labeled_queries = [
            ("reviews for Alpha Laptop", "review"),
            ("which is cheaper, Alpha Laptop or Beta Laptop", "price"),
            ("compare Alpha Laptop and Beta Laptop", "compare"),
            ("what is your return policy", "policy"),
            ("I need a laptop under $600", "recommend"),
        ]

        correct = 0
        misrouted = []
        for query, expected_route in labeled_queries:
            for spy_cls in routing_spies.values():
                spy_cls.reset_mock()

            coordinator = agents.CoordinatorAgent()
            coordinator.handle_query(UserQuery(query=query))

            actual_routes = [key for key, spy_cls in routing_spies.items() if spy_cls.called]
            if actual_routes == [expected_route]:
                correct += 1
            else:
                misrouted.append((query, expected_route, actual_routes))

        accuracy = correct / len(labeled_queries)
        assert accuracy == 1.0, f"routing accuracy {accuracy:.0%}, misrouted: {misrouted}"
