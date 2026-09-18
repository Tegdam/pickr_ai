from collections import Counter

import pytest

from bench.capture.generator import (
    DEFAULT_MIX, PROFILE_DEPTHS, Catalog, Conversation, GeneratedQuery,
    from_json, generate_conversations, generate_queries, to_json,
)
from tests.bench.conftest import POLICIES, PRODUCTS, REVIEWS


@pytest.fixture
def catalog():
    return Catalog(products=PRODUCTS, reviews=REVIEWS, policies=POLICIES)


def test_generate_queries_is_deterministic_for_a_seed(catalog):
    a = generate_queries(catalog, n=50, seed=7)
    b = generate_queries(catalog, n=50, seed=7)
    c = generate_queries(catalog, n=50, seed=8)
    assert [q.text for q in a] == [q.text for q in b]
    assert [q.text for q in a] != [q.text for q in c]


def test_generate_queries_count_ids_and_mix(catalog):
    qs = generate_queries(catalog, n=200, seed=1)
    assert len(qs) == 200
    assert [q.query_id for q in qs] == [f"q{i:06d}" for i in range(200)]
    counts = Counter(q.intent for q in qs)
    for intent, share in DEFAULT_MIX.items():
        assert abs(counts[intent] / 200 - share) < 0.06, (intent, counts[intent])
    assert {q.phrasing for q in qs} <= {"keyword", "natural"}
    natural_share = sum(q.phrasing == "natural" for q in qs) / 200
    assert 0.10 <= natural_share <= 0.30


def test_slots_are_filled_from_the_catalog(catalog):
    qs = generate_queries(catalog, n=300, seed=3)
    names = {p.name for p in PRODUCTS}
    brands = {p.brand for p in PRODUCTS}
    for q in qs:
        assert "{" not in q.text and "}" not in q.text, q.text
    review_qs = [q for q in qs if q.intent == "review"]
    assert review_qs and all(any(n in q.text for n in names) for q in review_qs)
    # review queries only name products that actually have reviews
    reviewed = {p.name for p in PRODUCTS if any(r.product_id == p.id for r in REVIEWS)}
    assert all(any(n in q.text for n in reviewed) for q in review_qs)
    brand_qs = [q for q in qs if q.intent == "recommendation_brand"]
    assert brand_qs and all(any(b in q.text for b in brands) for q in brand_qs)


def test_keyword_phrasings_hit_the_intended_keyword_rule(catalog):
    qs = generate_queries(catalog, n=300, seed=5)
    for q in qs:
        if q.phrasing != "keyword":
            continue
        t = q.text.lower()
        if q.intent == "review":
            assert "review" in t
        elif q.intent == "price_comparison":
            assert "cheaper" in t or ("price" in t and any(w in t for w in ("compare", "difference", "cost")))
        elif q.intent == "comparison":
            assert "compare" in t
        elif q.intent == "stock":
            assert "stock" in t or "availab" in t
        elif q.intent == "capabilities":
            # Both keyword templates are in CoordinatorAgent._CAPABILITY_PHRASES.
            assert "what can you do" in t or "what kind of questions can you answer" in t


def test_generate_conversations_profiles_and_depths(catalog):
    convs = generate_conversations(catalog, n_per_profile=5, seed=11)
    assert len(convs) == 15
    assert Counter(c.profile for c in convs) == {"shallow": 5, "medium": 5, "deep": 5}
    for c in convs:
        lo, hi = PROFILE_DEPTHS[c.profile]
        assert lo <= len(c.turns) <= hi
        assert c.turns[0].phrasing in ("keyword", "natural")
        assert all(t.query_id.startswith(c.conversation_id) for t in c.turns)
        assert [t.query_id for t in c.turns] == [f"{c.conversation_id}-t{i}" for i in range(len(c.turns))]
    assert convs == generate_conversations(catalog, n_per_profile=5, seed=11)


def test_json_round_trip(catalog):
    qs = generate_queries(catalog, n=10, seed=2)
    convs = generate_conversations(catalog, n_per_profile=1, seed=2)
    qs2, convs2 = from_json(to_json(qs, convs))
    assert qs2 == qs and convs2 == convs
