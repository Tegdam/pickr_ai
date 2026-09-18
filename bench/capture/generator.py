"""Seeded query and conversation generator over Pickr's real catalog.

Templates are written to land on each of CoordinatorAgent's keyword rules
("keyword" phrasing) and, for most intents, to miss them so the LLM intent
classifier fallback fires ("natural" phrasing). Slots are filled only from
products/brands/categories/policies that exist in the catalog, so the app's
retrieval steps find real context.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from app.models import Product, Review, StorePolicy


@dataclass(frozen=True)
class GeneratedQuery:
    query_id: str
    text: str
    intent: str
    phrasing: str  # "keyword" | "natural"


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    profile: str  # "shallow" | "medium" | "deep"
    turns: tuple[GeneratedQuery, ...]


@dataclass
class Catalog:
    products: list[Product]
    reviews: list[Review]
    policies: list[StorePolicy]

    @classmethod
    def from_app(cls) -> "Catalog":
        from app.db import load_products, load_reviews, load_store_policies
        return cls(load_products(), load_reviews(), load_store_policies())

    @property
    def in_stock(self) -> list[Product]:
        return [p for p in self.products if p.name and p.stock and p.stock > 0]

    @property
    def reviewed(self) -> list[Product]:
        ids = {r.product_id for r in self.reviews}
        return [p for p in self.products if p.name and p.id in ids]

    @property
    def categories(self) -> list[str]:
        return sorted({p.category for p in self.products if p.category})

    @property
    def brands(self) -> list[str]:
        return sorted({p.brand for p in self.products if p.brand})

    @property
    def policy_types(self) -> list[str]:
        return sorted({p.policy_type for p in self.policies if p.policy_type})


DEFAULT_MIX: dict[str, float] = {
    "recommendation_category_price": 0.20,
    "recommendation_brand": 0.07,
    "recommendation_browse": 0.03,
    "review": 0.22,
    "comparison": 0.14,
    "price_comparison": 0.10,
    "store_policy": 0.14,
    "stock": 0.07,
    "capabilities": 0.03,
}
NATURAL_SHARE = 0.20  # fraction of queries using the no-keyword phrasing, where one exists

PROFILE_DEPTHS = {"shallow": (2, 3), "medium": (4, 5), "deep": (6, 8)}

# (keyword templates, natural templates). Slots: {product} {p1} {p2} {category}
# {category_text} {brand} {price} {policy_type_text}
TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "review": (
        ["What do the reviews say about {product}?", "Summarize the reviews for {product}.",
         "Show me reviews of {product}."],
        ["What do customers think of {product}?", "Is {product} any good according to buyers?"],
    ),
    "price_comparison": (
        ["Which is cheaper, {p1} or {p2}?", "What's the price difference between {p1} and {p2}?",
         "Compare the price of {p1} and {p2}."],
        ["Is {p1} much more expensive than {p2}?"],
    ),
    "comparison": (
        ["Compare {p1} and {p2}.", "How does {p1} compare to {p2}?"],
        ["{p1} vs {p2} — which should I get?"],
    ),
    "recommendation_category_price": (
        ["Recommend a {category_text} under ${price}", "Can you recommend a {category_text} below ${price}?",
         "Suggest a good {category_text} for less than ${price}"],
        ["I need a {category_text}, budget is about ${price}"],
    ),
    "recommendation_brand": (
        ["Recommend a {brand} {category_text}", "Which {brand} {category_text} would you suggest?"],
        ["Got anything from {brand}?"],
    ),
    "recommendation_browse": (
        ["What do you recommend?", "What products do you carry?"],
        ["Just browsing, what's popular?"],
    ),
    "store_policy": (
        ["What is your {policy_type_text} policy?", "Can I return a {product}?",
         "What's the warranty on {product}?"],
        ["Do you price match?", "How long does delivery take?"],
    ),
    "stock": (
        ["Is {product} in stock?", "How many {product} are available?"],
        ["Do you have {product} right now?"],
    ),
    "capabilities": (
        ["What can you do?", "What kind of questions can you answer?"],
        [],
    ),
}

FOLLOW_UPS: dict[str, list[str]] = {
    "recommendation": ["What about something cheaper?", "Any of those from {brand}?",
                       "What do the reviews say about the first one?", "Is the first one in stock?"],
    "review": ["Is it in stock?", "Can I return it if I don't like it?", "Compare it with {p2}."],
    "comparison": ["Which one is cheaper?", "What's the warranty on the second one?",
                   "What do reviews say about the first one?"],
    "price_comparison": ["Which one is cheaper?", "What's the warranty on the second one?"],
    "store_policy": ["Does that apply to a {category_text}?", "What about exchanges?"],
    "stock": ["What do the reviews say about it?", "Recommend something similar under ${price}."],
    "capabilities": ["Recommend a {category_text} under ${price}"],
}


def _fill(template: str, catalog: Catalog, rng: random.Random) -> str:
    in_stock = catalog.in_stock
    p1, p2 = rng.sample(in_stock, 2)
    prices = sorted(p.price for p in in_stock if p.price is not None)
    price = int(rng.choice(prices[len(prices) // 4:]) // 10 * 10) if prices else 500
    category = rng.choice(catalog.categories) if catalog.categories else "laptop"
    policy_type = rng.choice(catalog.policy_types) if catalog.policy_types else "returns"
    return template.format(
        product=rng.choice(in_stock).name,
        p1=p1.name, p2=p2.name,
        category=category, category_text=category.replace("_", " "),
        brand=rng.choice(catalog.brands) if catalog.brands else "Acme",
        price=price,
        policy_type_text=policy_type.replace("_", " "),
    )


def _fill_review(template: str, catalog: Catalog, rng: random.Random) -> str:
    """Review queries must name a product that has reviews, or the agent
    short-circuits without an LLM call."""
    pool = catalog.reviewed or catalog.in_stock
    return template.format(product=rng.choice(pool).name)


def _make_query(query_id: str, intent: str, catalog: Catalog, rng: random.Random) -> GeneratedQuery:
    keyword, natural = TEMPLATES[intent]
    use_natural = bool(natural) and rng.random() < NATURAL_SHARE
    template = rng.choice(natural if use_natural else keyword)
    text = _fill_review(template, catalog, rng) if intent == "review" else _fill(template, catalog, rng)
    return GeneratedQuery(query_id, text, intent, "natural" if use_natural else "keyword")


def generate_queries(catalog: Catalog, n: int, seed: int, mix: dict[str, float] = DEFAULT_MIX) -> list[GeneratedQuery]:
    rng = random.Random(seed)
    intents = list(mix)
    weights = [mix[i] for i in intents]
    # Allocate counts proportionally then shuffle, so the mix is exact rather than sampled.
    counts = {i: int(n * w) for i, w in zip(intents, weights)}
    for i in intents[: n - sum(counts.values())]:
        counts[i] += 1
    schedule = [i for i in intents for _ in range(counts[i])]
    rng.shuffle(schedule)
    return [_make_query(f"q{k:06d}", intent, catalog, rng) for k, intent in enumerate(schedule)]


def _family(intent: str) -> str:
    return "recommendation" if intent.startswith("recommendation") else intent


def generate_conversations(catalog: Catalog, n_per_profile: int, seed: int) -> list[Conversation]:
    rng = random.Random(seed)
    openers = [i for i in DEFAULT_MIX if i != "capabilities"]
    convs: list[Conversation] = []
    for profile, (lo, hi) in PROFILE_DEPTHS.items():
        for k in range(n_per_profile):
            cid = f"conv-{profile}-{k:04d}"
            depth = rng.randint(lo, hi)
            opener_intent = rng.choice(openers)
            turns = [_make_query(f"{cid}-t0", opener_intent, catalog, rng)]
            family = _family(opener_intent)
            for t in range(1, depth):
                template = rng.choice(FOLLOW_UPS[family])
                text = _fill(template, catalog, rng)
                turns.append(GeneratedQuery(f"{cid}-t{t}", text, f"followup_{family}", "natural"))
                # Follow-ups drift the conversation's topic the way real ones do.
                family = rng.choice(list(FOLLOW_UPS))
            convs.append(Conversation(cid, profile, tuple(turns)))
    return convs


def to_json(queries: list[GeneratedQuery], conversations: list[Conversation]) -> dict:
    return {
        "queries": [asdict(q) for q in queries],
        "conversations": [
            {"conversation_id": c.conversation_id, "profile": c.profile, "turns": [asdict(t) for t in c.turns]}
            for c in conversations
        ],
    }


def from_json(d: dict) -> tuple[list[GeneratedQuery], list[Conversation]]:
    qs = [GeneratedQuery(**q) for q in d["queries"]]
    convs = [Conversation(c["conversation_id"], c["profile"], tuple(GeneratedQuery(**t) for t in c["turns"]))
             for c in d["conversations"]]
    return qs, convs
