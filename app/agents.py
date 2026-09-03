# Import Required Libraries needed for AI processing and data retrieval

import chromadb  # Persistent vector store used for the FAQ agent's retrieval step.
import hashlib  # Used to detect when the policy index is stale and needs rebuilding.
import json  # Used to parse the LLM intent-classifier's JSON response.
import logging  # Structured logging of CoordinatorAgent's routing decisions.
import re  # Used to pull price constraints (e.g. "under $800") out of free-text queries.
import time  # Used to time how long each routed agent takes to handle a query.

from .openai_client import client
from .guardrails import check_input, check_output

# Import functions to load product details, customer reviews, and store policies from CSV files.
from .db import load_products, load_reviews, load_store_policies

# Import the UserQuery model, which defines the structure of a user query.
from .models import UserQuery


logger = logging.getLogger(__name__)


# Fallback intent classifier -- only consulted when none of CoordinatorAgent's
# keyword rules match a query. Keyword routing stays the fast, free, fully
# deterministic path for the common case; this LLM call only runs on the
# minority of queries that would otherwise silently default to
# ProductRecommendationAgent, e.g. novel phrasing the keyword rules don't
# anticipate. Kept as a narrow fallback rather than replacing keyword routing
# outright, to avoid both a new cost/latency floor on every query and
# regression risk against the existing keyword-routing test suite.
_INTENT_CATEGORIES = ("review", "price_comparison", "comparison", "store_policy", "recommendation")

_INTENT_CLASSIFIER_SYSTEM_PROMPT = """You are an intent router for a retail shopping assistant with five specialized agents. Classify the customer's message into exactly one category and respond with ONLY a JSON object of this exact shape:
{"category": "review" | "price_comparison" | "comparison" | "store_policy" | "recommendation"}

- review: asking what customers think of a product, or for a summary of its reviews.
- price_comparison: asking for the exact price difference between two or more named products, or which one is cheaper.
- comparison: asking to compare two or more named products on features (not just price).
- store_policy: asking about returns, refunds, warranty, shipping, exchanges, or other store policies.
- recommendation: asking for a product suggestion, or anything that doesn't clearly fit the categories above."""


def _guarded_response(response_text: str, context_text: str) -> dict:
    """Return {"response": response_text}, or the guardrail's refusal message if
    check_output flags response_text as unsupported by context_text. Shared by
    every agent that generates a response from an LLM call, so the guardrail
    wiring lives in one place instead of being copy-pasted per agent.
    """
    output_check = check_output(response_text, context_text)
    if output_check["blocked"]:
        return {"response": output_check["message"]}
    return {"response": response_text}


# Define the CoordinatorAgent Class
# This class is responsible for processing user queries and routing them to the appropriate specialized agent.
# It acts as the central control unit for handling different types of customer inquiries.

class CoordinatorAgent:
    def __init__(self):
        """
        Initialize the CoordinatorAgent by loading product, review, and policy data.
        Ensure that the 'load_products', 'load_reviews', and 'load_store_policies' functions
        correctly read data from the respective CSV files.
        """
        self.products = load_products()  # Load product information
        self.reviews = load_reviews()  # Load customer reviews
        self.policies = load_store_policies()  # Load store policies

    def handle_query(self, query: UserQuery):
        """
        Process the user query and determine the appropriate specialized agent.
        The query parameter should be an instance of UserQuery.
        The function should check the content of the query and route it accordingly.
        """
        query_lower = query.query.lower()
        start = time.monotonic()
        agent_name = "unknown"
        status = "error"
        block_reason = None
        routed_via = "keyword"
        try:
            input_check = check_input(query.raw_query or query.query)
            if input_check["blocked"]:
                agent_name = "guardrails_input"
                status = "blocked"
                block_reason = input_check["reason"]
                return {"response": input_check["message"]}

            if "review" in query_lower:
                agent_name = "ReviewSummarizationAgent"
                result = ReviewSummarizationAgent().analyze_reviews(query)
            elif "cheaper" in query_lower or (
                "price" in query_lower
                and any(w in query_lower for w in ["compare", "difference", "cost"])
            ):
                agent_name = "PriceComparisonAgent"
                result = PriceComparisonAgent().compare_products(query)
                if result["response"] == PriceComparisonAgent.NOT_ENOUGH_PRODUCTS_MESSAGE:
                    # "cheaper"/"price" alone doesn't always mean "compare these two
                    # named products" -- e.g. "recommend something cheaper" has no
                    # second product to compare against. Fall back to recommendations
                    # rather than surfacing a dead-end "mention two products" reply.
                    agent_name = "ProductRecommendationAgent"
                    result = ProductRecommendationAgent().recommend_product(query)
            elif "compare" in query_lower:
                agent_name = "ProductComparisonAgent"
                result = ProductComparisonAgent().compare_products(query)
            elif self._is_policy_query(query_lower):
                agent_name = "StorePolicyAgent"
                result = StorePolicyAgent().get_policy_info(query)
                if result["response"] == StorePolicyAgent.NO_MATCH_MESSAGE:
                    agent_name = "FAQAgent"
                    result = FAQAgent().get_policy_info(query)
            else:
                # No keyword rule matched -- rather than silently assuming this is
                # a recommendation request, ask a small LLM classifier which of
                # the five agent categories actually fits. Narrow, contained
                # fallback: see the comment on _INTENT_CATEGORIES above.
                routed_via = "llm_fallback"
                category = self._classify_intent(query.query)
                if category == "review":
                    agent_name = "ReviewSummarizationAgent"
                    result = ReviewSummarizationAgent().analyze_reviews(query)
                elif category == "price_comparison":
                    agent_name = "PriceComparisonAgent"
                    result = PriceComparisonAgent().compare_products(query)
                    if result["response"] == PriceComparisonAgent.NOT_ENOUGH_PRODUCTS_MESSAGE:
                        agent_name = "ProductRecommendationAgent"
                        result = ProductRecommendationAgent().recommend_product(query)
                elif category == "comparison":
                    agent_name = "ProductComparisonAgent"
                    result = ProductComparisonAgent().compare_products(query)
                elif category == "store_policy":
                    agent_name = "StorePolicyAgent"
                    result = StorePolicyAgent().get_policy_info(query)
                    if result["response"] == StorePolicyAgent.NO_MATCH_MESSAGE:
                        agent_name = "FAQAgent"
                        result = FAQAgent().get_policy_info(query)
                else:  # "recommendation", or the classifier failed/returned something unexpected
                    agent_name = "ProductRecommendationAgent"
                    result = ProductRecommendationAgent().recommend_product(query)
            status = "ok"
            return result
        finally:
            # logfmt-style (key=value) rather than a JSON formatter: keeps this
            # readable in plain console output while still being greppable/parseable,
            # without coupling the root logging config to this logger's fields.
            logger.info(
                "coordinator_route agent=%s via=%s status=%s reason=%s elapsed_ms=%.1f query=%r",
                agent_name, routed_via, status, block_reason, (time.monotonic() - start) * 1000, query.query,
            )

    def _is_policy_query(self, query_lower: str) -> bool:
        """
        True if the query is about "policy" or mentions a known policy_type keyword
        (e.g. "refund"), even if the literal word "policy" never appears.
        """
        if "policy" in query_lower:
            return True
        return any(
            p.policy_type and any(word in query_lower for word in p.policy_type.lower().split())
            for p in self.policies
        )

    def _classify_intent(self, query_text: str) -> str:
        """Ask an LLM which of the five agent categories query_text belongs to.
        Only called when no keyword rule matched (see the else branch above).
        Fails open to "recommendation" on any error or unexpected response --
        the same category the pre-existing keyword-only default used, so a
        classifier hiccup degrades to prior behavior rather than raising.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": _INTENT_CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": query_text},
                ],
            )
            category = json.loads(response.choices[0].message.content).get("category")
            if category in _INTENT_CATEGORIES:
                return category
        except Exception:
            logger.warning("intent classifier fallback failed; defaulting to recommendation", exc_info=True)
        return "recommendation"


# Implement Specialized Agents to handle specific types of queries.
# Each agent is responsible for a particular function such as summarizing reviews, comparing products, or retrieving store policies.

# Review Summarization Agent
class ReviewSummarizationAgent:
    def __init__(self):
        self.products = load_products()
        self.reviews = load_reviews()

    def analyze_reviews(self, query: UserQuery):
        """

        This agent is responsible for analyzing and summarizing customer reviews for a given product using OpenAI's API.

        """
        # Find the product mentioned in the query
        query_lower = query.query.lower()
        matched_product = next(
            (p for p in self.products if p.name and p.name.lower() in query_lower), None
        )

        if not matched_product:
            return {"response": "I couldn't find a product matching your query. Please include a product name."}

        # Retrieve reviews for that product
        product_reviews = [r for r in self.reviews if r.product_id == matched_product.id]

        if not product_reviews:
            return {"response": f"No reviews found for {matched_product.name}."}

        # Format reviews and ask OpenAI to summarize them
        reviews_text = "\n".join([
            f"- Rating: {r.rating}/5 | {r.text}"
            for r in product_reviews
        ])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant. Summarize customer reviews concisely, highlighting common praise and complaints."},
                {"role": "user", "content": f"Summarize these reviews for {matched_product.name}:\n{reviews_text}"},
            ],
        )

        # Return the summary, unless the faithfulness/moderation guardrail flags it.
        response_text = response.choices[0].message.content
        return _guarded_response(response_text, reviews_text)


# Implement the Product Recommendation Agent
class ProductRecommendationAgent:
    def __init__(self):
        self.products = load_products()

    def recommend_product(self, query: UserQuery):
        """
        Provide personalized product recommendations based on query.

        Parses optional category/brand/price-ceiling constraints out of the query,
        filters the in-stock catalog against them, and asks OpenAI to phrase a
        recommendation from that shortlist (never from the full catalog, so it
        can't recommend something outside the filtered results).
        """
        query_lower = query.query.lower()

        category = self._match_category(query_lower)
        brand = self._match_brand(query_lower)
        max_price = self._match_price_ceiling(query_lower)

        context_lines = self.shortlist_context(category, brand, max_price)
        if not context_lines:
            return {"response": "I couldn't find any in-stock products matching your request. Try adjusting the category, brand, or price range."}

        product_details = "\n".join(context_lines)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant. Recommend products only from the provided shortlist, and briefly explain why each recommended product fits the customer's request."},
                {"role": "user", "content": f"Customer request: {query.query}\n\nShortlist of matching products:\n{product_details}"},
            ],
        )
        response_text = response.choices[0].message.content
        return _guarded_response(response_text, product_details)

    def shortlist_context(self, category, brand, max_price) -> list:
        """Return the formatted context lines for the top-5 in-stock shortlist
        matching the given constraints -- one string per product, mirroring
        PolicyIndex.search()'s list-of-chunks shape so both can be scored by
        the same ragas faithfulness/relevancy eval pattern (see evals/).
        """
        candidates = [p for p in self.products if p.stock and p.stock > 0]
        if category:
            candidates = [p for p in candidates if p.category and p.category.lower() == category]
        if brand:
            candidates = [p for p in candidates if p.brand and p.brand.lower() == brand]
        if max_price is not None:
            candidates = [p for p in candidates if p.price is not None and p.price <= max_price]

        # Highest-rated matches first; cap the shortlist so the prompt stays small.
        shortlist = sorted(candidates, key=lambda p: p.rating or 0, reverse=True)[:5]

        return [
            f"- {p.name} (Brand: {p.brand}, Category: {p.category}, Price: ${p.price}, Rating: {p.rating}/5): {p.description}"
            for p in shortlist
        ]

    def _match_category(self, query_lower: str):
        """Return a catalog category (e.g. 'smart_tv') if it's referenced in the query."""
        categories = {p.category.lower() for p in self.products if p.category}
        return next(
            (c for c in categories if c in query_lower or c.replace("_", " ") in query_lower),
            None,
        )

    def _match_brand(self, query_lower: str):
        """Return a catalog brand (e.g. 'techco') if it's referenced in the query."""
        brands = {p.brand.lower() for p in self.products if p.brand}
        return next((b for b in brands if b in query_lower), None)

    def _match_price_ceiling(self, query_lower: str):
        """Extract a price ceiling from phrasing like 'under $800' or 'less than 500'."""
        match = re.search(r"(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)", query_lower)
        return float(match.group(1)) if match else None


# Product Comparison Agent
class ProductComparisonAgent:
    def __init__(self):
        self.products = load_products()

    def compare_products(self, query: UserQuery):
        """
        This agent compares multiple products based on key attributes like price, features, and ratings.

        """
        # Find products whose names appear in the query
        query_lower = query.query.lower()
        matched = self.matched_products(query_lower)

        if len(matched) < 2:
            return {"response": "Please mention at least two product names to compare."}

        # Build a summary of each product's key attributes
        product_details = "\n".join(self.product_context(matched))

        # Ask OpenAI to produce a clear, structured comparison
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant. Compare products clearly and concisely."},
                {"role": "user", "content": f"Compare these products:\n{product_details}"},
            ],
        )
        response_text = response.choices[0].message.content
        return _guarded_response(response_text, product_details)

    def matched_products(self, query_lower: str) -> list:
        """Return catalog products whose name appears in the query."""
        return [p for p in self.products if p.name and p.name.lower() in query_lower]

    def product_context(self, products) -> list:
        """Format products as context lines -- shared by compare_products and
        by evals/eval_comparison.py, which needs the same retrieved_contexts
        the LLM call actually saw for faithfulness scoring."""
        return [
            f"- {p.name} (Brand: {p.brand}, Price: ${p.price}, Rating: {p.rating}/5): {p.description}"
            for p in products
        ]


# Price Comparison Agent
class PriceComparisonAgent:
    NOT_ENOUGH_PRODUCTS_MESSAGE = "Please mention at least two product names (with known prices) to compare."

    def __init__(self):
        self.products = load_products()

    def compare_products(self, query: UserQuery):
        """
        Compare products and return price differences.

        Deterministic (no LLM call): matches named products against the query,
        then reports their prices, which is cheapest, and the $ / % difference.
        """
        query_lower = query.query.lower()
        matched = [
            p for p in self.products
            if p.name and p.name.lower() in query_lower and p.price is not None
        ]

        if len(matched) < 2:
            return {"response": self.NOT_ENOUGH_PRODUCTS_MESSAGE}

        matched.sort(key=lambda p: p.price)
        cheapest, priciest = matched[0], matched[-1]

        lines = [f"- {p.name}: ${p.price:.2f}" for p in matched]

        if len(matched) == 2:
            diff = priciest.price - cheapest.price
            pct = (diff / priciest.price * 100) if priciest.price else 0
            lines.append(
                f"{cheapest.name} is cheaper than {priciest.name} by ${diff:.2f} ({pct:.1f}% less)."
            )
        else:
            spread = priciest.price - cheapest.price
            lines.append(
                f"Cheapest: {cheapest.name} (${cheapest.price:.2f}) | "
                f"Most expensive: {priciest.name} (${priciest.price:.2f}) | "
                f"Spread: ${spread:.2f}"
            )

        return {"response": "\n".join(lines)}


# Store Policy Agent
class StorePolicyAgent:
    NO_MATCH_MESSAGE = "I couldn't find a matching policy. Try asking about return, refund, shipping, or warranty policies."

    # Customers often mention "warranty" only to say it's lapsed (e.g. "no longer
    # under warranty", "out of warranty") while actually asking about something
    # else, like repairs. Without stripping that phrasing first, the keyword match
    # below would treat it as a warranty question and never fall through to
    # FAQAgent's semantic search, which is what should handle these.
    EXPIRED_WARRANTY_PATTERN = re.compile(r"(no longer|out of|outside of|expired|past).{0,15}warranty")

    def __init__(self):
        self.policies = load_store_policies()

    def get_policy_info(self, query: UserQuery):
        """
        This agent fetches store policies based on customer queries (e.g., refund, return, shipping policies).

        """
        # Find policies whose type keyword appears in the query
        query_lower = self.EXPIRED_WARRANTY_PATTERN.sub("", query.query.lower())
        matched = [
            p for p in self.policies
            if p.policy_type and any(word in query_lower for word in p.policy_type.lower().split())
        ]

        # Format and return the matching policy details
        if not matched:
            return {"response": self.NO_MATCH_MESSAGE}

        policy_text = "\n\n".join([
            f"{p.policy_type}\n{p.description}\nConditions: {p.conditions}\nTimeframe: {p.timeframe}"
            for p in matched
        ])
        return {"response": policy_text}


# Retrieval index over store policies, backed by Chroma. Embeds each policy once
# (persisted to disk so restarts don't re-pay the embedding cost) and retrieves the
# most relevant chunks for a query, instead of stuffing every policy into the prompt.
class PolicyIndex:
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_TEMPLATE = "{policy_type}\n{description}\nConditions: {conditions}\nTimeframe: {timeframe}"

    def __init__(self, policies, persist_path="data/chroma_db", collection_name="store_policies"):
        self.policies = policies
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection_name = collection_name
        self.collection = self._client.get_or_create_collection(collection_name)

        # A count check alone misses in-place edits (same number of rows, changed
        # content), so compare a content hash instead: any added, removed, or
        # edited row changes the hash and triggers a full rebuild.
        current_hash = self._content_hash()
        stored_hash = (self.collection.metadata or {}).get("content_hash")
        if stored_hash != current_hash:
            self._rebuild_index(current_hash)

    def _chunk_text(self, policy):
        return self.CHUNK_TEMPLATE.format(
            policy_type=policy.policy_type,
            description=policy.description,
            conditions=policy.conditions,
            timeframe=policy.timeframe,
        )

    def _content_hash(self):
        chunk_texts = sorted(self._chunk_text(p) for p in self.policies)
        return hashlib.sha256("\n".join(chunk_texts).encode()).hexdigest()

    def _embed(self, texts):
        response = client.embeddings.create(model=self.EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def _rebuild_index(self, content_hash):
        # Delete-and-recreate rather than a partial upsert: a partial upsert keyed
        # by list index would leave orphaned entries behind if the policy count
        # ever shrinks, on top of not handling in-place edits at all.
        self._client.delete_collection(self._collection_name)
        self.collection = self._client.get_or_create_collection(self._collection_name)

        chunks = [self._chunk_text(p) for p in self.policies]
        if chunks:
            ids = [f"policy-{i}" for i in range(len(chunks))]
            self.collection.upsert(ids=ids, embeddings=self._embed(chunks), documents=chunks)

        self.collection.modify(metadata={"content_hash": content_hash})

    def search(self, query_text: str, k: int = 3):
        """Return the top-k policy chunks most relevant to the query text."""
        if self.collection.count() == 0:
            return []
        query_embedding = self._embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
        )
        return results["documents"][0]


# Implement the FAQ & Store Policy Handling Agent
class FAQAgent:
    NO_MATCH_MESSAGE = "I couldn't find anything in our store policies relevant to that question."

    def __init__(self):
        self.policies = load_store_policies()
        self.index = PolicyIndex(self.policies)

    def get_policy_info(self, query: UserQuery):
        """
        Fetch and return relevant store policy information via retrieval-augmented generation.

        Fallback for when StorePolicyAgent's keyword match finds nothing: retrieves the
        most relevant policy chunks with PolicyIndex and lets OpenAI answer from just those,
        instead of requiring an exact policy_type word in the query or stuffing every policy
        into the prompt.
        """
        relevant_chunks = self.index.search(query.query, k=3)

        if not relevant_chunks:
            return {"response": self.NO_MATCH_MESSAGE}

        policy_text = "\n\n".join(relevant_chunks)

        # temperature=0: answering from retrieved policy text is a grounding task, not a
        # creative one, and default sampling occasionally produced a false "we don't have
        # that policy" refusal even when the right chunk was in context (e.g. hedging on
        # "over $1000" vs. a policy's literal "over $999" threshold).
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant. Answer the customer's question using only the store policies provided below. If none of the policies address the question, say so honestly rather than guessing."},
                {"role": "user", "content": f"Store policies:\n{policy_text}\n\nCustomer question: {query.query}"},
            ],
        )
        response_text = response.choices[0].message.content
        return _guarded_response(response_text, policy_text)
