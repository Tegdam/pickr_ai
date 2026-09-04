# ragas eval for ProductRecommendationAgent, scored on correctness,
# faithfulness, and relevancy against evals/data/recommendation_evals.json.
#
#     python -m evals.eval_recommendation
#
# Unlike tests/, this makes real API calls -- the agent under test runs live,
# and gpt-4o grades it -- so it costs money and is run deliberately rather
# than in CI.

from dotenv import load_dotenv
from ragas.metrics import AspectCritic, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample

from app.agents import ProductRecommendationAgent
from app.models import UserQuery

import json

load_dotenv()

# Initialize the LLM/embeddings wrappers used to score responses
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# Correctness: does the response match the reference answer.
correctness_scorer = AspectCritic(
    name="correctness",
    llm=evaluator_llm,
    definition="Verify if the response contains factually correct information according to the reference. Score lower if information is missing or contradicts the reference."
)

# Faithfulness: is every claim in the response backed by the shortlisted
# products actually given to the LLM, rather than added by the LLM.
faithfulness_scorer = Faithfulness(llm=evaluator_llm)

# Relevancy: does the response actually address the request asked, independent
# of correctness (a faithful but off-topic answer should still score low here).
relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

METRICS = {
    "correctness": correctness_scorer,
    "faithfulness": faithfulness_scorer,
    "relevancy": relevancy_scorer,
}

async def evaluate_single_recommendation_case(agent, test_case):
    """Score one (query, reference) case for correctness, faithfulness, and relevancy.

    Mirrors eval_agents.py's FAQAgent pattern: retrieved_contexts comes from
    ProductRecommendationAgent.shortlist_context -- the same shortlist-building
    step recommend_product's own LLM call used -- rather than being re-derived
    independently, so faithfulness is scored against what the LLM actually saw.
    """
    query = UserQuery(query=test_case["user_input"])
    query_lower = query.query.lower()
    category = agent._match_category(query_lower)
    brand = agent._match_brand(query_lower)
    max_price = agent._match_price_ceiling(query_lower)
    retrieved_contexts = agent.shortlist_context(category, brand, max_price)

    response_text = agent.recommend_product(query)["response"]

    print(f"\nQuery: {test_case['user_input']}")
    print(f"Response: {response_text}")

    sample = SingleTurnSample(
        user_input=test_case["user_input"],
        response=response_text,
        reference=test_case["reference"],
        retrieved_contexts=retrieved_contexts,
    )

    scores = {
        name: await scorer.single_turn_ascore(sample)
        for name, scorer in METRICS.items()
    }

    return {
        "query": test_case["user_input"],
        "response": response_text,
        **scores,
    }

async def main(eval_dataset_path: str):
    """Load dataset, evaluate ProductRecommendationAgent's responses, and report per-metric averages."""
    results = []

    dataset = json.load(open(eval_dataset_path))
    agent = ProductRecommendationAgent()

    for test_case in dataset:
        result = await evaluate_single_recommendation_case(agent, test_case)
        results.append(result)

    print("\n===== EVALUATION RESULTS =====")
    totals = {name: 0.0 for name in METRICS}

    for i, result in enumerate(results):
        print(f"\nTest case {i+1}:")
        print(f"Query: {result['query']}")
        for name in METRICS:
            print(f"{name.capitalize()}: {result[name]:.2f}")
            totals[name] += result[name]

    print("\n----- Averages -----")
    for name in METRICS:
        print(f"{name.capitalize()}: {totals[name] / len(results):.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main("evals/data/recommendation_evals.json"))
