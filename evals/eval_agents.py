# ragas eval for FAQAgent's retrieval-augmented policy answers, scored on
# correctness, faithfulness, and relevancy against
# evals/data/store_policies_evals.json.
#
#     python -m evals.eval_agents
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

from app.agents import FAQAgent
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

# Faithfulness: is every claim in the response backed by the retrieved policy
# chunks, rather than added by the LLM (i.e. hallucinated beyond what retrieval found).
faithfulness_scorer = Faithfulness(llm=evaluator_llm)

# Relevancy: does the response actually address the question asked, independent
# of correctness (a faithful but off-topic answer should still score low here).
relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

METRICS = {
    "correctness": correctness_scorer,
    "faithfulness": faithfulness_scorer,
    "relevancy": relevancy_scorer,
}

async def evaluate_single_store_policy_case(faq_agent, test_case):
    """Score one (query, reference) case for correctness, faithfulness, and relevancy.

    Retrieval is run directly against faq_agent.index rather than only reading
    faq_agent.get_policy_info's return value, because faithfulness needs the
    retrieved_contexts that response actually was (or wasn't) grounded in --
    the same chunks get_policy_info's own retrieval step would have used.
    """
    query = UserQuery(query=test_case["user_input"])
    retrieved_contexts = faq_agent.index.search(query.query, k=3)
    response_text = faq_agent.get_policy_info(query)["response"]

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
    """Load dataset, evaluate FAQAgent's responses, and report per-metric averages."""
    results = []

    dataset = json.load(open(eval_dataset_path))
    faq_agent = FAQAgent()

    for test_case in dataset:
        result = await evaluate_single_store_policy_case(faq_agent, test_case)
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
    asyncio.run(main("evals/data/store_policies_evals.json"))