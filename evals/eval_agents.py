from dotenv import load_dotenv
from ragas.metrics import AspectCritic
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample

from app.agents import CoordinatorAgent
from app.models import UserQuery

import json

load_dotenv()

# Initialize the LLM wrapper for evaluation
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# Define an evaluation metric for correctness
correctness_scorer = AspectCritic(
    name="correctness",
    llm=evaluator_llm,
    definition="Verify if the response contains factually correct information according to the reference. Score lower if information is missing or contradicts the reference."
)

async def evaluate_single_store_policy_case(coordinator, test_case):
    """Evaluate the factual correctness of the AI response for a given test case."""
    response = coordinator.handle_query(UserQuery(query=test_case["user_input"]))
    response_text = response["response"]

    print(f"\nQuery: {test_case['user_input']}")
    print(f"Response: {response_text}")

    # Add response to test case
    test_case_with_response = test_case.copy()
    test_case_with_response["response"] = response_text

    # Create a sample for evaluation
    sample = SingleTurnSample(**test_case_with_response)

    score = await correctness_scorer.single_turn_ascore(sample)

    return {
        "query": test_case["user_input"],
        "response": response_text,
        "score": score,
    }

async def main(eval_dataset_path: str):
    """Load dataset, evaluate responses, and compute an overall score."""
    results = []

    dataset = json.load(open(eval_dataset_path))
    coordinator = CoordinatorAgent()

    for test_case in dataset:
        result = await evaluate_single_store_policy_case(coordinator, test_case)
        results.append(result)

    print("\n===== EVALUATION RESULTS =====")
    total_score = 0

    for i, result in enumerate(results):
        print(f"\nTest case {i+1}:")
        print(f"Query: {result['query']}")
        print(f"Score: {result['score']}")
        total_score += result['score']

    avg_score = total_score / len(results)
    print(f"\nAverage score across {len(results)} test cases: {avg_score:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main("evals/data/store_policies_evals.json"))

# ===========================
# Instructions for Learners:
# ===========================
# How to create `store_policies_evals.json`:
# - This file should contain a list of test cases.
# - Each test case should have two keys:
#   1. `user_input`: The customer’s query.
#   2. `reference`: The correct or expected answer.
#
# Example JSON Structure:
# [
#     {
#         "user_input": "What is the minimum amount I have to spend to get free shipping?",
#         "reference": "The minimum amount you have to spend to get free shipping is $50."
#     },
#     {
#         "user_input": "Do you offer international shipping?",
#         "reference": "We currently only ship to addresses within the United States."
#     }
# ]
#
# This dataset will be used for evaluating the AI responses' factual correctness.