import argparse
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate, evaluate_existing
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from graph import visa_graph

load_dotenv()

# DATASET_NAME = "visa_qa_7_v1"
# DATASET_NAME = "visa_qa_germany_3_v1"
DATASET_NAME = "visa_qa_11_simple_v1"


class EvalScore(BaseModel):
    """Schema for guided structured output from the judge LLM."""

    reasoning: str = Field(description="Brief explanation of the score")
    score: float = Field(description="Score between 0.0 and 1.0")

ls_client = Client()

judge_llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0,
)

JUDGE_PROMPT = """You are an expert evaluator. Compare the model's answer with the reference answer for a visa-related question.

Rate how well the model's answer covers the key information from the reference on a categorical scale (ONLY scores 0.0 or 0.5 or 1.0):
- 1.0: All key facts from the reference are present and correct
- 0.5: Some key facts are present, but important details are missing or partially wrong
- 0.0: The answer is incorrect, irrelevant, or misses all key facts

Be strict about factual accuracy. Do not penalize differences in wording or extra harmless information.

Question: {question}

Reference answer: {reference}

Model's answer: {prediction}

Provide your reasoning and a score between 0.0 and 1.0."""


def target(inputs: dict) -> dict:
    """Run the RAG graph on a single test example."""
    result = visa_graph.invoke({
        "question": inputs["question"],
        "chat_docs": [],
        "official_data": {},
        "answer": "",
    })
    return {"answer": result["answer"]}


def correctness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: compare model answer against reference."""
    prediction = outputs.get("answer", "")
    reference = reference_outputs.get("expected_answer", "")

    prompt = JUDGE_PROMPT.format(
        question="",
        reference=reference,
        prediction=prediction,
    )
    structured_judge = judge_llm.with_structured_output(EvalScore)
    result: EvalScore = structured_judge.invoke([HumanMessage(content=prompt)])
    score = result.score
    comment = result.reasoning
    print(f"  [eval] score={score}")
    
    return {"key": "correctness", "score": score, "comment": comment}


def correctness_evaluator_existing(run, example) -> dict:
    """LLM-as-judge for evaluate_existing: takes (run, example) instead of (outputs, reference_outputs)."""
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("expected_answer", "")

    prompt = JUDGE_PROMPT.format(
        question="",
        reference=reference,
        prediction=prediction,
    )
    structured_judge = judge_llm.with_structured_output(EvalScore)
    result: EvalScore = structured_judge.invoke([HumanMessage(content=prompt)])
    score = result.score
    comment = result.reasoning
    print(f"  [eval] score={score}")

    return {"key": "correctness", "score": score, "comment": comment}


def main():
    dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)

    results = evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator],
        experiment_prefix="visa-rag",
    )

    print(f"Evaluated {DATASET_NAME}: {results}")


def main_existing(experiment_name: str):
    results = evaluate_existing(
        experiment_name,
        evaluators=[correctness_evaluator_existing],
    )

    print(f"Re-evaluated experiment '{experiment_name}': {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith evaluations")
    parser.add_argument(
        "--existing",
        metavar="EXPERIMENT_NAME",
        help="Re-evaluate an existing experiment by name (skips RAG)",
    )
    args = parser.parse_args()

    if args.existing:
        main_existing(args.existing)
        # main_existing(EXPERIMENT_NAME)
    else:
        main()
