import asyncio
import json
from pathlib import Path

import os
# from config import settings

from dotenv import load_dotenv

from agents import Agent, Runner, set_trace_processors
from agents.extensions.models.litellm_model import LitellmModel
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
from langsmith import Client


load_dotenv()

# TEST_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "test_cases.json"
TEST_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "test_cases_germany.json"


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


async def main():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    agent = Agent(
        name="Captain Obvious",
        instructions="You are Captain Obvious, the world's most literal technical support agent.",
        model=LitellmModel(
            model="mistral/mistral-small-latest",
            api_key=mistral_api_key,
        ),
        # model = ChatMistralAI(
        #     model="mistral-small-latest",
        #     api_key=mistral_api_key, #settings.mistral_api_key,
        #     temperature=0.1,
        # )
    )

    test_cases = load_test_cases()

    ls_client = Client()

    # DATASET_NAME = "visa_qa_7_v1"
    DATASET_NAME = "visa_qa_germany_3_v1"

    if not ls_client.has_dataset(dataset_name=DATASET_NAME):
        dataset = ls_client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Questions and answers about visas (Final Project).",
        )
        ls_client.create_examples(
            dataset_id=dataset.id,
            inputs=[
                {
                    "question": t["input"],
                    "topic": t["topic"],
                    "complexity": t["complexity"],
                    "message_ids": t["message_ids"],
                }
                for t in test_cases
            ],
            outputs=[
                {
                    "expected_answer": t["output"],
                    "case_number": t["case_number"],
                }
                for t in test_cases
            ],
        )
        print(f"Dataset '{DATASET_NAME}' created with {len(test_cases)} examples.")
    else:
        print(f"Dataset '{DATASET_NAME}' already exists.")


    # for i, case in enumerate(test_cases, 1):
    #     question = case["input"]
    #     expected = case["output"]
    #     topic = case["topic"]

    #     print(f"\n{'='*80}")
    #     print(f"Кейс {i}/{len(test_cases)} | Тема: {topic}")
    #     print(f"Вопрос: {question}")
    #     print(f"\nОжидаемый ответ:\n{expected}")
    #     print(f"{'='*80}")


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())