import asyncio
import os
# from config import settings

from dotenv import load_dotenv

# from langchain_mistralai import ChatMistralAI
#TODO не поняла, почему предыдущий импорт не сраотал, ведь он работает в соседнем файле rag.py

from agents import Agent, Runner, set_trace_processors
from agents.extensions.models.litellm_model import LitellmModel
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor


load_dotenv()


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

    question = "Why is my code failing when I try to divide by zero? I keep getting this error message."
    result = await Runner.run(agent, question)

    print(result.final_output)


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())