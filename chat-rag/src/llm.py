import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from config import settings


def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=settings.mistral_api_key,
    )


def get_llm(temperature: float = 0):
    primary = ChatMistralAI(
        model="mistral-small-latest",
        api_key=settings.mistral_api_key,
        temperature=temperature,
        max_retries=6,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it",
        api_key=settings.gemini_api_key,
        temperature=temperature,
    )
    return primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=(httpx.HTTPStatusError, ChatGoogleGenerativeAIError),
    )
