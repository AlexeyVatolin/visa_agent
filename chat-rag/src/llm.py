from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from config import settings


def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=settings.mistral_api_key,
    )


def get_llm(temperature: float = 0) -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=settings.mistral_api_key,
        temperature=temperature,
    )
