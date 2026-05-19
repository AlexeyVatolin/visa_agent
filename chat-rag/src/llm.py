import os
import threading

import httpx
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

from config import settings

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_ready = threading.Event()


def _init_embeddings() -> None:
    global _embeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name="Octen/Octen-Embedding-0.6B",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _embeddings_ready.set()


threading.Thread(target=_init_embeddings, daemon=True).start()


def get_embeddings() -> Embeddings:
    _embeddings_ready.wait()
    return _embeddings  # type: ignore[return-value]


def get_llm(temperature: float = 0) -> ChatMistralAI:
    primary = ChatMistralAI(
        model="mistral-small-latest",
        api_key=settings.mistral_api_key,
        temperature=temperature,
        max_retries=6,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        api_key=settings.gemini_api_key,
        temperature=temperature,
    )
    return primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=(httpx.HTTPStatusError, ChatGoogleGenerativeAIError),
    )
