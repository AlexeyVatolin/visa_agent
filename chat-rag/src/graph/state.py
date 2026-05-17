from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    classification: str
    chat_docs: list[Document]
    official_data: dict
    answer: str
