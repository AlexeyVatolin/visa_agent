from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    classification: str
    chat_docs: list[Document]
    official_data: dict
    answer: str
    # Set by input_guard node; present from that node onward.
    injection_flagged: NotRequired[bool]
    pii_redactions: NotRequired[int]
    out_of_scope: NotRequired[bool]
    refusal_message: NotRequired[str]
