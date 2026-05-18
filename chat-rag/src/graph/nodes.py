import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from graph.state import GraphState
from guardrails import run_input_guardrails, run_output_guardrails
from llm import get_embeddings, get_llm
from prompts import ANSWER_PROMPT, CLASSIFY_PROMPT


def _build_official_context(data: dict) -> str:
    """Flatten the official JSON into a readable string."""
    lines = [
        f"Source: {data.get('source', 'Official Source')}",
        f"Title: {data.get('title', '')}",
        f"Last updated: {data.get('last_updated', '')}",
        "",
    ]

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(v, f"{prefix}{k}: " if not prefix else f"{prefix} > {k}: ")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    lines.append(f"  - {prefix}{item}")
                else:
                    flatten(item, prefix)
        else:
            if obj not in (None, "", True, False) or isinstance(obj, bool):
                lines.append(f"{prefix}{obj}")

    for section, value in data.items():
        if section in ("source", "title", "last_updated"):
            continue
        lines.append(f"\n[{section.replace('_', ' ').upper()}]")
        flatten(value)

    return "\n".join(lines)


def input_guard(state: GraphState) -> dict:
    result = run_input_guardrails(state["question"])
    updates: dict = {
        "question": result.cleaned_text,
        "injection_flagged": result.injection_flagged,
    }
    if result.refusal_message:
        updates["refusal_message"] = result.refusal_message
    return updates


def output_guard(state: GraphState) -> dict:
    result = run_output_guardrails(state["answer"])
    return {"answer": result.response_text}


def classify_question(state: GraphState) -> dict:
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=state["question"]),
    ])
    parts = response.content.strip().lower().split()
    label = parts[0] if parts else ""
    return {"classification": "relevant" if label == "relevant" else "off_topic"}


def reject(state: GraphState) -> dict:
    msg = state.get(  # type: ignore[call-overload]
        "refusal_message",
        "I can only answer questions about visas, travel documents, and embassy processes. Please ask a related question.",
    )
    return {"answer": msg}


def retrieve_from_chat(state: GraphState) -> dict:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    docs = retriever.invoke(state["question"])
    return {"chat_docs": docs}


def load_official_data(_: GraphState) -> dict:
    with Path(settings.official_data_path).open(encoding="utf-8") as f:
        data = json.load(f)
    return {"official_data": data}


def generate_answer(state: GraphState) -> dict:
    chat_context = (
        "\n\n".join(doc.page_content for doc in state["chat_docs"])
        if state["chat_docs"]
        else "No relevant chat messages found."
    )
    official_context = _build_official_context(state["official_data"])

    official_source = state["official_data"].get("source", "Official Source")
    prompt_text = ANSWER_PROMPT.format(
        official_source=official_source,
        official_context=official_context,
        chat_context=chat_context,
        question=state["question"],
    )

    llm = get_llm(temperature=0.1)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {"answer": response.content}
