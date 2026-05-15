import json

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from config import settings
from prompts import ANSWER_PROMPT, CLASSIFY_PROMPT
from graph.state import GraphState


def _build_official_context(data: dict, question: str) -> str:
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


def classify_question(state: GraphState) -> dict:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=settings.mistral_api_key,
        temperature=0,
    )
    response = llm.invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=state["question"]),
    ])
    label = response.content.strip().lower().split()[0]
    return {"classification": "relevant" if label == "relevant" else "off_topic"}


def reject(_: GraphState) -> dict:
    return {"answer": "I can only answer questions about visas, travel documents, and embassy processes. Please ask a related question."}


def retrieve_from_chat(state: GraphState) -> dict:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=settings.mistral_api_key,
    )
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
    with open(settings.official_data_path, encoding="utf-8") as f:
        data = json.load(f)
    return {"official_data": data}


def generate_answer(state: GraphState) -> dict:
    chat_context = (
        "\n\n".join(doc.page_content for doc in state["chat_docs"])
        if state["chat_docs"]
        else "No relevant chat messages found."
    )
    official_context = _build_official_context(state["official_data"], state["question"])

    official_source = state["official_data"].get("source", "Official Source")
    prompt_text = ANSWER_PROMPT.format(
        official_source=official_source,
        official_context=official_context,
        chat_context=chat_context,
        question=state["question"],
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=settings.mistral_api_key,
        temperature=0.1,
    )
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {"answer": response.content}
