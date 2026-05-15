import json
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import settings

PROMPT_TEMPLATE = """You are a visa information assistant. Answer the user's question using two sources of information provided below.

Clearly distinguish between:
- [OFFICIAL] — information from the official German Embassy website
- [COMMUNITY] — information shared in community chats (may be personal experience, not guaranteed accurate)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OFFICIAL] Information from German Embassy Belgrade (belgrad.diplo.de):
{official_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[COMMUNITY] Information from chat history:
{chat_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: {question}

Instructions:
- Start your answer with official information when available, clearly marked as [OFFICIAL].
- Add community insights marked as [COMMUNITY] where they add useful context.
- If official data answers the question fully, say so.
- If the chat has no relevant info, say "No community insights found for this topic."
- Never mix sources without labeling them.

Answer:"""


CLASSIFY_PROMPT = """You are a guardrail for a German visa information assistant.
Classify the user's question. Reply with exactly one word:
- relevant  — if the question is about visas, travel documents, embassy processes, appointments, required documents, fees, waiting times, or related immigration topics
- off_topic — for anything else"""


class GraphState(TypedDict):
    question: str
    classification: str
    chat_docs: list[Document]
    official_data: dict
    answer: str


def _build_official_context(data: dict, question: str) -> str:
    """Flatten the official JSON into a readable string."""
    lines = [
        f"Source: {data.get('source', 'German Embassy Belgrade')}",
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


def route_after_classify(state: GraphState) -> list[str] | str:
    if state["classification"] == "relevant":
        return ["retrieve_from_chat", "load_official_data"]
    return "reject"


def retrieve_from_chat(state: GraphState) -> GraphState:
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


def load_official_data(state: GraphState) -> GraphState:
    with open(settings.official_data_path, encoding="utf-8") as f:
        data = json.load(f)
    return {"official_data": data}


def generate_answer(state: GraphState) -> GraphState:
    chat_context = (
        "\n\n".join(doc.page_content for doc in state["chat_docs"])
        if state["chat_docs"]
        else "No relevant chat messages found."
    )
    official_context = _build_official_context(state["official_data"], state["question"])

    prompt_text = PROMPT_TEMPLATE.format(
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


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_from_chat", retrieve_from_chat)
    graph.add_node("load_official_data", load_official_data)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("reject", reject)

    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges("classify_question", route_after_classify)

    # Fan-in: generate waits for both retrieval nodes
    graph.add_edge("retrieve_from_chat", "generate_answer")
    graph.add_edge("load_official_data", "generate_answer")

    graph.add_edge("generate_answer", END)
    graph.add_edge("reject", END)

    return graph.compile()


visa_graph = build_graph()
