import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import settings
from graph.state import GraphState
from guardrails import run_input_guardrails
from guardrails._internal_errors import rewrite_internal_error_leaks
from guardrails._output_tokens import redact_output_pii
from llm import get_embeddings, get_llm
from logging_ import get_logger, trace
from prompts import ANSWER_PROMPT, CLASSIFY_PROMPT, COUNTRY_DETECT_PROMPT, KNOWN_COUNTRIES

logger = get_logger(__name__)


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
    with trace("input_guard", question=state["question"][:60]):
        result = run_input_guardrails(state["question"])
        updates: dict = {
            "question": result.cleaned_text,
            "injection_flagged": result.injection_flagged,
        }
        if result.refusal_message:
            updates["refusal_message"] = result.refusal_message
        return updates


def output_guard(state: GraphState) -> dict:
    # PII redaction is applied to chat docs in retrieve_from_chat; here we only
    # scrub internal error details that the LLM may have echoed.
    with trace("output_guard"):
        cleaned, rewritten, concerns = rewrite_internal_error_leaks(state["answer"])
        if rewritten:
            logger.warning("output_guardrail internal_rewritten=True concerns=%s", concerns)
        else:
            logger.info("output_guardrail clean")
        return {"answer": cleaned}


def classify_question(state: GraphState) -> dict:
    with trace("classify_question", question=state["question"][:60]):
        llm = get_llm(temperature=0) | StrOutputParser()
        label = llm.invoke(
            [
                SystemMessage(content=CLASSIFY_PROMPT),
                HumanMessage(content=state["question"]),
            ]
        )
        if label not in ("relevant", "off_topic"):
            logger.warning("unexpected classification label %r; treating as off_topic", label)
        classification = "relevant" if label == "relevant" else "off_topic"
        logger.debug("classification=%s", classification)
        return {"classification": classification}


def reject(state: GraphState) -> dict:
    msg = state.get(  # type: ignore[call-overload]
        "refusal_message",
        "I can only answer questions about visas, travel documents, and embassy processes. Please ask a related question.",
    )
    return {"answer": msg}


def retrieve_from_chat(state: GraphState) -> dict:
    with trace("retrieve_from_chat"):
        try:
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
            for doc in docs:
                doc.page_content, n_pii, _ = redact_output_pii(doc.page_content)
                if n_pii:
                    logger.debug("redacted %d PII tokens from chat doc", n_pii)
            logger.debug("retrieved %d docs", len(docs))
            return {"chat_docs": docs}
        except Exception as e:
            logger.warning("chroma retrieval failed: %s", e)
            return {"chat_docs": [], "error": "Community knowledge base temporarily unavailable."}


def _detect_country_llm(question: str) -> str | None:
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=COUNTRY_DETECT_PROMPT),
        HumanMessage(content=question),
    ])
    slug = response.content.strip().lower().replace(" ", "_")
    return slug if slug in KNOWN_COUNTRIES else None


def load_official_data(state: GraphState) -> dict:
    with trace("load_official_data"):
        country = _detect_country_llm(state["question"])
        if country is None:
            logger.debug("no country detected in question")
            return {"official_data": {}, "country": ""}

        file_path = Path(settings.official_data_dir) / f"{country}_visa_official.json"
        try:
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("loaded official data country=%s keys=%s", country, list(data.keys())[:5])
            return {"official_data": data, "country": country}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("official data load failed country=%s: %s", country, e)
            return {"official_data": {}, "country": country, "error": "Official visa data temporarily unavailable."}


def generate_answer(state: GraphState) -> dict:
    with trace("generate_answer", docs=len(state.get("chat_docs") or [])):
        if state.get("error"):
            return {"answer": "I'm having trouble accessing my knowledge sources right now. Please try again shortly."}
        chat_context = (
            "\n\n".join(doc.page_content for doc in state["chat_docs"])
            if state["chat_docs"]
            else "No relevant chat messages found."
        )
        official_context = (
            _build_official_context(state["official_data"])
            if state["official_data"]
            else "No country-specific official visa data available for this query."
        )

        official_source = state["official_data"].get("source", "Official Source")
        prompt_text = ANSWER_PROMPT.format(
            official_source=official_source,
            official_context=official_context,
            chat_context=chat_context,
            question=state["question"],
        )

        llm = get_llm(temperature=0.1) | StrOutputParser()
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return {"answer": response}
