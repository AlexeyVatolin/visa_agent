from typing import Any

import gradio as gr

from graph import visa_graph
from ingest import ChunkMetadata


def content_to_text(content: Any) -> str:
    """Extract plain text from a Gradio message content field.

    Gradio's Chatbot normalizes content to a list of content blocks between
    .then() steps, so content may arrive as str or list[dict].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "content" in part and isinstance(part["content"], str):
                    parts.append(part["content"])
        return "".join(parts)
    return str(content) if content is not None else ""


def stage_user_message(question_text: str, history: list) -> tuple[list, str]:
    if not question_text.strip():
        return history, question_text
    return [*history, {"role": "user", "content": question_text}], ""


def complete_assistant_message(history: list) -> tuple[list, str]:
    if not history or history[-1]["role"] != "user":
        return gr.update(), gr.update()

    question_text = content_to_text(history[-1]["content"])
    result = visa_graph.invoke(
        {
            "question": question_text,
            "classification": "",
            "chat_docs": [],
            "official_data": {},
            "answer": "",
        }
    )
    answer = result["answer"]
    docs = result["chat_docs"]

    lines = []
    for doc in docs:
        meta = ChunkMetadata.model_validate(doc.metadata)
        lines.append(
            f"• Topic: {meta.topic} | "
            f"{meta.start_time} → {meta.end_time} | "
            f"Senders: {meta.senders}\n"
            f"{doc.page_content}"
        )
    sources_text = "\n\n".join(lines) if lines else "No sources found."

    return [*history, {"role": "assistant", "content": answer}], sources_text
