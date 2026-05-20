import html
from typing import Any

import gradio as gr

from graph import visa_graph
from ingest import ChunkMetadata
from logging_ import get_logger

logger = get_logger(__name__)


def _sources_to_html(docs: list) -> str:
    if not docs:
        return "<p style='color:#888;font-family:sans-serif;'>No sources found.</p>"

    cards: list[str] = []
    for doc in docs:
        meta = ChunkMetadata.model_validate(doc.metadata)
        topic = html.escape(meta.topic)
        time_range = html.escape(f"{meta.start_time.strftime('%Y-%m-%d %H:%M')} → {meta.end_time.strftime('%H:%M')}")
        senders = html.escape(meta.senders)
        preview = html.escape(doc.page_content[:280])
        if len(doc.page_content) > 280:
            preview += "…"
        preview = preview.replace("\n", "<br>")

        cards.append(
            f"""<div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;
                           margin-bottom:10px;background:#fafafa;font-family:sans-serif;">
              <div style="font-weight:600;margin-bottom:4px;">📂 {topic}</div>
              <div style="color:#666;font-size:12px;margin-bottom:2px;">🕐 {time_range}</div>
              <div style="color:#888;font-size:11px;margin-bottom:8px;">👥 {senders}</div>
              <div style="color:#333;font-size:13px;line-height:1.5;">{preview}</div>
            </div>"""
        )

    return (
        "<div style='height:500px;overflow-y:auto;padding-right:4px;'>"
        + "".join(cards)
        + "</div>"
    )


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
    try:
        result = visa_graph.invoke(
            {
                "question": question_text,
                "classification": "",
                "country": "",
                "chat_docs": [],
                "official_data": {},
                "answer": "",
            }
        )
        answer = result.get("answer") or "I couldn't generate a response. Please try again."
        docs = result.get("chat_docs", [])
    except Exception as e:
        logger.error("graph invocation failed: %s", e)
        answer = "Something went wrong. Please try your question again."
        docs = []

    return [*history, {"role": "assistant", "content": answer}], _sources_to_html(docs)
