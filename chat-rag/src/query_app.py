import typer
import gradio as gr
from dashboard import compute_tourist_stats
from graph import visa_graph
from models import ChunkMetadata


def chat(question_text, history):
    if not question_text.strip():
        return history, "", ""

    result = visa_graph.invoke({"question": question_text, "classification": "", "chat_docs": [], "official_data": {}, "answer": ""})
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

    history.append({"role": "user", "content": question_text})
    history.append({"role": "assistant", "content": answer})
    return history, "", sources_text


DASHBOARD_HEADERS = [
    "Country",
    "Avg Wait (days)",
    "Total Cases",
    "Approval Rate (%)",
    "Avg Duration (days)",
    "Multivisas",
]
DASHBOARD_DATATYPES = ["str", "number", "number", "number", "number", "number"]

with gr.Blocks(title="VISA Dashboard") as demo:
    with gr.Tab("Dashboard"):
        gr.Markdown("# Tourist Visa Dashboard")
        gr.Dataframe(
            headers=DASHBOARD_HEADERS,
            datatype=DASHBOARD_DATATYPES,
            value=compute_tourist_stats(),
            interactive=False,
            elem_classes="dashboard-table",
            wrap=True,
        )
    with gr.Tab("Chat"):
        gr.Markdown("# Chat History RAG")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    question = gr.Textbox(
                        placeholder="Ask a question…", show_label=False, scale=4
                    )
                    send_btn = gr.Button("Send", scale=1)
            with gr.Column(scale=1):
                sources_box = gr.Textbox(label="Sources", lines=20, interactive=False)

    send_btn.click(chat, [question, chatbot], [chatbot, question, sources_box])
    question.submit(chat, [question, chatbot], [chatbot, question, sources_box])


def main(share: bool = False):
    demo.launch(share=share)


if __name__ == "__main__":
    typer.run(main)
