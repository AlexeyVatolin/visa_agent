import gradio as gr
from rag import load_chain

chain, retriever = load_chain()


def chat(question_text, history):
    if not question_text.strip():
        return history, "", ""

    answer = chain.invoke(question_text)
    docs = retriever.invoke(question_text)

    lines = []
    for doc in docs:
        m = doc.metadata
        lines.append(
            f"• Topic: {m.get('topic', '')} | "
            f"{m.get('start_time', '')} → {m.get('end_time', '')} | "
            f"Senders: {m.get('senders', '')}"
        )
    sources_text = "\n".join(lines) if lines else "No sources found."

    history.append({"role": "user", "content": question_text})
    history.append({"role": "assistant", "content": answer})
    return history, "", sources_text


with gr.Blocks(title="VISA chat History RAG") as demo:
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

if __name__ == "__main__":
    demo.launch()  # For local view
    # demo.launch(share=True) # For public view
