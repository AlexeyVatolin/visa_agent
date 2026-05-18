import gradio as gr

from ui.handlers import complete_assistant_message, stage_user_message


def build_chat_tab() -> None:
    with gr.Tab("Chat"):
        gr.Markdown("# Chat History RAG")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, label="Chat")
                with gr.Row():
                    question = gr.Textbox(
                        placeholder="Ask a question…", show_label=False, scale=4, container=False
                    )
                    send_btn = gr.Button("Send", scale=1)
            with gr.Column(scale=1):
                sources_box = gr.Textbox(
                    label="Sources", lines=20, interactive=False
                )

    send_btn.click(
        fn=stage_user_message,
        inputs=[question, chatbot],
        outputs=[chatbot, question],
    ).then(
        fn=complete_assistant_message,
        inputs=[chatbot],
        outputs=[chatbot, sources_box],
    )
    question.submit(
        fn=stage_user_message,
        inputs=[question, chatbot],
        outputs=[chatbot, question],
    ).then(
        fn=complete_assistant_message,
        inputs=[chatbot],
        outputs=[chatbot, sources_box],
    )
