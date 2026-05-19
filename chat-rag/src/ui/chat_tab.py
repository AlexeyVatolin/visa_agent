import gradio as gr

from ui.handlers import complete_assistant_message, stage_user_message


def build_chat_tab() -> None:
    with gr.Tab("Chat"):
        gr.Markdown("# Chat History RAG")
        sources_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                with gr.Row():
                    question = gr.Textbox(
                        placeholder="Ask a question…", show_label=False, scale=4, container=False
                    )
                    send_btn = gr.Button("Send", scale=1)
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Sources")
                sources_box = gr.HTML(value="")

    chatbot.change(
        fn=lambda s: s,
        inputs=[sources_state],
        outputs=[sources_box],
    )

    send_btn.click(
        fn=stage_user_message,
        inputs=[question, chatbot],
        outputs=[chatbot, question],
    ).then(
        fn=complete_assistant_message,
        inputs=[chatbot],
        outputs=[chatbot, sources_state],
    )
    question.submit(
        fn=stage_user_message,
        inputs=[question, chatbot],
        outputs=[chatbot, question],
    ).then(
        fn=complete_assistant_message,
        inputs=[chatbot],
        outputs=[chatbot, sources_state],
    )
