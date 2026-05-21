from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import logging

from config import settings
from logging_ import configure_logging

configure_logging(level=getattr(logging, settings.loglevel, logging.INFO))

import gradio as gr
import typer

from ui.chat_tab import build_chat_tab
from ui.dashboard_tab import build_dashboard_tab

with gr.Blocks(title="VISA Dashboard") as demo:
    build_dashboard_tab()
    build_chat_tab()


def main(share: bool = False):
    # demo.launch(share=share, theme=gr.Theme.from_hub("hmb/windows95"))
    demo.launch(share=share, theme=gr.themes.Citrus())


if __name__ == "__main__":
    typer.run(main)
