from pathlib import Path

import modal

app = modal.App("visa-rag")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

SRC_DIR = "/root/chat-rag/src"
DATA_DIR = "/root/chat-rag/data"
CHROMA_DIR = "/root/chat-rag/chroma_db"
EXTRACTED_DIR = "/root/data/extracted"

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync()
    .env(
        {
            "PYTHONPATH": SRC_DIR,
            "DATA_PATH": f"{DATA_DIR}/messages.json",
            "OFFICIAL_DATA_PATH": f"{DATA_DIR}/germany_visa_official.json",
            "CHROMA_PATH": CHROMA_DIR,
        }
    )
    .add_local_dir(HERE / "src", remote_path=SRC_DIR)
    .add_local_dir(HERE / "data", remote_path=DATA_DIR)
    .add_local_dir(HERE / "chroma_db", remote_path=CHROMA_DIR)
    .add_local_dir(PROJECT_ROOT / "data" / "extracted", remote_path=EXTRACTED_DIR)
)

with image.imports():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from query_app import demo

secrets = (
    modal.Secret.from_dotenv()
    if Path(".env").exists()
    else modal.Secret.from_name("visa-rag-secrets")
)


@app.function(
    image=image,
    secrets=[secrets],
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
