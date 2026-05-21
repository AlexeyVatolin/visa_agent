from pathlib import Path

import modal

app = modal.App("visa-rag")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

SRC_DIR = "/root/chat-rag/src"
DATA_DIR = "/root/chat-rag/data"
CHROMA_DIR = "/root/chat-rag/chroma_db"
EXTRACTED_DIR = "/root/data/extracted"

CHROMA_ZIP_URL = (
    "https://storage.yandexcloud.net/ai-visa/chroma_db.zip"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=YCAJE4o7Ul_CBprgFUlN9JEfB%2F20260521%2Fru-central1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260521T130738Z"
    "&X-Amz-Expires=2592000"
    "&X-Amz-Signature=b3409ceb0ef54125a610ea4e606a46452ef33ec5b4e904d4cfb03db27b246ee7"
    "&X-Amz-SignedHeaders=host"
    "&response-content-disposition=attachment"
)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "unzip")
    .run_commands(
        f"curl -sL '{CHROMA_ZIP_URL}' -o /tmp/chroma_db.zip"
        f" && unzip -o /tmp/chroma_db.zip -d /root/chat-rag/"
        f" && rm -rf /tmp/chroma_db.zip /root/chat-rag/__MACOSX"
    )
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
    env={"CHROMA_DIR": CHROMA_DIR},
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
