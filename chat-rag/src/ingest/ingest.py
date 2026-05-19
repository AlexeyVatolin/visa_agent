import json
import time
from pathlib import Path

import typer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from models import ChatExport, ChunkMetadata, Message

from config import settings
from llm import get_embeddings

BATCH_SIZE = 100
BATCH_DELAY = 3  # seconds between batches

app = typer.Typer()


def load_messages(path: Path) -> list[Message]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    export = ChatExport.model_validate(data)

    for msg in export.messages:
        msg.chat_name = export.name
        msg.topic = export.topic

    return export.messages


def messages_to_documents(messages: list[Message]) -> list[Document]:
    convos: dict[str, list[Message]] = {}
    for msg in messages:
        convos.setdefault(msg.topic or "default", []).append(msg)

    documents = []
    window_size = 5  # TODO: Think about logic for creating chunks
    step = 2  # 2 steps betwen messages in chunk to save context
    # chunk 1 - [1, 2, 3, 4, 5]
    # chunk 2 - [3, 4, 5, 6, 7]
    # chunk 3 - [5, 6, 6, 7, 8]

    for conv_id, msgs in convos.items():
        msgs = sorted(msgs, key=lambda m: m.date)

        for i in range(0, len(msgs), step):
            chunk = msgs[i : i + window_size]
            text_msgs = [m for m in chunk if m.text]

            if not text_msgs:
                continue

            text = "\n".join(f"[{m.date.isoformat()}] {m.from_}: {m.text}" for m in text_msgs)

            meta = ChunkMetadata(
                conversation_id=conv_id,
                chat_name=chunk[0].chat_name,
                topic=conv_id,
                start_time=chunk[0].date,
                end_time=chunk[-1].date,
                senders=", ".join(sorted({m.from_ for m in chunk})),
            )

            documents.append(Document(page_content=text, metadata=meta.model_dump(mode="json")))

    return documents


def build_vectorstore(documents: list[Document]) -> Chroma:
    embeddings = get_embeddings()

    batches = [documents[i : i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
    vectorstore = None

    for idx, batch in enumerate(batches):
        print(f"Indexing batch {idx + 1}/{len(batches)} ({len(batch)} chunks)...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=settings.collection_name,
                persist_directory=settings.chroma_path,
            )
        else:
            vectorstore.add_documents(batch)

        if idx < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    print(f"✅ Indexed {len(documents)} chunks into ChromaDB.")
    return vectorstore


@app.command()
def ingest(
    data_dir: Path | None = typer.Option(
        None,
        help="Directory with per-country subfolders containing messages.json. "
        "Defaults to settings.data_path parent's parsed/ dir.",
    ),
    limit: int = typer.Option(None, help="Limit number of messages per country"),
    countries: list[str] = typer.Option(None, help="Specific country folders to ingest (repeatable)"),
) -> None:
    if data_dir is None:
        data_dir = settings.data_path.parent.parent / "parsed"

    if not data_dir.is_dir():
        typer.echo(f"Data directory not found: {data_dir}", err=True)
        raise typer.Exit(1)

    # Discover per-country message files
    if countries:
        dirs = [data_dir / c for c in countries]
    else:
        dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "messages.json").exists())

    if not dirs:
        typer.echo("No country folders with messages.json found.", err=True)
        raise typer.Exit(1)

    all_messages: list[Message] = []
    for d in dirs:
        messages = load_messages(d / "messages.json")
        if limit:
            messages = messages[:limit]
        all_messages.extend(messages)
        typer.echo(f"  {d.name}: {len(messages)} messages (topic: {messages[0].topic if messages else 'n/a'})")

    typer.echo(f"\nLoaded {len(all_messages)} messages from {len(dirs)} countries.")

    docs = messages_to_documents(all_messages)
    typer.echo(f"Created {len(docs)} document chunks.")

    build_vectorstore(docs)


if __name__ == "__main__":
    app()
