import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from models import ChatExport, ChunkMetadata, Message

from config import settings
from llm import get_embeddings
from logging_ import configure_logging, get_logger

logger = get_logger(__name__)

BATCH_SIZE = 100
BATCH_DELAY = 3  # seconds between batches


def load_messages(path: str) -> list[Message]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    export = ChatExport.model_validate(data)

    # Adding metadata for each message
    for msg in export.messages:
        msg.chat_name = export.name
        msg.topic = export.topic

    return export.messages


def messages_to_documents(messages: list[Message]) -> list[Document]:
    # All message from one chat — one conversation_id
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
        logger.info("indexing batch %d/%d (%d chunks)", idx + 1, len(batches), len(batch))
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

    logger.info("indexed %d chunks into ChromaDB", len(documents))
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of messages to ingest"
    )
    args = parser.parse_args()

    configure_logging()
    messages = load_messages(settings.data_path)
    if args.limit:
        messages = messages[: args.limit]
    logger.info("loaded %d messages", len(messages))
    docs = messages_to_documents(messages)
    logger.info("created %d document chunks", len(docs))
    build_vectorstore(docs)
