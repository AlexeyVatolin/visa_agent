import json
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from config import settings
from models import ChatExport, ChunkMetadata, Message


def load_messages(path: str) -> list[Message]:
    with open(path, "r", encoding="utf-8") as f:
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

            text = "\n".join(
                f"[{m.date.isoformat()}] {m.from_}: {m.text}" for m in text_msgs
            )

            meta = ChunkMetadata(
                conversation_id=conv_id,
                chat_name=chunk[0].chat_name,
                topic=conv_id,
                start_time=chunk[0].date,
                end_time=chunk[-1].date,
                senders=", ".join(sorted({m.from_ for m in chunk})),
            )

            documents.append(
                Document(page_content=text, metadata=meta.model_dump(mode="json"))
            )

    return documents


def build_vectorstore(documents: list[Document]) -> Chroma:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=settings.mistral_api_key,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_path,
    )
    print(f"✅ Indexed {len(documents)} chunks into ChromaDB.")
    return vectorstore


if __name__ == "__main__":
    messages = load_messages(settings.data_path)
    print(f"Loaded {len(messages)} messages.")
    docs = messages_to_documents(messages)
    print(f"Created {len(docs)} document chunks.")
    build_vectorstore(docs)
