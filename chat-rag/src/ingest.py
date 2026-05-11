import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from constants import CHROMA_PATH, COLLECTION_NAME, DATA_PATH

load_dotenv()


def load_messages(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Getting metadata from topic and messages
    chat_name = data.get("name", "")
    topic = data.get("topic", "")
    messages = data.get("messages", [])

    # Adding metadata for each message
    for msg in messages:
        msg["chat_name"] = chat_name
        msg["topic"] = topic

    return messages


def messages_to_documents(messages: list[dict]) -> list[Document]:
    # All message from one chat — one conversation_id
    convos: dict[str, list[dict]] = {}
    for msg in messages:
        cid = msg.get("topic", "default")
        convos.setdefault(cid, []).append(msg)

    documents = []
    window_size = 5  # TODO: Think about logic for creating chunks
    step = 2  # 2 steps betwen messages in chunk to save context
    # chunk 1 - [1, 2, 3, 4, 5]
    # chunk 2 - [3, 4, 5, 6, 7]
    # chunk 3 - [5, 6, 6, 7, 8]

    for conv_id, msgs in convos.items():
        msgs = sorted(msgs, key=lambda m: m.get("date", ""))

        for i in range(0, len(msgs), step):
            chunk = msgs[i : i + window_size]

            text = "\n".join(
                f"[{m.get('date', '')}] {m.get('from', 'Unknown')}: {m.get('text', '')}"
                for m in chunk
                if m.get("text")  # skipping empty messages
            )

            if not text.strip():
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "conversation_id": conv_id,
                    "chat_name": chunk[0].get("chat_name", ""),
                    "topic": conv_id,
                    "start_time": chunk[0].get("date", ""),
                    "end_time": chunk[-1].get("date", ""),
                    "senders": ", ".join(sorted(set(m.get("from", "") for m in chunk))),
                },
            )
            documents.append(doc)

    return documents


def build_vectorstore(documents: list[Document]) -> Chroma:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"✅ Indexed {len(documents)} chunks into ChromaDB.")
    return vectorstore


if __name__ == "__main__":
    messages = load_messages(DATA_PATH)
    print(f"Loaded {len(messages)} messages.")
    docs = messages_to_documents(messages)
    print(f"Created {len(docs)} document chunks.")
    build_vectorstore(docs)
