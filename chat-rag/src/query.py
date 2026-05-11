from rag import load_chain
from models import ChunkMetadata


def main():
    print("🔍 Chat History RAG — type 'quit' to exit\n")
    chain, retriever = load_chain()

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\n💬 Answer:\n{answer}\n")

        docs = retriever.invoke(question)
        print("📎 Sources:")
        for doc in docs:
            meta = ChunkMetadata.model_validate(doc.metadata)
            print(
                f"  • Topic: {meta.topic} | "
                f"{meta.start_time} → {meta.end_time} | "
                f"Senders: {meta.senders}"
            )
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
