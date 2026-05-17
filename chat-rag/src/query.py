from graph import visa_graph
from ingest import ChunkMetadata


def main():
    print("Chat History + Official Visa Info RAG — type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = visa_graph.invoke({"question": question, "classification": "", "chat_docs": [], "official_data": {}, "answer": ""})

        print(f"\nAnswer:\n{result['answer']}\n")

        print("Sources (chat history):")
        for doc in result["chat_docs"]:
            meta = ChunkMetadata.model_validate(doc.metadata)
            print(
                f"  • Topic: {meta.topic} | "
                f"{meta.start_time} → {meta.end_time} | "
                f"Senders: {meta.senders}"
            )
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
