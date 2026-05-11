from rag import load_chain


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
            m = doc.metadata
            print(
                f"  • Topic: {m.get('topic', '')} | "
                f"{m.get('start_time', '')} → {m.get('end_time', '')} | "
                f"Senders: {m.get('senders', '')}"
            )
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
