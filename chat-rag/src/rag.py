import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "chat_history"

PROMPT_TEMPLATE = """You are an assistant analyzing Telegram chat history.
Use the retrieved conversation excerpts below to answer the question accurately.
If the answer is not in the excerpts, say "I could not find that in the chat history."

Conversation excerpts:
{context}

Question: {question}

Answer:"""


def load_chain():
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.environ["MISTRAL_API_KEY"],
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever