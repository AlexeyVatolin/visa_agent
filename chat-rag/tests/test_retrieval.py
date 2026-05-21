import unittest

from langchain_chroma import Chroma

from config import settings
from graph.nodes import retrieve_from_chat
from graph.state import GraphState
from llm import get_embeddings
from prompts import slug_to_collection


class TestRetrievalCollection(unittest.TestCase):
    """Integration tests that require data to be ingested into ChromaDB beforehand."""

    @classmethod
    def setUpClass(cls):
        cls.embeddings = get_embeddings()
        cls.germany_collection = slug_to_collection("germany")
        cls.vectorstore = Chroma(
            collection_name=cls.germany_collection,
            embedding_function=cls.embeddings,
            persist_directory=settings.chroma_path,
        )

    def test_collection_is_not_empty(self):
        count = self.vectorstore._collection.count()
        self.assertGreater(
            count, 0, "ChromaDB collection is empty — run `uv run src/ingest/ingest.py` first"
        )

    def test_retrieve_returns_docs_for_simple_question(self):
        state: GraphState = {
            "question": "Как получить визу в Германию?",
            "classification": "",
            "country": "germany",
            "chat_docs": [],
            "official_data": {},
            "answer": "",
        }
        result = retrieve_from_chat(state)
        docs = result.get("chat_docs", [])
        self.assertGreater(
            len(docs), 0, "retrieve_from_chat returned no documents for a simple visa question"
        )

    def test_retrieve_error_field_absent_on_success(self):
        state: GraphState = {
            "question": "документы для визы в Италию",
            "classification": "",
            "country": "italy",
            "chat_docs": [],
            "official_data": {},
            "answer": "",
        }
        result = retrieve_from_chat(state)
        self.assertNotIn("error", result, "Unexpected 'error' key in retrieve result")

    def test_slug_to_collection_special_cases(self):
        self.assertEqual(slug_to_collection("bulgaria"), "Bulgaria_Romania_Cyprus")
        self.assertEqual(slug_to_collection("poland"), "Poland_visa_D")
        self.assertEqual(slug_to_collection("united_kingdom"), "United_Kingdom")
        self.assertEqual(slug_to_collection("usa"), "USA")

    def test_slug_to_collection_default(self):
        self.assertEqual(slug_to_collection("germany"), "Germany")
        self.assertEqual(slug_to_collection("france"), "France")
        self.assertEqual(slug_to_collection("united_kingdom"), "United_Kingdom")


if __name__ == "__main__":
    unittest.main()
