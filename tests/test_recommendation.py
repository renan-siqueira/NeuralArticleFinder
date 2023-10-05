"""
Unit tests for the recommendation functionalities of NeuralArticleFinder project.
"""
import unittest
import numpy as np
from gensim.models import Word2Vec
from src.recommendation.recommend import vectorize_document, most_similar_docs


class TestRecommendationFunctions(unittest.TestCase):
    """Test suite for recommendation functions."""

    @classmethod
    def setUpClass(cls):
        cls.docs = [["hello", "world"], ["test", "function"], ["another", "document"]]
        cls.model = Word2Vec(
            sentences=cls.docs,
            vector_size=10,
            window=2,
            min_count=1,
            workers=1,
            epochs=10
        )

    def test_vectorize_document(self):
        """Tests vector representation of documents using Word2Vec model."""

        # Testing a document with words known by the model
        vec = vectorize_document(['hello', 'world'], self.model)
        self.assertTrue(np.any(vec))

        # Testing a document without words known to the model
        vec = vectorize_document(['unknown', 'words'], self.model)
        self.assertTrue(np.all(vec == 0))

    def test_most_similar_docs(self):
        """Tests ranking of documents based on their similarity to a target document."""

        # Very similar document
        top_docs = most_similar_docs(['hello', 'world'], self.model, self.docs)
        self.assertEqual(top_docs[0][0], ['hello', 'world'])

        # Document without words known by the model
        top_docs = most_similar_docs(['unknown', 'words'], self.model, self.docs)
        self.assertEqual(top_docs, [])


if __name__ == '__main__':
    unittest.main()
