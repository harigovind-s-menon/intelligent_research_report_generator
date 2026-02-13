"""Tests for the embeddings module."""

import numpy as np
import pytest
from src.models.embeddings import (
    EmbeddingModel,
    get_embedding_model,
    embed_text,
    embed_texts,
    cosine_similarity,
    cosine_similarity_batch,
)


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    @pytest.fixture
    def model(self):
        """Get the embedding model."""
        return get_embedding_model()

    def test_model_loads(self, model):
        """Test that model loads successfully."""
        assert model is not None
        assert model.dimension == 384  # all-MiniLM-L6-v2

    def test_embed_single_text(self, model):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = model.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()

    def test_embed_batch(self, model):
        """Test batch embedding."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]
        embeddings = model.embed_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()

    def test_embed_empty_batch(self, model):
        """Test embedding empty list."""
        embeddings = model.embed_batch([])
        assert len(embeddings) == 0

    def test_similarity_identical_texts(self, model):
        """Test that identical texts have similarity ~1.0."""
        text = "Machine learning is fascinating."
        similarity = model.similarity(text, text)
        assert similarity > 0.99

    def test_similarity_similar_texts(self, model):
        """Test that similar texts have high similarity."""
        text1 = "Machine learning is a type of artificial intelligence."
        text2 = "AI and machine learning are related technologies."
        similarity = model.similarity(text1, text2)
        assert similarity > 0.5

    def test_similarity_different_texts(self, model):
        """Test that different texts have lower similarity."""
        text1 = "Machine learning is fascinating."
        text2 = "I like to eat pizza for dinner."
        similarity = model.similarity(text1, text2)
        assert similarity < 0.5

    def test_find_most_similar(self, model):
        """Test finding most similar texts."""
        query = "What is artificial intelligence?"
        candidates = [
            "AI is a branch of computer science.",
            "Pizza is delicious Italian food.",
            "Machine learning enables computers to learn.",
            "The weather is sunny today.",
        ]
        
        results = model.find_most_similar(query, candidates, top_k=2)
        
        assert len(results) == 2
        # AI-related texts should be most similar
        top_texts = [text for _, text, _ in results]
        assert "AI is a branch of computer science." in top_texts
        assert "Machine learning enables computers to learn." in top_texts

    def test_find_most_similar_empty_candidates(self, model):
        """Test find_most_similar with empty candidates."""
        results = model.find_most_similar("query", [], top_k=5)
        assert results == []

    def test_similarity_matrix(self, model):
        """Test similarity matrix computation."""
        texts = ["Hello world", "Hi there", "Goodbye"]
        matrix = model.similarity_matrix(texts)
        
        assert matrix.shape == (3, 3)
        # Diagonal should be ~1.0 (self-similarity)
        assert all(matrix[i, i] > 0.99 for i in range(3))
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_embed_text(self):
        """Test embed_text convenience function."""
        embedding = embed_text("Test sentence")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_texts(self):
        """Test embed_texts convenience function."""
        embeddings = embed_texts(["First", "Second"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity + 1.0) < 0.001

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_batch(self):
        """Test batch cosine similarity."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # partial match
        ])
        similarities = cosine_similarity_batch(query, candidates)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 0.001  # identical
        assert abs(similarities[1]) < 0.001  # orthogonal
        assert 0 < similarities[2] < 1  # partial match
