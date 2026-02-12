"""Sentence embeddings using sentence-transformers.

This module provides utilities for:
- Generating embeddings for text
- Computing semantic similarity
- Batch embedding for efficiency
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast
# Alternative: "all-mpnet-base-v2"  # 768 dimensions, higher quality


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (num_texts, dimension)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(cosine_similarity(emb1, emb2))

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.

        Args:
            texts: List of texts

        Returns:
            Numpy array of shape (num_texts, num_texts)
        """
        embeddings = self.embed_batch(texts)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return np.dot(normalized, normalized.T)

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (index, text, score) tuples, sorted by similarity
        """
        if not candidates:
            return []

        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = cosine_similarity_batch(query_emb, candidate_embs)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and multiple candidates."""
    # Normalize query
    query_norm = query / np.linalg.norm(query)
    # Normalize candidates
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidates_normalized = candidates / candidate_norms
    # Compute similarities
    return np.dot(candidates_normalized, query_norm)


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def embed_text(text: str) -> np.ndarray:
    """Convenience function to embed a single text."""
    return get_embedding_model().embed(text)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convenience function to embed multiple texts."""
    return get_embedding_model().embed_batch(texts)


if __name__ == "__main__":
    # Test the embedding model
    logging.basicConfig(level=logging.INFO)

    model = EmbeddingModel()

    # Test single embedding
    text = "What are the latest developments in quantum computing?"
    embedding = model.embed(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {model.dimension}")

    # Test similarity
    texts = [
        "What are the latest developments in quantum computing?",
        "Recent advances in quantum computers and qubits",
        "How to make a chocolate cake",
        "Quantum mechanics and computing breakthroughs",
    ]

    print("\nSimilarity to query:")
    results = model.find_most_similar(texts[0], texts[1:], top_k=3)
    for idx, text, score in results:
        print(f"  {score:.3f}: {text}")

    # Test batch embedding
    embeddings = model.embed_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")

    # Test similarity matrix
    sim_matrix = model.similarity_matrix(texts)
    print(f"\nSimilarity matrix:\n{sim_matrix.round(3)}")
