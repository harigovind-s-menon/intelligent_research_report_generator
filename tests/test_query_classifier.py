"""Tests for the query classifier."""

import pytest
from src.models.query_classifier import QueryClassifier, get_classifier, QUERY_TYPES


class TestQueryClassifier:
    """Tests for QueryClassifier."""

    @pytest.fixture
    def classifier(self):
        """Get the trained classifier."""
        clf = get_classifier()
        if not clf.is_loaded:
            pytest.skip("Classifier model not trained. Run: python -m src.models.train")
        return clf

    def test_classifier_is_loaded(self, classifier):
        """Test that classifier loads successfully."""
        assert classifier.is_loaded
        assert classifier.pipeline is not None

    def test_predict_returns_valid_type(self, classifier):
        """Test that predict returns a valid query type."""
        result = classifier.predict("What is the capital of France?")
        assert result in QUERY_TYPES

    def test_predict_proba_returns_all_types(self, classifier):
        """Test that predict_proba returns probabilities for all types."""
        probas = classifier.predict_proba("What is machine learning?")
        assert set(probas.keys()) == set(QUERY_TYPES)
        assert all(0 <= p <= 1 for p in probas.values())
        assert abs(sum(probas.values()) - 1.0) < 0.01  # Probabilities sum to 1

    def test_predict_with_confidence(self, classifier):
        """Test that predict_with_confidence returns type and score."""
        query_type, confidence = classifier.predict_with_confidence("What is Python?")
        assert query_type in QUERY_TYPES
        assert 0 <= confidence <= 1

    # Test classification of different query types
    @pytest.mark.parametrize("query,expected_type", [
        ("What is the capital of France?", "factual"),
        ("Who invented the telephone?", "factual"),
        ("When was Python created?", "factual"),
        ("How does machine learning work?", "exploratory"),
        ("Tell me about quantum computing", "exploratory"),
        ("What are the key aspects of blockchain?", "exploratory"),
        ("Python vs JavaScript which is better?", "comparative"),
        ("Compare React and Vue", "comparative"),
        ("What is the difference between SQL and NoSQL?", "comparative"),
        ("What are the latest news on AI?", "current_events"),
        ("Recent developments in quantum computing", "current_events"),
        ("What happened with OpenAI today?", "current_events"),
    ])
    def test_classification_accuracy(self, classifier, query, expected_type):
        """Test that queries are classified correctly."""
        predicted, confidence = classifier.predict_with_confidence(query)
        # Allow for some flexibility - check if expected is in top 2
        probas = classifier.predict_proba(query)
        sorted_types = sorted(probas.keys(), key=lambda x: probas[x], reverse=True)
        top_2 = sorted_types[:2]
        assert expected_type in top_2, f"Expected {expected_type} for '{query}', got {predicted} (top 2: {top_2})"

    def test_empty_query_handling(self, classifier):
        """Test handling of empty or short queries."""
        # Should not crash, even with minimal input
        result = classifier.predict("test")
        assert result in QUERY_TYPES

    def test_long_query_handling(self, classifier):
        """Test handling of long queries."""
        long_query = "What are the implications of " + "very " * 100 + "long queries?"
        result = classifier.predict(long_query)
        assert result in QUERY_TYPES


class TestQueryClassifierNotLoaded:
    """Tests for classifier when model is not loaded."""

    def test_predict_raises_without_model(self):
        """Test that predict raises error when no model is loaded."""
        from pathlib import Path
        clf = QueryClassifier(model_path=Path("/nonexistent/path.joblib"))
        with pytest.raises(ValueError, match="No model loaded"):
            clf.predict("test query")

    def test_is_loaded_returns_false(self):
        """Test is_loaded returns False when no model."""
        from pathlib import Path
        clf = QueryClassifier(model_path=Path("/nonexistent/path.joblib"))
        assert clf.is_loaded is False
