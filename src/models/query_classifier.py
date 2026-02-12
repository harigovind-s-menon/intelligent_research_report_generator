"""Query classifier using scikit-learn.

This module provides a text classifier that categorizes research queries
into one of five types: factual, exploratory, comparative, analytical, current_events.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

logger = logging.getLogger(__name__)

# Query types
QUERY_TYPES = ["factual", "exploratory", "comparative", "analytical", "current_events"]

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "query_classifier.joblib"


class QueryClassifier:
    """Scikit-learn based query classifier."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to saved model. If None, uses default path.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model if it exists."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded query classifier from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.pipeline = None
        else:
            logger.info("No trained model found. Call train() first.")

    def train(
        self,
        train_data: list[dict],
        test_data: Optional[list[dict]] = None,
        save: bool = True,
    ) -> dict:
        """
        Train the classifier.

        Args:
            train_data: List of {"query": str, "label": str} dicts
            test_data: Optional test data for evaluation
            save: Whether to save the trained model

        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training classifier on {len(train_data)} samples")

        # Extract features and labels
        X_train = [d["query"] for d in train_data]
        y_train = [d["label"] for d in train_data]

        # Create pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                stop_words="english",
            )),
            ("classifier", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        # Train
        self.pipeline.fit(X_train, y_train)
        logger.info("Training complete")

        # Evaluate
        results = {
            "train_samples": len(train_data),
            "train_accuracy": self.pipeline.score(X_train, y_train),
        }

        if test_data:
            X_test = [d["query"] for d in test_data]
            y_test = [d["label"] for d in test_data]

            y_pred = self.pipeline.predict(X_test)
            results["test_samples"] = len(test_data)
            results["test_accuracy"] = self.pipeline.score(X_test, y_test)
            results["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True
            )
            results["confusion_matrix"] = confusion_matrix(
                y_test, y_pred, labels=QUERY_TYPES
            ).tolist()

            # Print report
            print("\n" + "=" * 60)
            print("CLASSIFICATION REPORT")
            print("=" * 60)
            print(classification_report(y_test, y_pred))

        # Save model
        if save:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)
            logger.info(f"Saved model to {self.model_path}")
            results["model_path"] = str(self.model_path)

        return results

    def predict(self, query: str) -> str:
        """
        Predict the query type.

        Args:
            query: The research query string

        Returns:
            Predicted query type (factual, exploratory, etc.)

        Raises:
            ValueError: If no model is loaded
        """
        if self.pipeline is None:
            raise ValueError("No model loaded. Call train() first or provide a valid model path.")

        prediction = self.pipeline.predict([query])[0]
        return prediction

    def predict_proba(self, query: str) -> dict[str, float]:
        """
        Get prediction probabilities for each class.

        Args:
            query: The research query string

        Returns:
            Dictionary mapping query types to probabilities
        """
        if self.pipeline is None:
            raise ValueError("No model loaded. Call train() first or provide a valid model path.")

        probas = self.pipeline.predict_proba([query])[0]
        classes = self.pipeline.classes_

        return {cls: float(prob) for cls, prob in zip(classes, probas)}

    def predict_with_confidence(self, query: str) -> tuple[str, float]:
        """
        Predict query type with confidence score.

        Args:
            query: The research query string

        Returns:
            Tuple of (predicted_type, confidence)
        """
        probas = self.predict_proba(query)
        predicted = max(probas, key=probas.get)
        confidence = probas[predicted]
        return predicted, confidence

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.pipeline is not None


def train_classifier(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> QueryClassifier:
    """
    Convenience function to train the classifier from data files.

    Args:
        train_path: Path to training data JSON
        test_path: Path to test data JSON

    Returns:
        Trained QueryClassifier instance
    """
    data_dir = Path(__file__).parent.parent.parent / "data"

    train_path = train_path or data_dir / "query_classification_train.json"
    test_path = test_path or data_dir / "query_classification_test.json"

    # Load data
    with open(train_path) as f:
        train_data = json.load(f)

    test_data = None
    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)

    # Train
    classifier = QueryClassifier()
    results = classifier.train(train_data, test_data)

    print(f"\nTraining accuracy: {results['train_accuracy']:.2%}")
    if test_data:
        print(f"Test accuracy: {results['test_accuracy']:.2%}")

    return classifier


# Singleton instance for use in the agent
_classifier_instance: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    """Get the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Train the classifier when run directly
    logging.basicConfig(level=logging.INFO)
    
    print("Training query classifier...")
    classifier = train_classifier()
    
    # Test some examples
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    test_queries = [
        "What is the capital of France?",
        "How does machine learning work?",
        "Python vs JavaScript which is better?",
        "Why did the stock market crash?",
        "What are the latest developments in AI?",
    ]
    
    for query in test_queries:
        pred, conf = classifier.predict_with_confidence(query)
        print(f"\nQuery: {query}")
        print(f"Predicted: {pred} (confidence: {conf:.2%})")
