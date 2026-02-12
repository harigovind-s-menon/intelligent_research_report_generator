#!/usr/bin/env python
"""Train the query classifier.

Usage:
    python -m src.models.train

This script:
1. Generates synthetic training data (if not exists)
2. Trains the scikit-learn classifier
3. Evaluates on test set
4. Saves the model to models/query_classifier.joblib
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.generate_training_data import main as generate_data
from src.models.query_classifier import train_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train the query classifier."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    train_path = data_dir / "query_classification_train.json"
    
    # Generate data if it doesn't exist
    if not train_path.exists():
        logger.info("Generating training data...")
        generate_data()
    else:
        logger.info("Training data already exists")
    
    # Train the classifier
    logger.info("Training classifier...")
    classifier = train_classifier()
    
    logger.info("Training complete!")
    
    # Interactive test
    print("\n" + "=" * 60)
    print("INTERACTIVE TEST")
    print("Type a query to classify (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
                
            pred, conf = classifier.predict_with_confidence(query)
            probas = classifier.predict_proba(query)
            
            print(f"Predicted: {pred} (confidence: {conf:.2%})")
            print("All probabilities:")
            for label, prob in sorted(probas.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"  {label:15} {prob:6.2%} {bar}")
                
        except KeyboardInterrupt:
            break
    
    print("\nDone!")


if __name__ == "__main__":
    main()
