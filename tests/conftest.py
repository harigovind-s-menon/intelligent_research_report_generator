"""Shared pytest fixtures and configuration."""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure we're not accidentally hitting production
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    yield


@pytest.fixture
def sample_sources():
    """Sample sources for testing."""
    return [
        {
            "url": "https://example.com/ml",
            "title": "Machine Learning Guide",
            "snippet": "Machine learning is a subset of AI that enables systems to learn from data.",
            "content": "Machine learning (ML) is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed.",
        },
        {
            "url": "https://example.com/ai",
            "title": "AI Overview",
            "snippet": "Artificial intelligence simulates human intelligence.",
            "content": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        },
    ]


@pytest.fixture
def sample_facts():
    """Sample facts for testing."""
    return [
        {
            "claim": "Machine learning is a subset of AI",
            "source_url": "https://example.com/ml",
            "confidence": 0.95,
        },
        {
            "claim": "AI simulates human intelligence",
            "source_url": "https://example.com/ai",
            "confidence": 0.9,
        },
    ]


@pytest.fixture
def sample_report():
    """Sample report for testing."""
    return """
# Research Report: Machine Learning

## Executive Summary
This report provides an overview of machine learning and its relationship to AI.

## Key Findings

### What is Machine Learning?
Machine learning is a subset of artificial intelligence that enables systems to learn from data.

### How it Works
ML algorithms identify patterns in data to make predictions.

## Contradictions or Uncertainties
No major contradictions were found in the sources.

## Conclusion
Machine learning is a powerful technology with broad applications.

## Sources
- [Machine Learning Guide](https://example.com/ml)
- [AI Overview](https://example.com/ai)
"""


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is machine learning and how does it relate to AI?"
