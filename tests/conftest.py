"""Pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys",
    )


@pytest.fixture
def sample_query():
    """Sample research query for testing."""
    return "What are the main differences between Python and JavaScript?"


@pytest.fixture
def sample_state():
    """Sample research state for testing."""
    from src.agent.state import QueryComplexity, QueryType, ResearchState

    return ResearchState(
        original_query="test query",
        query_type=QueryType.COMPARATIVE,
        query_complexity=QueryComplexity.MODERATE,
        sub_queries=["Python features", "JavaScript features", "Python vs JavaScript"],
    )
