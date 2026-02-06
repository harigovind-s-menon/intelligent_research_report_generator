"""Tests for the research agent."""

import pytest

from src.agent.state import QueryComplexity, QueryType, ResearchState


class TestResearchState:
    """Test the ResearchState dataclass."""

    def test_default_state(self):
        """Test creating a state with defaults."""
        state = ResearchState(original_query="test query")
        assert state.original_query == "test query"
        assert state.query_type is None
        assert state.sources == []
        assert state.current_step == "start"

    def test_state_with_values(self):
        """Test creating a state with specific values."""
        state = ResearchState(
            original_query="test query",
            query_type=QueryType.FACTUAL,
            query_complexity=QueryComplexity.SIMPLE,
        )
        assert state.query_type == QueryType.FACTUAL
        assert state.query_complexity == QueryComplexity.SIMPLE


class TestQueryTypes:
    """Test query type enums."""

    def test_query_types(self):
        """Test all query types exist."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.EXPLORATORY.value == "exploratory"
        assert QueryType.COMPARATIVE.value == "comparative"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.CURRENT_EVENTS.value == "current_events"

    def test_complexity_levels(self):
        """Test all complexity levels exist."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"


# Integration tests (require API keys - skip in CI without keys)
@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Integration tests require --run-integration flag",
)
class TestAgentIntegration:
    """Integration tests for the full agent."""

    def test_simple_query(self):
        """Test a simple factual query."""
        from src.agent.graph import run_research

        result = run_research("What is the capital of France?")
        assert result["final_report"]
        assert len(result["sources"]) > 0
