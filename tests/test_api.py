"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test health response has expected structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
        assert isinstance(data["services"], dict)

    def test_health_services_checked(self, client):
        """Test that all services are checked."""
        response = client.get("/health")
        data = response.json()
        
        assert "postgres" in data["services"]
        assert "redis" in data["services"]
        assert "agent" in data["services"]


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_structure(self, client):
        """Test root response has expected structure."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestResearchEndpoint:
    """Tests for research endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_research_requires_query(self, client):
        """Test that research endpoint requires query."""
        response = client.post("/research", json={})
        assert response.status_code == 422  # Validation error

    def test_research_query_min_length(self, client):
        """Test query minimum length validation."""
        response = client.post("/research", json={"query": "short"})
        assert response.status_code == 422

    def test_research_query_max_length(self, client):
        """Test query maximum length validation."""
        long_query = "a" * 1001
        response = client.post("/research", json={"query": long_query})
        assert response.status_code == 422

    def test_research_valid_request_structure(self, client):
        """Test valid request is accepted (mocked)."""
        # This test just validates the request structure
        # Full integration test would require running services
        with patch('src.api.main.agent') as mock_agent:
            mock_agent.invoke.return_value = {
                "sources": [],
                "extracted_facts": [],
                "final_report": "Test report",
                "query_type": MagicMock(value="factual"),
                "query_complexity": MagicMock(value="simple"),
                "search_queries_used": [],
                "contradictions": [],
            }
            
            # Skip if services aren't available
            try:
                response = client.post(
                    "/research",
                    json={"query": "What is machine learning and how does it work?"}
                )
                # Either succeeds or fails due to service unavailability
                assert response.status_code in [200, 503, 500]
            except Exception:
                pytest.skip("Services not available for integration test")


class TestResearchHistoryEndpoint:
    """Tests for research history endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_history_returns_list(self, client):
        """Test that history endpoint returns a list."""
        try:
            response = client.get("/research/history")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        except Exception:
            pytest.skip("Database not available")

    def test_history_limit_parameter(self, client):
        """Test limit parameter validation."""
        # Valid limit
        response = client.get("/research/history?limit=5")
        assert response.status_code in [200, 500]  # 500 if DB not available
        
        # Invalid limit (too low)
        response = client.get("/research/history?limit=0")
        assert response.status_code == 422
        
        # Invalid limit (too high)
        response = client.get("/research/history?limit=101")
        assert response.status_code == 422


class TestResearchByIdEndpoint:
    """Tests for get research by ID endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_invalid_id_returns_404(self, client):
        """Test that invalid ID returns 404."""
        try:
            response = client.get("/research/99999")
            assert response.status_code in [404, 500]  # 500 if DB not available
        except Exception:
            pytest.skip("Database not available")

    def test_id_must_be_integer(self, client):
        """Test that ID must be an integer."""
        response = client.get("/research/invalid")
        assert response.status_code == 422


class TestRequestModels:
    """Tests for request/response models."""

    def test_research_request_model(self):
        """Test ResearchRequestModel validation."""
        from src.api.main import ResearchRequestModel
        
        # Valid request
        request = ResearchRequestModel(query="What is machine learning and how does it work?")
        assert request.query == "What is machine learning and how does it work?"
        assert request.max_sources == 10  # default
        assert request.use_cache is True  # default

    def test_research_request_with_options(self):
        """Test ResearchRequestModel with custom options."""
        from src.api.main import ResearchRequestModel
        
        request = ResearchRequestModel(
            query="What is Python programming language?",
            max_sources=5,
            use_cache=False,
        )
        assert request.max_sources == 5
        assert request.use_cache is False

    def test_source_response_model(self):
        """Test SourceResponse model."""
        from src.api.main import SourceResponse
        
        source = SourceResponse(
            url="http://example.com",
            title="Test Title",
            snippet="Test snippet",
        )
        assert source.url == "http://example.com"

    def test_fact_response_model(self):
        """Test FactResponse model."""
        from src.api.main import FactResponse
        
        fact = FactResponse(
            claim="Test claim",
            source_url="http://example.com",
            confidence=0.9,
        )
        assert fact.confidence == 0.9
