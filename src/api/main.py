"""FastAPI application for the research report generator."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.graph import create_research_agent
from src.agent.state import ResearchState
from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global agent
    logger.info("Initializing research agent...")
    agent = create_research_agent()
    logger.info("Agent initialized successfully")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Intelligent Research Report Generator",
    description="Generate comprehensive research reports using AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""

    query: str = Field(..., min_length=10, max_length=1000, description="The research topic or question")
    max_sources: int = Field(default=10, ge=1, le=20, description="Maximum number of sources to retrieve")


class SourceResponse(BaseModel):
    """Source information in the response."""

    url: str
    title: str
    snippet: str | None


class FactResponse(BaseModel):
    """Extracted fact in the response."""

    claim: str
    source_url: str
    confidence: float


class ResearchResponse(BaseModel):
    """Response model for research endpoint."""

    query: str
    report: str
    sources: list[SourceResponse]
    facts: list[FactResponse]
    query_type: str | None
    complexity: str | None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
    )


@app.post("/research", response_model=ResearchResponse)
async def generate_research(request: ResearchRequest):
    """
    Generate a research report for the given query.

    This endpoint:
    1. Analyzes and decomposes the query
    2. Searches for relevant sources
    3. Extracts facts and identifies contradictions
    4. Generates a comprehensive report
    """
    global agent

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start_time = datetime.utcnow()
    logger.info(f"Research request: {request.query}")

    try:
        # Create initial state
        initial_state = ResearchState(original_query=request.query)

        # Run the agent
        result = agent.invoke(initial_state)

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Build response
        sources = [
            SourceResponse(
                url=s.url,
                title=s.title,
                snippet=s.snippet,
            )
            for s in result.get("sources", [])[:request.max_sources]
        ]

        facts = [
            FactResponse(
                claim=f.claim,
                source_url=f.source_url,
                confidence=f.confidence,
            )
            for f in result.get("extracted_facts", [])
        ]

        return ResearchResponse(
            query=request.query,
            report=result.get("final_report", ""),
            sources=sources,
            facts=facts,
            query_type=result.get("query_type").value if result.get("query_type") else None,
            complexity=result.get("query_complexity").value if result.get("query_complexity") else None,
            processing_time_seconds=processing_time,
        )

    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research generation failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Intelligent Research Report Generator",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
