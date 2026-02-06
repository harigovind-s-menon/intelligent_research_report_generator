"""FastAPI application for the research report generator."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agent.graph import create_research_agent
from src.agent.state import ResearchState
from src.config import get_settings
from src.db.cache import check_redis_health, get_cached_result, set_cached_result
from src.db.models import ResearchRequest, SessionLocal, get_db, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global agent instance
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global agent
    logger.info("Initializing application...")
    
    # Initialize database tables
    try:
        init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Initialize agent
    agent = create_research_agent()
    logger.info("Research agent initialized")
    
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


class ResearchRequestModel(BaseModel):
    """Request model for research endpoint."""

    query: str = Field(..., min_length=10, max_length=1000, description="The research topic or question")
    max_sources: int = Field(default=10, ge=1, le=20, description="Maximum number of sources to retrieve")
    use_cache: bool = Field(default=True, description="Whether to use cached results if available")


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
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    services: dict[str, str]


class RequestHistoryResponse(BaseModel):
    """Response for request history endpoint."""

    id: int
    query: str
    query_type: str | None
    complexity: str | None
    sources_count: int | None
    processing_time_seconds: float | None
    created_at: datetime


# =============================================================================
# Health Check
# =============================================================================


def check_postgres_health() -> bool:
    """Check if PostgreSQL is healthy."""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status."""
    postgres_healthy = check_postgres_health()
    redis_healthy = check_redis_health()
    
    services = {
        "postgres": "healthy" if postgres_healthy else "unhealthy",
        "redis": "healthy" if redis_healthy else "unhealthy",
        "agent": "healthy" if agent is not None else "unhealthy",
    }
    
    overall_status = "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        services=services,
    )


# =============================================================================
# Research Endpoints
# =============================================================================


@app.post("/research", response_model=ResearchResponse)
async def generate_research(
    request: ResearchRequestModel,
    db: Session = Depends(get_db),
):
    """
    Generate a research report for the given query.

    This endpoint:
    1. Checks cache for existing results
    2. Analyzes and decomposes the query
    3. Searches for relevant sources
    4. Extracts facts and identifies contradictions
    5. Generates a comprehensive report
    6. Caches and logs the result
    """
    global agent

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Check cache first
    if request.use_cache:
        cached = get_cached_result(request.query)
        if cached:
            return ResearchResponse(**cached, cached=True)

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

        response_data = {
            "query": request.query,
            "report": result.get("final_report", ""),
            "sources": sources,
            "facts": facts,
            "query_type": result.get("query_type").value if result.get("query_type") else None,
            "complexity": result.get("query_complexity").value if result.get("query_complexity") else None,
            "processing_time_seconds": processing_time,
        }

        # Log to database
        try:
            db_request = ResearchRequest(
                query=request.query,
                query_type=response_data["query_type"],
                complexity=response_data["complexity"],
                report=response_data["report"],
                sources_count=len(sources),
                facts_count=len(facts),
                contradictions_count=len(result.get("contradictions", [])),
                processing_time_seconds=processing_time,
                sources_json=[s.model_dump() for s in sources],
                facts_json=[f.model_dump() for f in facts],
                completed_at=datetime.utcnow(),
            )
            db.add(db_request)
            db.commit()
            logger.info(f"Logged request to database: id={db_request.id}")
        except Exception as e:
            logger.error(f"Failed to log to database: {e}")
            db.rollback()

        # Cache the result
        cache_data = {**response_data, "sources": [s.model_dump() for s in sources], "facts": [f.model_dump() for f in facts]}
        set_cached_result(request.query, cache_data)

        return ResearchResponse(**response_data, cached=False)

    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research generation failed: {str(e)}")


@app.get("/research/history", response_model=list[RequestHistoryResponse])
async def get_research_history(
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get recent research request history."""
    try:
        requests = (
            db.query(ResearchRequest)
            .order_by(ResearchRequest.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            RequestHistoryResponse(
                id=r.id,
                query=r.query,
                query_type=r.query_type,
                complexity=r.complexity,
                sources_count=r.sources_count,
                processing_time_seconds=r.processing_time_seconds,
                created_at=r.created_at,
            )
            for r in requests
        ]
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


@app.get("/research/{request_id}", response_model=ResearchResponse)
async def get_research_by_id(
    request_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific research result by ID."""
    try:
        result = db.query(ResearchRequest).filter(ResearchRequest.id == request_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="Research request not found")
        
        sources = [SourceResponse(**s) for s in (result.sources_json or [])]
        facts = [FactResponse(**f) for f in (result.facts_json or [])]
        
        return ResearchResponse(
            query=result.query,
            report=result.report or "",
            sources=sources,
            facts=facts,
            query_type=result.query_type,
            complexity=result.complexity,
            processing_time_seconds=result.processing_time_seconds or 0,
            cached=False,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch research: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch research")


# =============================================================================
# Root
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Intelligent Research Report Generator",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /research": "Generate a research report",
            "GET /research/history": "View recent requests",
            "GET /research/{id}": "Get a specific result",
        },
    }
