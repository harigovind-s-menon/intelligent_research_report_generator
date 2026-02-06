"""Database connection and models."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import get_settings

settings = get_settings()

# Create engine
engine = create_engine(settings.database_url, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class ResearchRequest(Base):
    """Track research requests and their results."""

    __tablename__ = "research_requests"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False, index=True)
    query_type = Column(String(50))
    complexity = Column(String(50))
    
    # Results
    report = Column(Text)
    sources_count = Column(Integer)
    facts_count = Column(Integer)
    contradictions_count = Column(Integer)
    
    # Metadata
    processing_time_seconds = Column(Float)
    sources_json = Column(JSONB)  # Store full source details
    facts_json = Column(JSONB)    # Store extracted facts
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    def __repr__(self):
        return f"<ResearchRequest(id={self.id}, query='{self.query[:50]}...')>"


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
