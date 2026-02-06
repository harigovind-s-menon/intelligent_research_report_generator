"""State schema for the research agent.

This defines the data structure that flows through the LangGraph agent.
Each node can read from and write to this state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class QueryType(str, Enum):
    """Classification of query types."""

    FACTUAL = "factual"  # Simple fact lookup
    EXPLORATORY = "exploratory"  # Open-ended research
    COMPARATIVE = "comparative"  # Compare multiple things
    ANALYTICAL = "analytical"  # Deep analysis required
    CURRENT_EVENTS = "current_events"  # Recent news/events


class QueryComplexity(str, Enum):
    """Complexity level of the query."""

    SIMPLE = "simple"  # Single search, direct answer
    MODERATE = "moderate"  # Multiple searches, synthesis needed
    COMPLEX = "complex"  # Deep research, multiple perspectives


class Source(BaseModel):
    """A retrieved source document."""

    url: str
    title: str
    content: str
    snippet: str | None = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float | None = None

    class Config:
        arbitrary_types_allowed = True


class Fact(BaseModel):
    """An extracted fact with source attribution."""

    claim: str
    source_url: str
    confidence: float  # 0-1
    supporting_quotes: list[str] = field(default_factory=list)


class Contradiction(BaseModel):
    """A detected contradiction between sources."""

    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    description: str


class ReportSection(BaseModel):
    """A section of the generated report."""

    title: str
    content: str
    sources: list[str] = field(default_factory=list)


class QualityScores(BaseModel):
    """Quality metrics for the generated report."""

    relevance: float | None = None  # How relevant to original query
    coherence: float | None = None  # How well-structured and readable
    factual_accuracy: float | None = None  # Based on source verification
    source_diversity: float | None = None  # Variety of sources used
    completeness: float | None = None  # Coverage of the topic


# Use dataclass for the state since LangGraph works well with it
@dataclass
class ResearchState:
    """
    The state that flows through the research agent graph.

    This is the central data structure - each node reads what it needs
    and writes its outputs back to the state.
    """

    # Input
    original_query: str = ""

    # Query Understanding (filled by query_understanding node)
    query_type: QueryType | None = None
    query_complexity: QueryComplexity | None = None
    sub_queries: list[str] = field(default_factory=list)
    research_plan: dict = field(default_factory=dict)

    # Retrieval (filled by retriever node)
    sources: list[Source] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)

    # Analysis (filled by analyzer node)
    extracted_facts: list[Fact] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    cluster_labels: dict[str, list[str]] = field(default_factory=dict)  # topic -> source urls

    # Generation (filled by writer node)
    report_sections: list[ReportSection] = field(default_factory=list)
    final_report: str = ""

    # Evaluation (filled by evaluation nodes)
    quality_scores: QualityScores = field(default_factory=QualityScores)

    # Control flow
    current_step: str = "start"
    iteration_count: int = 0
    max_iterations: int = 3
    errors: list[str] = field(default_factory=list)

    # Messages for LLM interactions (using LangGraph's message handling)
    messages: Annotated[list, add_messages] = field(default_factory=list)
