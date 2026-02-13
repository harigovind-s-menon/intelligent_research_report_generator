"""Evaluation metrics for the research report generator.

This module provides metrics for:
- Retrieval quality (source relevance)
- Factual grounding (facts supported by sources)
- Report quality (structure, coherence, citations)
- Performance metrics (latency, token usage)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.models.embeddings import get_embedding_model, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    
    query: str
    num_sources: int
    avg_relevance_score: float  # Average semantic similarity to query
    min_relevance_score: float
    max_relevance_score: float
    relevance_scores: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "num_sources": self.num_sources,
            "avg_relevance_score": round(self.avg_relevance_score, 4),
            "min_relevance_score": round(self.min_relevance_score, 4),
            "max_relevance_score": round(self.max_relevance_score, 4),
        }


@dataclass
class FactGroundingMetrics:
    """Metrics for factual grounding."""
    
    num_facts: int
    num_grounded: int  # Facts with supporting evidence
    num_ungrounded: int  # Facts without clear support
    grounding_rate: float  # Percentage of grounded facts
    avg_confidence: float  # Average confidence of extracted facts
    fact_details: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "num_facts": self.num_facts,
            "num_grounded": self.num_grounded,
            "num_ungrounded": self.num_ungrounded,
            "grounding_rate": round(self.grounding_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass
class ReportQualityMetrics:
    """Metrics for report quality."""
    
    has_executive_summary: bool
    has_key_findings: bool
    has_conclusion: bool
    has_sources_section: bool
    num_sections: int
    num_citations: int
    word_count: int
    avg_section_length: float
    structure_score: float  # 0-1 based on presence of expected sections
    
    def to_dict(self) -> dict:
        return {
            "has_executive_summary": self.has_executive_summary,
            "has_key_findings": self.has_key_findings,
            "has_conclusion": self.has_conclusion,
            "has_sources_section": self.has_sources_section,
            "num_sections": self.num_sections,
            "num_citations": self.num_citations,
            "word_count": self.word_count,
            "avg_section_length": round(self.avg_section_length, 1),
            "structure_score": round(self.structure_score, 4),
        }


@dataclass
class PerformanceMetrics:
    """Metrics for performance."""
    
    total_time_seconds: float
    query_understanding_time: Optional[float] = None
    retrieval_time: Optional[float] = None
    analysis_time: Optional[float] = None
    generation_time: Optional[float] = None
    num_llm_calls: int = 0
    num_search_calls: int = 0
    
    def to_dict(self) -> dict:
        return {
            "total_time_seconds": round(self.total_time_seconds, 2),
            "query_understanding_time": round(self.query_understanding_time, 2) if self.query_understanding_time else None,
            "retrieval_time": round(self.retrieval_time, 2) if self.retrieval_time else None,
            "analysis_time": round(self.analysis_time, 2) if self.analysis_time else None,
            "generation_time": round(self.generation_time, 2) if self.generation_time else None,
            "num_llm_calls": self.num_llm_calls,
            "num_search_calls": self.num_search_calls,
        }


@dataclass 
class EvaluationResult:
    """Complete evaluation result."""
    
    retrieval: RetrievalMetrics
    grounding: FactGroundingMetrics
    report_quality: ReportQualityMetrics
    performance: PerformanceMetrics
    overall_score: float  # Weighted combination
    
    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "retrieval": self.retrieval.to_dict(),
            "grounding": self.grounding.to_dict(),
            "report_quality": self.report_quality.to_dict(),
            "performance": self.performance.to_dict(),
        }


class ResearchEvaluator:
    """Evaluates research report quality."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
    
    def evaluate_retrieval(
        self,
        query: str,
        sources: list[dict],
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality using semantic similarity.
        
        Args:
            query: Original research query
            sources: List of source dicts with 'title', 'content', 'snippet'
        
        Returns:
            RetrievalMetrics with relevance scores
        """
        if not sources:
            return RetrievalMetrics(
                query=query,
                num_sources=0,
                avg_relevance_score=0.0,
                min_relevance_score=0.0,
                max_relevance_score=0.0,
                relevance_scores=[],
            )
        
        # Embed the query
        query_embedding = self.embedding_model.embed(query)
        
        # Calculate relevance for each source
        relevance_scores = []
        for source in sources:
            # Combine title and snippet for comparison
            source_text = f"{source.get('title', '')}. {source.get('snippet', '') or source.get('content', '')[:500]}"
            source_embedding = self.embedding_model.embed(source_text)
            
            similarity = cosine_similarity(query_embedding, source_embedding)
            relevance_scores.append(float(similarity))
        
        return RetrievalMetrics(
            query=query,
            num_sources=len(sources),
            avg_relevance_score=float(np.mean(relevance_scores)),
            min_relevance_score=float(np.min(relevance_scores)),
            max_relevance_score=float(np.max(relevance_scores)),
            relevance_scores=relevance_scores,
        )
    
    def evaluate_grounding(
        self,
        facts: list[dict],
        sources: list[dict],
    ) -> FactGroundingMetrics:
        """
        Evaluate how well facts are grounded in sources.
        
        Args:
            facts: List of fact dicts with 'claim', 'source_url', 'confidence'
            sources: List of source dicts with 'url', 'content'
        
        Returns:
            FactGroundingMetrics with grounding analysis
        """
        if not facts:
            return FactGroundingMetrics(
                num_facts=0,
                num_grounded=0,
                num_ungrounded=0,
                grounding_rate=0.0,
                avg_confidence=0.0,
                fact_details=[],
            )
        
        # Build URL to content mapping
        source_content = {s.get('url'): s.get('content') or s.get('snippet', '') for s in sources}
        
        fact_details = []
        num_grounded = 0
        confidences = []
        
        for fact in facts:
            claim = fact.get('claim', '')
            source_url = fact.get('source_url', '')
            confidence = fact.get('confidence', 0.5)
            confidences.append(confidence)
            
            # Check if the fact's source exists
            source_exists = source_url in source_content
            
            # Check semantic similarity between claim and source content
            grounding_score = 0.0
            if source_exists and source_content[source_url]:
                claim_embedding = self.embedding_model.embed(claim)
                # Use chunks of source content for comparison
                content = source_content[source_url][:2000]
                content_embedding = self.embedding_model.embed(content)
                grounding_score = float(cosine_similarity(claim_embedding, content_embedding))
            
            # Consider grounded if similarity > 0.3 (relatively low threshold)
            is_grounded = grounding_score > 0.3
            if is_grounded:
                num_grounded += 1
            
            fact_details.append({
                "claim": claim[:100],
                "source_exists": source_exists,
                "grounding_score": round(grounding_score, 4),
                "is_grounded": is_grounded,
                "confidence": confidence,
            })
        
        return FactGroundingMetrics(
            num_facts=len(facts),
            num_grounded=num_grounded,
            num_ungrounded=len(facts) - num_grounded,
            grounding_rate=num_grounded / len(facts) if facts else 0.0,
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            fact_details=fact_details,
        )
    
    def evaluate_report_quality(
        self,
        report: str,
    ) -> ReportQualityMetrics:
        """
        Evaluate report structure and quality.
        
        Args:
            report: The generated report text (markdown)
        
        Returns:
            ReportQualityMetrics with structure analysis
        """
        if not report:
            return ReportQualityMetrics(
                has_executive_summary=False,
                has_key_findings=False,
                has_conclusion=False,
                has_sources_section=False,
                num_sections=0,
                num_citations=0,
                word_count=0,
                avg_section_length=0.0,
                structure_score=0.0,
            )
        
        report_lower = report.lower()
        
        # Check for expected sections
        has_executive_summary = 'executive summary' in report_lower or 'summary' in report_lower
        has_key_findings = 'key findings' in report_lower or 'findings' in report_lower
        has_conclusion = 'conclusion' in report_lower
        has_sources_section = 'sources' in report_lower or 'references' in report_lower
        
        # Count sections (markdown headers)
        sections = re.findall(r'^#{1,3}\s+.+$', report, re.MULTILINE)
        num_sections = len(sections)
        
        # Count citations (URLs or bracketed references)
        citations = re.findall(r'\[.*?\]\(.*?\)|\[\d+\]', report)
        num_citations = len(citations)
        
        # Word count
        word_count = len(report.split())
        
        # Average section length
        if num_sections > 0:
            avg_section_length = word_count / num_sections
        else:
            avg_section_length = word_count
        
        # Structure score (0-1)
        structure_components = [
            has_executive_summary,
            has_key_findings,
            has_conclusion,
            has_sources_section,
            num_sections >= 3,
            num_citations >= 1,
            word_count >= 200,
        ]
        structure_score = sum(structure_components) / len(structure_components)
        
        return ReportQualityMetrics(
            has_executive_summary=has_executive_summary,
            has_key_findings=has_key_findings,
            has_conclusion=has_conclusion,
            has_sources_section=has_sources_section,
            num_sections=num_sections,
            num_citations=num_citations,
            word_count=word_count,
            avg_section_length=avg_section_length,
            structure_score=structure_score,
        )
    
    def evaluate(
        self,
        query: str,
        sources: list[dict],
        facts: list[dict],
        report: str,
        processing_time: float,
        num_llm_calls: int = 3,
        num_search_calls: int = 5,
    ) -> EvaluationResult:
        """
        Run full evaluation on a research result.
        
        Args:
            query: Original query
            sources: Retrieved sources
            facts: Extracted facts
            report: Generated report
            processing_time: Total processing time in seconds
            num_llm_calls: Number of LLM API calls
            num_search_calls: Number of search API calls
        
        Returns:
            EvaluationResult with all metrics
        """
        logger.info("Running evaluation...")
        
        # Evaluate each component
        retrieval = self.evaluate_retrieval(query, sources)
        grounding = self.evaluate_grounding(facts, sources)
        report_quality = self.evaluate_report_quality(report)
        
        performance = PerformanceMetrics(
            total_time_seconds=processing_time,
            num_llm_calls=num_llm_calls,
            num_search_calls=num_search_calls,
        )
        
        # Calculate overall score (weighted average)
        # Weights: retrieval 30%, grounding 30%, report 40%
        overall_score = (
            0.30 * retrieval.avg_relevance_score +
            0.30 * grounding.grounding_rate +
            0.40 * report_quality.structure_score
        )
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.2%}")
        
        return EvaluationResult(
            retrieval=retrieval,
            grounding=grounding,
            report_quality=report_quality,
            performance=performance,
            overall_score=overall_score,
        )


# Singleton instance
_evaluator: Optional[ResearchEvaluator] = None


def get_evaluator() -> ResearchEvaluator:
    """Get the singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ResearchEvaluator()
    return _evaluator


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)
    
    evaluator = ResearchEvaluator()
    
    # Sample data
    query = "What are recent advances in AI?"
    
    sources = [
        {
            "url": "https://example.com/ai1",
            "title": "AI Breakthroughs 2026",
            "content": "Recent advances in artificial intelligence include improvements in large language models, computer vision, and robotics.",
            "snippet": "AI breakthroughs in LLMs and robotics",
        },
        {
            "url": "https://example.com/ai2", 
            "title": "Machine Learning Trends",
            "content": "Deep learning continues to advance with new architectures and training methods.",
            "snippet": "Deep learning advances",
        },
    ]
    
    facts = [
        {
            "claim": "Large language models have improved significantly",
            "source_url": "https://example.com/ai1",
            "confidence": 0.9,
        },
        {
            "claim": "Robotics is advancing rapidly",
            "source_url": "https://example.com/ai1",
            "confidence": 0.8,
        },
    ]
    
    report = """
# Research Report: Recent Advances in AI

## Executive Summary
AI has seen significant advances in 2026.

## Key Findings
1. Large language models improved
2. Robotics advancing

## Conclusion
AI continues to evolve rapidly.

## Sources
- [AI Breakthroughs](https://example.com/ai1)
"""
    
    result = evaluator.evaluate(
        query=query,
        sources=sources,
        facts=facts,
        report=report,
        processing_time=45.0,
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Score: {result.overall_score:.2%}")
    print(f"\nRetrieval:")
    print(f"  Sources: {result.retrieval.num_sources}")
    print(f"  Avg Relevance: {result.retrieval.avg_relevance_score:.2%}")
    print(f"\nGrounding:")
    print(f"  Facts: {result.grounding.num_facts}")
    print(f"  Grounded: {result.grounding.num_grounded}")
    print(f"  Rate: {result.grounding.grounding_rate:.2%}")
    print(f"\nReport Quality:")
    print(f"  Structure Score: {result.report_quality.structure_score:.2%}")
    print(f"  Word Count: {result.report_quality.word_count}")
    print(f"  Citations: {result.report_quality.num_citations}")
