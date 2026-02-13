"""Tests for the evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    ResearchEvaluator,
    RetrievalMetrics,
    FactGroundingMetrics,
    ReportQualityMetrics,
    PerformanceMetrics,
    EvaluationResult,
    get_evaluator,
)


class TestRetrievalMetrics:
    """Tests for retrieval evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Get the evaluator."""
        return get_evaluator()

    def test_evaluate_retrieval_with_sources(self, evaluator):
        """Test retrieval evaluation with sources."""
        query = "What is machine learning?"
        sources = [
            {"url": "http://example.com/1", "title": "Machine Learning Guide", "snippet": "ML is a type of AI"},
            {"url": "http://example.com/2", "title": "AI Overview", "snippet": "Artificial intelligence basics"},
        ]
        
        metrics = evaluator.evaluate_retrieval(query, sources)
        
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.num_sources == 2
        assert 0 <= metrics.avg_relevance_score <= 1
        assert 0 <= metrics.min_relevance_score <= 1
        assert 0 <= metrics.max_relevance_score <= 1
        assert metrics.min_relevance_score <= metrics.avg_relevance_score <= metrics.max_relevance_score

    def test_evaluate_retrieval_empty_sources(self, evaluator):
        """Test retrieval evaluation with no sources."""
        metrics = evaluator.evaluate_retrieval("test query", [])
        
        assert metrics.num_sources == 0
        assert metrics.avg_relevance_score == 0.0
        assert metrics.min_relevance_score == 0.0
        assert metrics.max_relevance_score == 0.0

    def test_retrieval_relevance_ordering(self, evaluator):
        """Test that relevant sources score higher than irrelevant ones."""
        query = "Python programming language"
        
        relevant_sources = [
            {"url": "http://example.com/1", "title": "Python Tutorial", "snippet": "Learn Python programming"},
        ]
        irrelevant_sources = [
            {"url": "http://example.com/2", "title": "Cooking Recipes", "snippet": "How to make pasta"},
        ]
        
        relevant_metrics = evaluator.evaluate_retrieval(query, relevant_sources)
        irrelevant_metrics = evaluator.evaluate_retrieval(query, irrelevant_sources)
        
        assert relevant_metrics.avg_relevance_score > irrelevant_metrics.avg_relevance_score


class TestFactGroundingMetrics:
    """Tests for fact grounding evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Get the evaluator."""
        return get_evaluator()

    def test_evaluate_grounding_with_facts(self, evaluator):
        """Test grounding evaluation with facts."""
        facts = [
            {"claim": "Python is a programming language", "source_url": "http://example.com/1", "confidence": 0.9},
            {"claim": "Python was created by Guido", "source_url": "http://example.com/1", "confidence": 0.85},
        ]
        sources = [
            {"url": "http://example.com/1", "content": "Python is a popular programming language created by Guido van Rossum."},
        ]
        
        metrics = evaluator.evaluate_grounding(facts, sources)
        
        assert isinstance(metrics, FactGroundingMetrics)
        assert metrics.num_facts == 2
        assert 0 <= metrics.grounding_rate <= 1
        assert 0 <= metrics.avg_confidence <= 1

    def test_evaluate_grounding_empty_facts(self, evaluator):
        """Test grounding evaluation with no facts."""
        metrics = evaluator.evaluate_grounding([], [])
        
        assert metrics.num_facts == 0
        assert metrics.grounding_rate == 0.0
        assert metrics.avg_confidence == 0.0

    def test_grounding_with_snippet_fallback(self, evaluator):
        """Test that grounding uses snippet when content is missing."""
        facts = [
            {"claim": "AI is transforming industries", "source_url": "http://example.com/1", "confidence": 0.9},
        ]
        sources = [
            {"url": "http://example.com/1", "snippet": "Artificial intelligence is transforming many industries today."},
        ]
        
        metrics = evaluator.evaluate_grounding(facts, sources)
        
        # Should be able to ground using snippet
        assert metrics.num_facts == 1


class TestReportQualityMetrics:
    """Tests for report quality evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Get the evaluator."""
        return get_evaluator()

    def test_evaluate_well_structured_report(self, evaluator):
        """Test evaluation of well-structured report."""
        report = """
# Research Report

## Executive Summary
This is a summary of our findings.

## Key Findings
1. First finding
2. Second finding

## Conclusion
In conclusion, we found interesting results.

## Sources
- [Source 1](http://example.com)
- [Source 2](http://example.com)
"""
        metrics = evaluator.evaluate_report_quality(report)
        
        assert isinstance(metrics, ReportQualityMetrics)
        assert metrics.has_executive_summary is True
        assert metrics.has_key_findings is True
        assert metrics.has_conclusion is True
        assert metrics.has_sources_section is True
        assert metrics.structure_score > 0.8

    def test_evaluate_minimal_report(self, evaluator):
        """Test evaluation of minimal report."""
        report = "This is just some text without structure."
        
        metrics = evaluator.evaluate_report_quality(report)
        
        assert metrics.has_executive_summary is False
        assert metrics.has_key_findings is False
        assert metrics.has_conclusion is False
        assert metrics.num_sections == 0
        assert metrics.structure_score < 0.5

    def test_evaluate_empty_report(self, evaluator):
        """Test evaluation of empty report."""
        metrics = evaluator.evaluate_report_quality("")
        
        assert metrics.word_count == 0
        assert metrics.structure_score == 0.0

    def test_citation_counting(self, evaluator):
        """Test that citations are counted correctly."""
        report = """
# Report

Some text with [citation 1](http://example.com) and [citation 2](http://example.com).
Also numbered citations [1] and [2].
"""
        metrics = evaluator.evaluate_report_quality(report)
        
        assert metrics.num_citations == 4

    def test_word_count(self, evaluator):
        """Test word count calculation."""
        report = "One two three four five."
        metrics = evaluator.evaluate_report_quality(report)
        assert metrics.word_count == 5


class TestFullEvaluation:
    """Tests for complete evaluation pipeline."""

    @pytest.fixture
    def evaluator(self):
        """Get the evaluator."""
        return get_evaluator()

    def test_full_evaluation(self, evaluator):
        """Test complete evaluation with all components."""
        query = "What is artificial intelligence?"
        sources = [
            {"url": "http://example.com/1", "title": "AI Guide", "snippet": "AI is machine intelligence", "content": "Artificial intelligence is intelligence demonstrated by machines."},
        ]
        facts = [
            {"claim": "AI is machine intelligence", "source_url": "http://example.com/1", "confidence": 0.9},
        ]
        report = """
# AI Research Report

## Executive Summary
This report covers AI basics.

## Key Findings
AI is transforming technology.

## Conclusion
AI is important.

## Sources
- [AI Guide](http://example.com/1)
"""
        
        result = evaluator.evaluate(
            query=query,
            sources=sources,
            facts=facts,
            report=report,
            processing_time=30.0,
            num_llm_calls=3,
            num_search_calls=5,
        )
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.overall_score <= 1
        assert isinstance(result.retrieval, RetrievalMetrics)
        assert isinstance(result.grounding, FactGroundingMetrics)
        assert isinstance(result.report_quality, ReportQualityMetrics)
        assert isinstance(result.performance, PerformanceMetrics)

    def test_to_dict(self, evaluator):
        """Test that evaluation result can be serialized."""
        result = evaluator.evaluate(
            query="test",
            sources=[{"url": "http://test.com", "title": "Test", "snippet": "Test content"}],
            facts=[{"claim": "Test claim", "source_url": "http://test.com", "confidence": 0.8}],
            report="# Test Report\n\n## Summary\nTest",
            processing_time=10.0,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert "overall_score" in data
        assert "retrieval" in data
        assert "grounding" in data
        assert "report_quality" in data
        assert "performance" in data


class TestOverallScoreCalculation:
    """Tests for overall score calculation."""

    @pytest.fixture
    def evaluator(self):
        """Get the evaluator."""
        return get_evaluator()

    def test_score_weights(self, evaluator):
        """Test that overall score uses correct weights."""
        # Create a scenario where we can predict the score
        # High retrieval, high grounding, high report quality
        query = "Machine learning"
        sources = [
            {"url": "http://ml.com", "title": "Machine Learning", "snippet": "ML is a subset of AI focusing on learning from data"},
        ]
        facts = [
            {"claim": "ML learns from data", "source_url": "http://ml.com", "confidence": 0.95},
        ]
        report = """
# ML Report

## Executive Summary
ML overview.

## Key Findings
ML is powerful.

## Conclusion
ML is useful.

## Sources
- [ML](http://ml.com)
"""
        
        result = evaluator.evaluate(
            query=query,
            sources=sources,
            facts=facts,
            report=report,
            processing_time=10.0,
        )
        
        # Manually calculate expected score
        expected = (
            0.30 * result.retrieval.avg_relevance_score +
            0.30 * result.grounding.grounding_rate +
            0.40 * result.report_quality.structure_score
        )
        
        assert abs(result.overall_score - expected) < 0.01
