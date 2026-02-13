"""LangGraph research agent definition.

This module defines the graph structure that orchestrates the research pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    analyzer_node,
    query_understanding_node,
    retriever_node,
    writer_node,
)
from src.agent.state import ResearchState

logger = logging.getLogger(__name__)


def create_research_agent() -> StateGraph:
    """
    Create the research agent graph.

    The graph flow:
    1. query_understanding: Analyze and decompose the query
    2. retriever: Search and retrieve sources
    3. analyzer: Extract facts and find contradictions
    4. writer: Generate the final report

    Returns:
        Compiled LangGraph that can be invoked with a query
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("writer", writer_node)

    # Define the flow
    workflow.set_entry_point("query_understanding")

    # Add edges (linear flow for Phase 1)
    workflow.add_edge("query_understanding", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "writer")
    workflow.add_edge("writer", END)

    # Compile the graph
    agent = workflow.compile()

    logger.info("Research agent compiled successfully")

    return agent


def run_research(query: str) -> dict:
    """
    Convenience function to run research on a query.

    Args:
        query: The research topic or question

    Returns:
        Dictionary containing the final state with report and metadata
    """
    agent = create_research_agent()

    initial_state = ResearchState(original_query=query)

    logger.info(f"Starting research: {query}")
    result = agent.invoke(initial_state)
    logger.info("Research complete")

    return result


# Allow running directly for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    query = sys.argv[1] if len(sys.argv) > 1 else "What are the latest developments in quantum computing?"

    result = run_research(query)

    print("\n" + "=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    print(result["final_report"])
    print("\n" + "=" * 80)
    print(f"Sources used: {len(result['sources'])}")
    print(f"Facts extracted: {len(result['extracted_facts'])}")
    print(f"Contradictions found: {len(result['contradictions'])}")
