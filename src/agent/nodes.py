"""Agent nodes for the research pipeline.

Each node is a function that:
1. Takes the current state
2. Performs some operation (LLM call, search, analysis)
3. Returns updates to the state
"""

import json
import logging

from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.agent.state import (
    Contradiction,
    Fact,
    QueryComplexity,
    QueryType,
    ReportSection,
    ResearchState,
    Source,
)
from src.config import get_settings
from src.db import vector_store
from src.models.query_classifier import get_classifier
from src.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Initialize clients
settings = get_settings()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
tavily_client = TavilyClient(api_key=settings.tavily_api_key)

# Confidence threshold for ML classifier (below this, use LLM)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5


# =============================================================================
# Node: Query Understanding
# =============================================================================


def query_understanding_node(state: ResearchState) -> dict:
    """
    Analyze the input query to understand its type, complexity, and create a research plan.

    This node:
    - Classifies the query type using ML classifier (fast, cheap)
    - Falls back to LLM if confidence is low
    - Uses LLM for complexity assessment and query decomposition
    - Creates a research plan
    """
    logger.info(f"Query understanding for: {state.original_query}")

    # Step 1: Try ML classifier first
    query_type = None
    classifier_used = False
    
    try:
        classifier = get_classifier()
        if classifier.is_loaded:
            predicted_type, confidence = classifier.predict_with_confidence(state.original_query)
            logger.info(f"ML classifier: {predicted_type} (confidence: {confidence:.2%})")
            
            if confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                query_type = QueryType(predicted_type)
                classifier_used = True
                logger.info(f"Using ML classification: {query_type.value}")
            else:
                logger.info(f"Confidence too low ({confidence:.2%}), falling back to LLM")
    except Exception as e:
        logger.warning(f"ML classifier failed: {e}, falling back to LLM")

    # Step 2: Use LLM for complexity, sub-queries, and optionally query_type
    if classifier_used:
        # Simpler prompt - just need complexity and decomposition
        prompt = f"""Analyze this research query and provide a structured analysis.

Query: {state.original_query}
Query Type (already classified): {query_type.value}

Respond with valid JSON only:
{{
    "complexity": "simple|moderate|complex",
    "sub_queries": ["list", "of", "sub-questions", "to", "research"],
    "research_plan": {{
        "approach": "description of how to research this",
        "key_aspects": ["aspect1", "aspect2"],
        "expected_sources": ["type of sources to look for"]
    }}
}}

Complexity guidelines:
- simple: Can be answered with 1-2 searches
- moderate: Needs 3-5 searches and synthesis
- complex: Needs deep research, multiple perspectives"""
    else:
        # Full prompt - need everything from LLM
        prompt = f"""Analyze this research query and provide a structured analysis.

Query: {state.original_query}

Respond with valid JSON only:
{{
    "query_type": "factual|exploratory|comparative|analytical|current_events",
    "complexity": "simple|moderate|complex",
    "sub_queries": ["list", "of", "sub-questions", "to", "research"],
    "research_plan": {{
        "approach": "description of how to research this",
        "key_aspects": ["aspect1", "aspect2"],
        "expected_sources": ["type of sources to look for"]
    }}
}}

Guidelines:
- factual: Simple fact lookup (e.g., "What year was X founded?")
- exploratory: Open-ended research (e.g., "What are the implications of X?")
- comparative: Compare multiple things (e.g., "X vs Y")
- analytical: Deep analysis (e.g., "Why did X happen?")
- current_events: Recent news (e.g., "Latest developments in X")

- simple: Can be answered with 1-2 searches
- moderate: Needs 3-5 searches and synthesis
- complex: Needs deep research, multiple perspectives"""

    response = llm.invoke(prompt)

    try:
        # Parse the JSON response
        content = response.content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
        # Ensure at least 1 sub-query (use original query as fallback)
        sub_queries = analysis.get("sub_queries", [])
        if not sub_queries:
            sub_queries = [state.original_query]
            logger.info("No sub-queries from LLM, using original query")

        # Use ML classifier result if available, otherwise use LLM result
        final_query_type = query_type if classifier_used else QueryType(analysis["query_type"])

        return {
            "query_type": final_query_type,
            "query_complexity": QueryComplexity(analysis["complexity"]),
            "sub_queries": sub_queries,
            "research_plan": analysis["research_plan"],
            "current_step": "retrieval",
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Failed to parse query analysis: {e}")
        # Fallback to defaults
        return {
            "query_type": query_type if classifier_used else QueryType.EXPLORATORY,
            "query_complexity": QueryComplexity.MODERATE,
            "sub_queries": [state.original_query],
            "research_plan": {"approach": "general research", "key_aspects": [], "expected_sources": []},
            "current_step": "retrieval",
            "errors": state.errors + [f"Query analysis failed: {str(e)}"],
        }


# =============================================================================
# Node: Retriever
# =============================================================================


def retriever_node(state: ResearchState) -> dict:
    """
    Search for and retrieve relevant sources.

    This node:
    - Uses Tavily to search for each sub-query
    - Collects and deduplicates sources
    - Stores source content for later analysis
    """
    logger.info(f"Retrieving sources for {len(state.sub_queries)} sub-queries")
    # Get vector store for storing retrieved documents
    vector_store = get_vector_store()

    all_sources: list[Source] = []
    search_queries_used: list[str] = []
    seen_urls: set[str] = set()

    # Determine search depth based on complexity
    max_results_per_query = {
        QueryComplexity.SIMPLE: 3,
        QueryComplexity.MODERATE: 5,
        QueryComplexity.COMPLEX: 7,
    }.get(state.query_complexity, 5)

    for sub_query in state.sub_queries[:5]:  # Limit to 5 sub-queries
        try:
            logger.info(f"Searching: {sub_query}")
            search_queries_used.append(sub_query)

            results = tavily_client.search(
                query=sub_query,
                max_results=max_results_per_query,
                include_raw_content=True,
            )

            # Handle case where results is None or malformed
            if results is None:
                logger.warning(f"No results returned for '{sub_query}'")
                continue

            result_list = results.get("results") if isinstance(results, dict) else []
            if not result_list:
                logger.warning(f"Empty results for '{sub_query}'")
                continue

            for result in result_list:
                if not isinstance(result, dict):
                    continue
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    # Store in vector store for future semantic search
                    try:
                        vector_store.add_document(
                            url=url,
                            title=result.get("title", ""),
                            content=result.get("content", "") or result.get("raw_content", ""),
                            snippet=result.get("content", "")[:500] if result.get("content") else None,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store document in vector store: {e}")
                    # Safely get content with fallbacks
                    raw_content = result.get("raw_content") or ""
                    content = result.get("content") or ""
                    all_sources.append(
                        Source(
                            url=url,
                            title=result.get("title", ""),
                            content=(raw_content or content)[:5000],  # Limit content size
                            snippet=content[:500] if content else None,
                            relevance_score=result.get("score"),
                        )
                    )

        except Exception as e:
            logger.error(f"Search failed for '{sub_query}': {e}")

    logger.info(f"Retrieved {len(all_sources)} unique sources")

    return {
        "sources": all_sources,
        "search_queries_used": search_queries_used,
        "current_step": "analysis",
    }


# =============================================================================
# Node: Analyzer
# =============================================================================


def analyzer_node(state: ResearchState) -> dict:
    """
    Analyze retrieved sources to extract facts and identify contradictions.

    This node:
    - Extracts key facts from each source
    - Identifies contradictions between sources
    - Clusters sources by topic/aspect
    """
    logger.info(f"Analyzing {len(state.sources)} sources")

    if not state.sources:
        return {
            "extracted_facts": [],
            "contradictions": [],
            "current_step": "generation",
            "errors": state.errors + ["No sources to analyze"],
        }

    # Prepare source summaries for the LLM
    source_summaries = []
    for i, source in enumerate(state.sources[:10]):  # Limit to 10 sources
        source_summaries.append(f"[Source {i+1}] {source.title}\nURL: {source.url}\nContent: {source.snippet}")

    sources_text = "\n\n".join(source_summaries)

    prompt = f"""Analyze these sources about: {state.original_query}

{sources_text}

Extract key facts and identify any contradictions. Respond with valid JSON only:
{{
    "facts": [
        {{
            "claim": "the factual claim",
            "source_url": "url of the source",
            "confidence": 0.9,
            "supporting_quotes": ["relevant quote from source"]
        }}
    ],
    "contradictions": [
        {{
            "claim_a": "first claim",
            "source_a": "url",
            "claim_b": "contradicting claim",
            "source_b": "url",
            "description": "explanation of the contradiction"
        }}
    ],
    "topic_clusters": {{
        "topic name": ["url1", "url2"]
    }}
}}

Focus on:
- Extracting 5-10 most important facts
- Identifying any contradictions or disagreements
- Grouping sources by subtopic"""

    response = llm.invoke(prompt)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)

        facts = [
            Fact(
                claim=f["claim"],
                source_url=f["source_url"],
                confidence=f.get("confidence", 0.8),
                supporting_quotes=f.get("supporting_quotes", []),
            )
            for f in analysis.get("facts", [])
        ]

        contradictions = [
            Contradiction(
                claim_a=c["claim_a"],
                source_a=c["source_a"],
                claim_b=c["claim_b"],
                source_b=c["source_b"],
                description=c["description"],
            )
            for c in analysis.get("contradictions", [])
        ]

        return {
            "extracted_facts": facts,
            "contradictions": contradictions,
            "cluster_labels": analysis.get("topic_clusters", {}),
            "current_step": "generation",
        }

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse analysis: {e}")
        return {
            "extracted_facts": [],
            "contradictions": [],
            "current_step": "generation",
            "errors": state.errors + [f"Analysis failed: {str(e)}"],
        }


# =============================================================================
# Node: Writer
# =============================================================================


def writer_node(state: ResearchState) -> dict:
    """
    Generate the final research report.

    This node:
    - Synthesizes all gathered information
    - Structures the report with clear sections
    - Includes proper citations
    """
    logger.info("Generating report")

    # Prepare context for the writer
    facts_text = "\n".join([f"- {f.claim} (confidence: {f.confidence})" for f in state.extracted_facts[:15]])

    contradictions_text = ""
    if state.contradictions:
        contradictions_text = "\n\nContradictions found:\n" + "\n".join(
            [f"- {c.description}" for c in state.contradictions]
        )

    sources_text = "\n".join([f"- [{s.title}]({s.url})" for s in state.sources[:10]])

    prompt = f"""Write a comprehensive research report on: {state.original_query}

Based on the following extracted facts:
{facts_text}
{contradictions_text}

Available sources:
{sources_text}

Write a well-structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (main body, organized by topic)
3. Contradictions or Uncertainties (if any)
4. Conclusion
5. Sources

Guidelines:
- Be objective and balanced
- Cite sources using [Source Title] format
- Acknowledge uncertainties
- Keep it concise but comprehensive (500-800 words)"""

    response = llm.invoke(prompt)
    report = response.content

    # Create structured sections (simplified for Phase 1)
    sections = [
        ReportSection(
            title="Research Report",
            content=report,
            sources=[s.url for s in state.sources[:10]],
        )
    ]

    return {
        "report_sections": sections,
        "final_report": report,
        "current_step": "complete",
    }


# =============================================================================
# Node: Should Continue (Conditional Edge)
# =============================================================================


def should_continue(state: ResearchState) -> str:
    """
    Determine if the agent should continue iterating or finish.

    Returns the name of the next node to execute.
    """
    # Check for errors that should stop execution
    if len(state.errors) > 3:
        logger.warning(f"Too many errors, stopping: {state.errors}")
        return "end"

    # Check iteration limit
    if state.iteration_count >= state.max_iterations:
        logger.info("Max iterations reached")
        return "end"

    # Route based on current step
    step_routing = {
        "start": "query_understanding",
        "retrieval": "retriever",
        "analysis": "analyzer",
        "generation": "writer",
        "complete": "end",
    }

    return step_routing.get(state.current_step, "end")