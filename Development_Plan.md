# Development Journal: Intelligent Research Report Generator

## Project Overview

A production-ready system that generates comprehensive, well-sourced research reports using multi-agent orchestration, traditional ML, and LLM capabilities. Built to demonstrate Staff AI Engineer competencies.

### Target Skills (from JD)

| Requirement | How This Project Addresses It |
|-------------|------------------------------|
| Agentic workflows (LangGraph) | Multi-step agent with routing, retrieval, reasoning, generation |
| Traditional ML/NLP | Query classification, document clustering, embeddings |
| LLM prompting & fine-tuning | Multiple prompt strategies, optional fine-tuning for summarization |
| Evaluation pipelines | Retrieval quality, factual accuracy, coherence metrics, human feedback |
| Data-to-text generation | Structured data (sources, facts) → narrative report |
| Production deployment | AWS (ECS, RDS, S3), monitoring, CI/CD |

---

## Phase 1: Foundation ✅

**Status:** Complete  
**Dates:** Feb 5, 2026

### Goals
- Set up project structure and development environment
- Implement basic LangGraph agent with linear flow
- Integrate Tavily for web search
- Create FastAPI endpoint
- Validate end-to-end pipeline works

### What Was Built

#### Project Structure
```
intelligent-research-report-generator/
├── src/
│   ├── agent/
│   │   ├── state.py      # ResearchState dataclass, enums, Pydantic models
│   │   ├── nodes.py      # 4 LangGraph nodes (query_understanding, retriever, analyzer, writer)
│   │   └── graph.py      # LangGraph workflow definition
│   ├── api/
│   │   └── main.py       # FastAPI app with /research endpoint
│   ├── models/           # (placeholder for Phase 2)
│   ├── evaluation/       # (placeholder for Phase 3)
│   ├── db/               # (placeholder for Phase 2)
│   ├── utils/
│   └── config.py         # pydantic-settings configuration
├── tests/
├── docker-compose.yml    # PostgreSQL + Redis for local dev
├── requirements.txt
├── run.py                # CLI test script
└── .env                  # API keys (not committed)
```

#### Agent Architecture (Phase 1)
```
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Query           │────▶│  Retriever  │────▶│  Analyzer   │────▶│   Writer    │
│ Understanding   │     │             │     │             │     │             │
└─────────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │                      │                   │                   │
        ▼                      ▼                   ▼                   ▼
  - Classify type        - Tavily search     - Extract facts     - Generate report
  - Assess complexity    - Collect sources   - Find contradictions - Add citations
  - Decompose query      - Deduplicate       - Cluster by topic
  - Create plan
```

#### Key Components

**1. ResearchState (`src/agent/state.py`)**
- Central dataclass that flows through the graph
- Contains: query info, sources, facts, contradictions, report sections, quality scores
- Uses `QueryType` and `QueryComplexity` enums for classification

**2. Agent Nodes (`src/agent/nodes.py`)**
- `query_understanding_node`: LLM-based query analysis → type, complexity, sub-queries
- `retriever_node`: Tavily search for each sub-query, deduplication
- `analyzer_node`: LLM extracts facts, identifies contradictions
- `writer_node`: LLM generates structured report with citations

**3. Graph (`src/agent/graph.py`)**
- Linear flow: query_understanding → retriever → analyzer → writer → END
- Compiled with `StateGraph(ResearchState)`

**4. API (`src/api/main.py`)**
- FastAPI with `/research` POST endpoint
- Request/response models with Pydantic
- Health check endpoint

### Configuration

**Environment Variables (`.env`)**
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
HUGGINGFACE_API_KEY=hf_...
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research-report-generator
DATABASE_URL=postgresql://postgres:localdev@localhost:5432/research_reports
REDIS_URL=redis://localhost:6379
S3_BUCKET_NAME=intelligent-research-report-generator
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-north-1
```

### Test Results

**First successful run:**
```
Query: "What are the latest developments in quantum computing?"
Query Type: CURRENT_EVENTS
Complexity: MODERATE
Sources Retrieved: 13
Facts Extracted: 7
Contradictions Found: 0
```

### Issues & Fixes

1. **Config variable naming mismatch**
   - Problem: Code expected `huggingface_token`, env had `HUGGINGFACE_API_KEY`
   - Fix: Updated `src/config.py` to match env variable names

2. **Tavily search failures**
   - Problem: `'NoneType' object is not subscriptable` on some searches
   - Fix: Added defensive checks for None/malformed responses in `retriever_node`

### Commits
- Initial project scaffolding
- LangGraph agent implementation
- FastAPI endpoint
- Config fixes for env variable naming
- Tavily error handling improvement

---

## Technical Decisions Log

### Decision 1: LangGraph vs. Custom Agent
**Date:** Feb 5, 2026  
**Decision:** Use LangGraph  
**Rationale:** 
- Explicitly required in JD
- Built-in state management and checkpointing
- Good debugging with LangSmith integration
- Industry standard for agent orchestration

### Decision 2: pgvector vs. Pinecone/FAISS
**Date:** Feb 5, 2026  
**Decision:** pgvector for production, FAISS for local dev  
**Rationale:**
- pgvector integrates with existing PostgreSQL (simpler ops)
- No additional service to manage
- Sufficient performance for our scale
- FAISS for fast local iteration without Docker

### Decision 3: OpenAI vs. Claude for LLM
**Date:** Feb 5, 2026  
**Decision:** OpenAI (GPT-4o-mini)  
**Rationale:**
- Existing OpenAI credits
- gpt-4o-mini is cost-effective for development
- Can swap to Claude later via LangChain abstraction, if needed

### Decision 4: Evaluation-First Approach
**Date:** Feb 5, 2026  
**Decision:** Build evaluation pipeline before optimization  
**Rationale:**
- Can't improve what you can't measure
- Required for meaningful A/B testing

---

## Resources

### Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [pgvector](https://github.com/pgvector/pgvector)
- [Tavily API](https://docs.tavily.com/)

### Tutorials
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Fine-tuning with Trainer](https://huggingface.co/docs/transformers/training)
- [RAGAS for RAG Evaluation](https://docs.ragas.io/)

---

## Changelog

### 2026-02-05
- Initial project setup
- Phase 1 complete: basic agent working end-to-end
- First successful research query executed
- Fixed config variable naming issues
- Improved Tavily error handling
