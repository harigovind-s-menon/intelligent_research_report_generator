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
**Dates:** Feb 5-6, 2026

### Goals
- Set up project structure and development environment
- Implement basic LangGraph agent with linear flow
- Integrate Tavily for web search
- Create FastAPI endpoint
- Validate end-to-end pipeline works
- Add database logging and caching
- Build frontend for visualization

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
│   │   └── main.py       # FastAPI app with /research endpoint + history + caching
│   ├── db/
│   │   ├── models.py     # SQLAlchemy models (ResearchRequest)
│   │   └── cache.py      # Redis caching utilities
│   ├── models/           # (placeholder)
│   ├── evaluation/       # (placeholder)
│   ├── utils/
│   └── config.py         # pydantic-settings configuration
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── requirements.txt  # Frontend dependencies
├── tests/
├── docker-compose.yml    # PostgreSQL + Redis for local dev
├── requirements.txt
├── run.py                # CLI test script
└── .env                  # API keys (not committed)
```

#### Agent Architecture
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

#### Infrastructure Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   /health   │  │  /research  │  │  /research/history      │ │
│  │             │  │             │  │  /research/{id}         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────┐    ┌─────────────┐        ┌─────────────┐
│   Redis     │    │  LangGraph  │        │ PostgreSQL  │
│   Cache     │    │   Agent     │        │  (Logging)  │
└─────────────┘    └─────────────┘        └─────────────┘
```

#### Frontend (Streamlit)
```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ New Research│  │   History   │  │     View Report         │ │
│  │    Tab      │  │    Tab      │  │        Tab              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│  Sidebar: API URL config, Health status (postgres, redis, agent)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    FastAPI Backend (:8000)
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
- FastAPI with full CRUD endpoints
- Request/response models with Pydantic
- Database logging and Redis caching

**5. Database (`src/db/models.py`)**
- SQLAlchemy model for `ResearchRequest`
- Stores query, results, metadata, and timing

**6. Cache (`src/db/cache.py`)**
- Redis-based caching with 1-hour TTL
- Cache key: SHA256 hash of normalized query

**7. Frontend (`frontend/app.py`)**
- Streamlit app with three tabs: New Research, History, View Report
- Real-time health status in sidebar
- Configurable API URL for deployment flexibility

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check with service status (postgres, redis, agent) |
| `/research` | POST | Generate research report (with caching) |
| `/research/history` | GET | List recent research requests |
| `/research/{id}` | GET | Retrieve specific research result |

#### Database Schema

**research_requests table:**
```sql
id                      SERIAL PRIMARY KEY
query                   TEXT NOT NULL
query_type              VARCHAR(50)
complexity              VARCHAR(50)
report                  TEXT
sources_count           INTEGER
facts_count             INTEGER
contradictions_count    INTEGER
processing_time_seconds FLOAT
sources_json            JSONB
facts_json              JSONB
created_at              TIMESTAMP
completed_at            TIMESTAMP
```

#### Caching Strategy
- Cache key: SHA256 hash of normalized query (lowercase, trimmed)
- TTL: 1 hour (3600 seconds)
- Bypass: Set `use_cache: false` in request body
- Response includes `"cached": true/false` indicator

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

### Running the Application

**Terminal 1 — Backend:**
```bash
uvicorn src.api.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
# UI available at http://localhost:8501
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

**Infrastructure verified:**
- Health check shows all services healthy (postgres, redis, agent)
- Caching works — repeated queries return instantly with `"cached": true`
- History endpoint returns logged requests
- Retrieval by ID works
- Streamlit frontend connects to API and displays results

### Issues & Fixes

1. **Config variable naming mismatch**
   - Problem: Code expected `huggingface_token`, env had `HUGGINGFACE_API_KEY`
   - Fix: Updated `src/config.py` to match env variable names

2. **Tavily search failures**
   - Problem: `'NoneType' object is not subscriptable` on some searches
   - Fix: Added defensive checks for None/malformed responses in `retriever_node`

3. **Missing redis package**
   - Problem: `ModuleNotFoundError: No module named 'redis'`
   - Fix: Added `redis` to dependencies, installed via `uv pip install redis`

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

### Decision 5: Redis for Caching
**Date:** Feb 6, 2026  
**Decision:** Use Redis with 1-hour TTL for query caching  
**Rationale:**
- Identical queries are common (same user retrying, multiple users same topic)
- Saves API costs (OpenAI, Tavily)
- Reduces latency for cached queries to near-zero
- Simple key-value model fits our use case

### Decision 6: PostgreSQL for Request Logging
**Date:** Feb 6, 2026  
**Decision:** Log all requests to PostgreSQL with JSONB for flexible storage  
**Rationale:**
- Enables analytics on query patterns
- Supports debugging and auditing
- JSONB allows storing variable-length sources/facts without schema changes
- Can query historical results without re-running agent

### Decision 7: Streamlit for Frontend
**Date:** Feb 6, 2026  
**Decision:** Use Streamlit for the initial frontend  
**Rationale:**
- Fastest path to a working UI (Python-only, no JS build step)
- Good for demos and internal tools
- Deploys easily to AWS (ECS, App Runner, EC2)
- Can be replaced with React later if needed

---

## Resources

### Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [pgvector](https://github.com/pgvector/pgvector)
- [Tavily API](https://docs.tavily.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Tutorials
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Fine-tuning with Trainer](https://huggingface.co/docs/transformers/training)
- [RAGAS for RAG Evaluation](https://docs.ragas.io/)

---

## Changelog

### 2026-02-05
- Initial project setup
- Phase 1 agent complete: basic agent working end-to-end
- First successful research query executed
- Fixed config variable naming issues
- Improved Tavily error handling

### 2026-02-06
- Added PostgreSQL integration with SQLAlchemy
  - `ResearchRequest` model for logging all queries
  - Auto-creates tables on startup
- Added Redis caching layer
  - 1-hour TTL for identical queries
  - Cache hit returns instantly with `"cached": true`
  - `use_cache: false` option to bypass
- Enhanced health check endpoint
  - Shows status of Postgres, Redis, and agent
  - Returns "degraded" if any service unhealthy
- New API endpoints:
  - `GET /research/history` — view recent requests
  - `GET /research/{id}` — retrieve specific result by ID
- Added `redis` to dependencies
- Added Streamlit frontend (`frontend/app.py`)
  - Three tabs: New Research, History, View Report
  - Real-time health status in sidebar
  - Configurable API URL for deployment
  - Displays report, sources, and extracted facts
