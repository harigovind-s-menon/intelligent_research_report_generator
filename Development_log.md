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

3. **Missing redis package**
   - Problem: `ModuleNotFoundError: No module named 'redis'`
   - Fix: Added `redis` to dependencies, installed via `uv pip install redis`

---

## Phase 2: Traditional ML ✅

**Status:** Complete  
**Dates:** Feb 12, 2026

### Goals
- Add scikit-learn query classifier to replace LLM-based classification
- Implement sentence embeddings for semantic similarity
- Set up pgvector for document storage and retrieval

### What Was Built

#### Query Classifier (`src/models/query_classifier.py`)

A scikit-learn pipeline that classifies queries into 5 types:
- `factual` — Simple fact lookup
- `exploratory` — Open-ended research
- `comparative` — Compare multiple things
- `analytical` — Deep analysis required
- `current_events` — Recent news/events

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│   TF-IDF    │────▶│  Logistic   │────▶ Query Type
│   Text      │     │ Vectorizer  │     │ Regression  │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Training Data:**
- Generated synthetically using templates + fill-in values
- 1000 training samples (200 per class)
- 250 test samples (50 per class)
- Templates cover various phrasings for each query type

**Model Performance:**
```
Training accuracy: 98.50%
Test accuracy:     95.20%
```

**Classification Report:**
```
                precision    recall  f1-score   support
       factual       0.94      0.98      0.96        50
   exploratory       0.94      0.92      0.93        50
   comparative       0.98      1.00      0.99        50
    analytical       0.92      0.88      0.90        50
current_events       0.98      0.98      0.98        50
```

#### Sentence Embeddings (`src/models/embeddings.py`)

Uses sentence-transformers for generating document embeddings.

**Model:** `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Fast inference on CPU
- Good balance of speed and quality

**Features:**
- Single text embedding
- Batch embedding for efficiency
- Cosine similarity computation
- Find most similar texts

#### Vector Store (`src/db/vector_store.py`)

PostgreSQL with pgvector extension for semantic search.

**Documents table:**
```sql
id          SERIAL PRIMARY KEY
url         VARCHAR(2048) UNIQUE
title       VARCHAR(512)
content     TEXT
snippet     TEXT
embedding   vector(384)
created_at  TIMESTAMP
updated_at  TIMESTAMP
```

**Features:**
- Store documents with embeddings
- Semantic similarity search
- Deduplication by URL
- IVFFlat index for fast search

#### Integration with Agent

**Query Understanding Node (Hybrid ML + LLM):**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Understanding Node                     │
│                                                                 │
│  1. ML Classifier (fast, free)                                  │
│     ├─ Confidence ≥ 50% → Use ML result                        │
│     └─ Confidence < 50% → Fall back to LLM                     │
│                                                                 │
│  2. LLM (still used for):                                       │
│     ├─ Complexity assessment (simple/moderate/complex)          │
│     ├─ Query decomposition (sub-queries)                        │
│     └─ Research plan generation                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Retriever Node (with Vector Store):**
```
┌─────────────────────────────────────────────────────────────────┐
│                       Retriever Node                            │
│                                                                 │
│  For each sub-query:                                            │
│  1. Search via Tavily API                                       │
│  2. Deduplicate by URL                                          │
│  3. Store in pgvector with embedding                            │
│                                                                 │
│  Documents accumulate across queries for future semantic search │
└─────────────────────────────────────────────────────────────────┘
```

#### Files Added

```
src/models/
├── __init__.py                 # Package exports
├── generate_training_data.py   # Synthetic data generation
├── query_classifier.py         # Classifier model
├── train.py                    # Training script
└── embeddings.py               # Sentence embeddings

src/db/
├── vector_store.py             # pgvector document store

data/
├── query_classification_train.json  # 1000 training samples
└── query_classification_test.json   # 250 test samples

models/
└── query_classifier.joblib     # Trained model
```

### Test Results

**ML Classifier in Action:**
```
Query: "What is the capital of France?"
ML classifier: factual (confidence: 86.82%)
Using ML classification: factual

Query: "What are the latest developments in quantum computing?"
ML classifier: current_events (confidence: 66.23%)
Using ML classification: current_events

Query: "What are recent advances in AI?"
ML classifier: current_events (confidence: 60.67%)
Using ML classification: current_events
Sources Retrieved: 25
Documents stored in vector store: 28+
```

**Vector Store Search:**
```
Query: "quantum computing advances"
Result: Quantum Computing Breakthrough (similarity: 0.787)
```

### Issues & Fixes

1. **Deprecated scikit-learn parameter**
   - Problem: `LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`
   - Fix: Removed `multi_class="multinomial"` (default behavior in newer versions)

2. **Zero sub-queries from LLM**
   - Problem: Simple queries returned 0 sub-queries, leading to no sources
   - Fix: Added fallback to use original query if sub_queries is empty

3. **pgvector type casting**
   - Problem: `operator does not exist: vector <=> numeric[]`
   - Fix: Changed `::vector` to `CAST(:embedding AS vector)` to avoid SQLAlchemy parameter binding conflicts

---

## Phase 3: Evaluation Pipeline ✅

**Status:** Complete  
**Dates:** Feb 12, 2026

### Goals
- Build metrics to measure retrieval quality
- Evaluate factual grounding of extracted facts
- Assess report structure and quality
- Track performance metrics
- Integrate evaluation into API endpoints

### What Was Built

#### Evaluation Metrics (`src/evaluation/metrics.py`)

Four categories of metrics with an overall weighted score:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Evaluation Pipeline                        │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  Retrieval  │  │  Grounding  │  │   Report    │  │ Perf   │ │
│  │   (30%)     │  │   (30%)     │  │   (40%)     │  │        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
│        │                │                │               │      │
│        ▼                ▼                ▼               ▼      │
│   Semantic         Fact vs Source    Structure      Time,      │
│   Similarity       Verification      Analysis       API Calls  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      Overall Score (0-1)
```

#### Retrieval Metrics
Measures how relevant retrieved sources are to the original query.

| Metric | Description |
|--------|-------------|
| `num_sources` | Total sources retrieved |
| `avg_relevance_score` | Mean semantic similarity (query ↔ source) |
| `min_relevance_score` | Lowest relevance source |
| `max_relevance_score` | Highest relevance source |

**Method:** Embeds query and each source's title+snippet, computes cosine similarity.

#### Fact Grounding Metrics
Verifies that extracted facts are supported by their cited sources.

| Metric | Description |
|--------|-------------|
| `num_facts` | Total facts extracted |
| `num_grounded` | Facts with supporting evidence |
| `num_ungrounded` | Facts without clear support |
| `grounding_rate` | Percentage grounded (0-1) |
| `avg_confidence` | Mean confidence of facts |

**Method:** Embeds each fact's claim and its source content, considers grounded if similarity > 0.3.

#### Report Quality Metrics
Evaluates report structure and completeness.

| Metric | Description |
|--------|-------------|
| `has_executive_summary` | Contains summary section |
| `has_key_findings` | Contains findings section |
| `has_conclusion` | Contains conclusion |
| `has_sources_section` | Lists sources/references |
| `num_sections` | Total markdown sections |
| `num_citations` | Number of citations |
| `word_count` | Total words |
| `structure_score` | Overall structure (0-1) |

**Method:** Rule-based checks for expected sections and formatting.

#### Performance Metrics
Tracks operational performance.

| Metric | Description |
|--------|-------------|
| `total_time_seconds` | End-to-end processing time |
| `num_llm_calls` | LLM API calls made |
| `num_search_calls` | Search API calls made |

#### Overall Score Formula
```
overall_score = 0.30 × avg_relevance_score 
              + 0.30 × grounding_rate 
              + 0.40 × structure_score
```

Weights prioritize report quality (40%) as the final deliverable, with retrieval and grounding equally weighted (30% each).

#### API Integration

Evaluation runs automatically on every research request:

```python
# In POST /research response:
{
  "query": "...",
  "report": "...",
  "sources": [...],
  "facts": [...],
  "evaluation": {
    "overall_score": 0.91,
    "retrieval": {...},
    "grounding": {...},
    "report_quality": {...},
    "performance": {...}
  }
}
```

Also available when fetching historical results via `GET /research/{id}`.

#### Files Added

```
src/evaluation/
├── __init__.py    # Package exports
└── metrics.py     # Evaluation classes and functions
```

### Test Results

**Sample Evaluation Output:**
```
Query: "What are Machine Learning and Deep Learning?"

Overall Score: 91.27%

Retrieval:
  Sources: 10
  Avg Relevance: 70.89%
  Min Relevance: 62.91%
  Max Relevance: 77.89%

Grounding:
  Facts: 7
  Grounded: 7 (100%)
  Avg Confidence: 88.57%

Report Quality:
  Structure Score: 100%
  Has Executive Summary: ✓
  Has Key Findings: ✓
  Has Conclusion: ✓
  Has Sources: ✓
  Sections: 10
  Citations: 10
  Word Count: 716

Performance:
  Total Time: 69.75s
  LLM Calls: 3
  Search Calls: 5
```

### Issues & Fixes

1. **Evaluation returning null**
   - Problem: Evaluation code in wrong location, variables undefined
   - Fix: Added proper try/catch block with `evaluation_dict` variable

2. **Grounding rate always 0%**
   - Problem: Looking for `content` field but sources only have `snippet`
   - Fix: Changed to `s.get('content') or s.get('snippet', '')`

3. **Undefined eval_result in get_by_id**
   - Problem: `eval_result` referenced but never defined in endpoint
   - Fix: Added full evaluation block to `GET /research/{id}` endpoint

---

## Test Suite ✅

**Status:** Complete  
**Dates:** Feb 13, 2026

### Goals
- Add comprehensive unit tests for all components
- Set up pytest configuration
- Enable integration testing with flag
- Achieve reasonable code coverage

### What Was Built

#### Test Structure
```
tests/
├── conftest.py              # Shared fixtures, pytest configuration
├── test_query_classifier.py # 20 tests for ML classifier
├── test_embeddings.py       # Tests for embeddings and similarity
├── test_evaluation.py       # Tests for evaluation metrics
├── test_api.py              # Tests for FastAPI endpoints
└── test_agent.py            # State tests + integration test

pytest.ini                   # Pytest configuration
```

#### Test Coverage

| Module | Coverage | Tests |
|--------|----------|-------|
| `state.py` | 100% | Dataclass, enums |
| `config.py` | 100% | Settings loading |
| `evaluation/metrics.py` | 86% | All metric types |
| `embeddings.py` | 82% | Embedding, similarity |
| `api/main.py` | 61% | Endpoints, validation |
| `query_classifier.py` | 48% | Classification accuracy |
| **Overall** | **51%** | **73 tests** |

#### Test Categories

**Query Classifier Tests (`test_query_classifier.py`):**
- Model loading and initialization
- Prediction returns valid types
- Probability distribution validation
- Classification accuracy for all 5 query types
- Edge cases (empty, long queries)
- Error handling when model not loaded

**Embedding Tests (`test_embeddings.py`):**
- Model loading (384 dimensions)
- Single and batch embedding
- Cosine similarity calculations
- Similar vs dissimilar text detection
- Find most similar functionality
- Similarity matrix computation

**Evaluation Tests (`test_evaluation.py`):**
- Retrieval metrics calculation
- Fact grounding verification
- Report quality assessment
- Full evaluation pipeline
- Score weight verification
- Serialization to dict

**API Tests (`test_api.py`):**
- Health endpoint response
- Root endpoint structure
- Request validation (min/max length)
- History endpoint with pagination
- Get by ID error handling
- Request/response model validation

**Agent Tests (`test_agent.py`):**
- ResearchState creation
- Query type enums
- Complexity level enums
- Integration test (requires `--run-integration`)

#### Running Tests

```bash
# Run all unit tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_query_classifier.py

# Run integration tests (requires services)
pytest --run-integration

# Run only integration tests
pytest tests/test_agent.py::TestAgentIntegration --run-integration

# Skip slow tests
pytest -m "not slow"
```

#### Fixtures Available

| Fixture | Description |
|---------|-------------|
| `sample_sources` | List of source dicts with url, title, snippet, content |
| `sample_facts` | List of fact dicts with claim, source_url, confidence |
| `sample_report` | Well-structured markdown report |
| `sample_query` | Example research query |

### Test Results

```
========================== test session starts ==========================
collected 73 items

tests/test_query_classifier.py .................... [27%]
tests/test_embeddings.py .................... [55%]
tests/test_evaluation.py .................... [82%]
tests/test_api.py .............. [96%]
tests/test_agent.py ... [100%]

========================== 73 passed in 12.65s ==========================
```

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

### Decision 8: Hybrid ML + LLM Classification
**Date:** Feb 12, 2026  
**Decision:** Use ML classifier with LLM fallback for query type classification  
**Rationale:**
- ML classifier is faster and free (no API cost)
- 95% accuracy is sufficient for routing decisions
- 50% confidence threshold catches uncertain cases
- LLM provides fallback for edge cases and novel phrasings
- Reduces LLM token usage without sacrificing quality

### Decision 9: Synthetic Training Data
**Date:** Feb 12, 2026  
**Decision:** Generate synthetic training data using templates  
**Rationale:**
- Fast to create (vs. manual labeling)
- Covers diverse phrasings for each query type
- Reproducible (seeded random generation)
- Good enough for initial model (95% accuracy achieved)
- Can be augmented with real user queries later

### Decision 10: all-MiniLM-L6-v2 for Embeddings
**Date:** Feb 12, 2026  
**Decision:** Use all-MiniLM-L6-v2 sentence transformer model  
**Rationale:**
- 384 dimensions (compact, fast)
- Good quality for semantic similarity
- Fast inference on CPU
- Widely used and well-tested
- Can upgrade to all-mpnet-base-v2 (768 dims) if needed

### Decision 11: Store Retrieved Documents in pgvector
**Date:** Feb 12, 2026  
**Decision:** Automatically store all retrieved documents in vector store  
**Rationale:**
- Builds up a knowledge base over time
- Enables future semantic search across past research
- Deduplication by URL prevents redundant storage
- Foundation for RAG-style retrieval augmentation

### Decision 12: Semantic Similarity for Evaluation
**Date:** Feb 12, 2026  
**Decision:** Use embedding-based semantic similarity for retrieval and grounding metrics  
**Rationale:**
- More nuanced than keyword matching
- Captures meaning rather than exact words
- Reuses existing embedding infrastructure
- Provides continuous scores (0-1) rather than binary
- Consistent with modern NLP evaluation approaches

### Decision 13: Weighted Overall Score
**Date:** Feb 12, 2026  
**Decision:** Weight report quality (40%) higher than retrieval and grounding (30% each)  
**Rationale:**
- Report is the final deliverable users see
- Well-structured report can compensate for minor retrieval gaps
- Grounding ensures factual accuracy, critical for trust
- Balanced approach that values end output while maintaining quality standards

---

## Resources

### Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [pgvector](https://github.com/pgvector/pgvector)
- [Tavily API](https://docs.tavily.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [sentence-transformers](https://www.sbert.net/)

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

### 2026-02-12
- **Phase 2: Traditional ML — Complete**
- Built query classifier using scikit-learn
  - TF-IDF vectorizer + Logistic Regression pipeline
  - 98.5% training accuracy, 95.2% test accuracy
  - Synthetic training data generator (1000 samples)
- Integrated classifier into `query_understanding_node`
  - ML classifier runs first (fast, free)
  - Falls back to LLM if confidence < 50%
  - LLM still handles complexity and decomposition
- Added sentence embeddings (`src/models/embeddings.py`)
  - all-MiniLM-L6-v2 model (384 dimensions)
  - Single and batch embedding support
  - Cosine similarity utilities
- Added pgvector document store (`src/db/vector_store.py`)
  - Documents table with vector column
  - Semantic similarity search
  - IVFFlat index for performance
- Integrated vector store into retriever node
  - All retrieved documents automatically stored with embeddings
  - Deduplication by URL
  - Building knowledge base for future semantic search
- Fixed deprecated `multi_class` parameter in scikit-learn
- Fixed zero sub-queries fallback
- Fixed pgvector CAST syntax for SQLAlchemy compatibility

- **Phase 3: Evaluation Pipeline — Complete**
- Built evaluation metrics (`src/evaluation/metrics.py`)
  - RetrievalMetrics: semantic similarity between query and sources
  - FactGroundingMetrics: verify facts are supported by cited sources
  - ReportQualityMetrics: structure, sections, citations, word count
  - PerformanceMetrics: time, LLM calls, search calls
  - Overall weighted score (30% retrieval, 30% grounding, 40% report)
- Integrated evaluation into API
  - `POST /research` returns evaluation in response
  - `GET /research/{id}` also includes evaluation
- Fixed grounding metric to use snippet when content unavailable
- Fixed undefined variable in get_by_id endpoint
- Achieved 91%+ overall score on test queries

### 2026-02-13
- **Test Suite — Complete**
- Added comprehensive pytest test suite with 73 tests
- Test coverage: 51% overall
  - `state.py`: 100%
  - `config.py`: 100%
  - `evaluation/metrics.py`: 86%
  - `embeddings.py`: 82%
  - `api/main.py`: 61%
- Test files added:
  - `test_query_classifier.py`: 20 tests for ML classifier
  - `test_embeddings.py`: Tests for embedding generation and similarity
  - `test_evaluation.py`: Tests for all evaluation metrics
  - `test_api.py`: Tests for FastAPI endpoints
  - `test_agent.py`: State tests + integration test
  - `conftest.py`: Shared fixtures and pytest configuration
- Added `pytest.ini` for test configuration
- Added `--run-integration` flag for integration tests
- All 73 tests passing