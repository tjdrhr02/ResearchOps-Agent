# ResearchOps Agent

An agentic AI service that automatically collects, analyzes, and summarizes research into a structured **Research Brief** — given a single research question.

---

## What It Does

Instead of manually searching papers, blogs, and news and piecing them together, you submit a research question and get back a structured brief with:

- Executive summary
- Key trends (evidence-backed)
- Source comparison (papers vs. blogs vs. news perspective)
- RAG-retrieved evidence with citations and relevance scores
- Open questions for further research

---

## How It Works

```
POST /research/run  →  202 Accepted (immediate)
                              ↓  background
          ┌───────────────────────────────────────┐
          │         ResearchWorkflowService        │
          │                                        │
          │  PlannerAgent     analyze query        │
          │       ↓           generate search plan │
          │  CollectorAgent   arXiv + DuckDuckGo   │
          │       ↓           fetch real sources   │
          │  RAG Pipeline     chunk → embed        │
          │       ↓           → vector search      │
          │  SynthesizerAgent analyze + compare    │
          │       ↓           sources & evidence   │
          │  ReporterAgent    write Research Brief  │
          └───────────────────────────────────────┘

GET /research/{job_id}  →  poll for result
```

All agents are powered by **Gemini LLM** via LangChain.  
RAG runs locally with **sentence-transformers** (no additional API key needed).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI, uvicorn |
| LLM | Google Gemini (via LangChain) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| Papers | arXiv REST API |
| Blogs & News | DuckDuckGo Search (ddgs) |
| Article fetch | httpx + BeautifulSoup4 |
| Vector store | In-memory (cosine similarity) |
| Config | python-dotenv |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd ResearchOps-Agent

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-2.0-flash
```

Get a free API key at https://aistudio.google.com

### 3. Run

```bash
uvicorn src.main:app --reload --port 8000
```

Open Swagger UI: http://localhost:8000/docs

### 4. Try It

```bash
# Submit a research job
curl -X POST http://localhost:8000/research/run \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What are the latest advances in RAG for large language models?", "max_sources": 6}'

# Returns immediately:
# {"job_id": "abc-123", "status": "pending", ...}

# Poll for result
curl http://localhost:8000/research/abc-123
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research/run` | Submit a research job (202 Accepted) |
| `GET` | `/research/{job_id}` | Poll job status and retrieve result |
| `GET` | `/research/{job_id}/sources` | List collected source documents |
| `POST` | `/notes/save` | Save a research note |
| `GET` | `/notes/search` | Search saved notes |
| `GET` | `/metrics` | Observability metrics |

---

## Research Brief Output

```json
{
  "executive_summary": "...",
  "key_trends": ["...", "..."],
  "evidence": [
    "[E1] [1] Title — https://... (tech_blog, score: 0.79)\n  \"snippet...\""
  ],
  "source_comparison": ["..."],
  "open_questions": ["..."],
  "citations": ["[1] Title — https://..."],
  "metadata": {
    "research_type": "comparison",
    "source_count": 6,
    "evidence_count": 5
  }
}
```

---

## Project Structure

```
src/
├── api/              FastAPI routers, schemas, DI
├── application/      Orchestrator, WorkflowService, DTOs
├── agents/           Planner / Collector / Synthesizer / Reporter
├── tools/            arXiv, DuckDuckGo, httpx crawlers
├── rag/              chunking → embedding → vector store → retrieval
├── domain/           models, ports, error hierarchy
├── infrastructure/   settings, config
└── observability/    metrics
```

---

## Gemini Free Tier Limits

| Model | Free limit |
|-------|-----------|
| `gemini-2.5-flash` | 20 req/day |
| `gemini-2.0-flash` | 200 req/day |
| `gemini-1.5-flash` | 1,500 req/day |

Switch models in `.env` → `LLM_MODEL=gemini-1.5-flash`

---

## Production Upgrade Path

| Current | Target |
|---------|--------|
| In-memory vector store | PostgreSQL + pgvector |
| Local embeddings (384-dim) | OpenAI / Google embeddings |
| asyncio background task | Celery + Redis job queue |
| DuckDuckGo (unofficial) | Brave Search / SerpAPI |

---

## Documentation

- [Architecture](ARCHITECTURE.md) — system design, agent pipeline, RAG details
- [Development Guide](DEVELOPMENT_GUIDE.md) — setup, project structure, how to extend
- [Project Overview](PROJECT_OVERVIEW.md) — problem statement, features, tech stack
