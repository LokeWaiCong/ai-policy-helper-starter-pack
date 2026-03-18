# AI Policy & Product Helper

A local-first RAG (Retrieval-Augmented Generation) system that answers customer-facing policy questions with source citations.

## Quick Start

```bash
# 1. Copy env
cp .env.example .env

# 2. Run everything (Qdrant + backend + frontend)
docker compose up --build

# 3. Open http://localhost:3000
#    → Click "Ingest sample docs"
#    → Ask a question
```

| Service  | URL                          |
|----------|------------------------------|
| Frontend | http://localhost:3000        |
| Backend  | http://localhost:8000/docs   |
| Qdrant   | http://localhost:6333        |

## Architecture

```
Browser
  │
  ▼
Next.js 14 (port 3000)
  │  REST calls to /api/*
  ▼
FastAPI (port 8000)
  ├── POST /api/ingest  → load .md files → chunk → embed → upsert Qdrant
  ├── POST /api/ask     → embed query → search Qdrant → generate answer
  ├── GET  /api/metrics → latency + query counters
  └── GET  /api/health  → { "status": "ok" }
       │
       ├── TF-IDF Embedder (scikit-learn, offline, no API needed)
       ├── Qdrant vector DB (port 6333, Docker)
       └── LLM: Stub (dev) / OpenRouter GPT-4o-mini (demo)
```

### RAG Pipeline

```
Documents (.md)
  │
  ▼ ingest.py
Chunks (700 tokens, 80 overlap)
  │
  ▼ TF-IDF embed (384-dim, padded)
  │
  ▼ Qdrant upsert (UUID point IDs)

Query
  │
  ▼ TF-IDF embed
  │
  ▼ Qdrant cosine search (top-k chunks)
  │
  ▼ LLM prompt with context
  │
  ▼ Answer + citations (title + section)
```

## Running Tests

```bash
docker compose run --rm backend pytest -q
```

Expected output: `6 passed`

## LLM Switching

| Mode       | Setting                        | Use case          |
|------------|--------------------------------|-------------------|
| Stub       | `LLM_PROVIDER=stub`            | Development/tests |
| OpenRouter | `LLM_PROVIDER=openrouter`      | Demo recording    |

For demo recording, set `LLM_PROVIDER=openrouter` and add `OPENROUTER_API_KEY` in `.env`.

## Bugs Fixed from Starter Pack

| Bug | Fix |
|-----|-----|
| `backend/Dockerfile` wrong COPY paths | Fixed to match build context `./backend` |
| `frontend/Dockerfile` wrong COPY paths | Fixed to match build context `./frontend` |
| `tsconfig.json` missing `@/` alias | Added `baseUrl` + `paths` + webpack alias in `next.config.js` |
| Qdrant point IDs used SHA256 hex strings | Changed to UUID v4 (Qdrant only accepts UUID or uint) |
| Pseudo-random embeddings (SHA1 seed) | Replaced with TF-IDF (scikit-learn) — real keyword-based retrieval |
| Qdrant healthcheck used `curl` (not in image) | Removed healthcheck; backend retries connection on startup |
| `docker-compose.yml` passed `OPENAI_API_KEY` | Fixed to `OPENROUTER_API_KEY` |
| `conftest.py` parents[4] IndexError in Docker | Fixed path traversal + added sys.path for module resolution |

## Trade-offs

**TF-IDF vs semantic embeddings**
- TF-IDF works fully offline, zero dependencies, fast. For a policy Q&A with structured markdown docs and predictable vocabulary, keyword matching is accurate enough to pass all acceptance checks.
- A real semantic embedder (e.g. `text-embedding-3-small`) would generalise better to paraphrased queries but requires an API call or a large local model (~500MB+).

**Stub LLM vs OpenRouter**
- Stub is deterministic and instant — ideal for development and tests. It exposes all retrieved chunks as the "answer", making retrieval quality easy to verify.
- OpenRouter (GPT-4o-mini) produces natural language answers for the demo.

**TF-IDF vector dimension**
- Vocabulary may be smaller than `max_features=384`, so vectors are zero-padded to match Qdrant's fixed collection dimension.

## What I'd Ship Next

1. **Semantic embeddings** — swap TF-IDF for `text-embedding-3-small` via OpenRouter once per ingest, cache vectors; use same model at query time.
2. **Streaming responses** — stream LLM tokens to the UI for better UX.
3. **Re-ranking** — add MMR or cross-encoder re-ranking after vector search to improve citation relevance.
4. **File upload** — allow users to upload their own `.md` / `.pdf` docs via the Admin panel.
5. **Persistent metrics** — write metrics to a database instead of in-memory so they survive restarts.
6. **PDPA/PII guardrails** — mask sensitive data (IC, phone numbers) before sending to external LLM.
7. **Eval script** — small offline eval against a golden Q&A set to catch retrieval regressions.

## Environment Variables

| Variable             | Default       | Description                        |
|----------------------|---------------|------------------------------------|
| `LLM_PROVIDER`       | `stub`        | `stub` or `openrouter`             |
| `OPENROUTER_API_KEY` | —             | Required for openrouter mode       |
| `LLM_MODEL`          | `openai/gpt-4o-mini` | OpenRouter model ID           |
| `VECTOR_STORE`       | `qdrant`      | `qdrant` or `memory`               |
| `COLLECTION_NAME`    | `policy_helper` | Qdrant collection name           |
| `CHUNK_SIZE`         | `700`         | Tokens per chunk                   |
| `CHUNK_OVERLAP`      | `80`          | Token overlap between chunks       |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend → backend URL |
