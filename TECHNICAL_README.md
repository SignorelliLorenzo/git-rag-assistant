# git-rag-assistant (Technical)

This document is the technical deep dive for `git-rag-assistant`.

For a short, presentation-oriented overview, see:
- `README.md`

## Goals
- Run a full RAG workflow **locally** on a consumer-grade PC.
- Index repositories once, then answer questions quickly using persisted vector stores.
- Keep embedding model usage consistent between indexing and retrieval.

## System architecture
### Components
- **Web UI (`web/`)**
  - Next.js app for login, repo selection, indexing, and chat.
  - Calls the backend via a small typed API client.
- **Backend API (`src/api/`)**
  - FastAPI application exposing:
    - `GET /repos`
    - `POST /index`
    - `POST /ask`
    - `DELETE /repos/{repo_id}`
  - Orchestrates indexing and the RAG pipeline.
- **Chunking / ingestion (`src/loader/`)**
  - Walks the repository and emits chunks suitable for embedding.
  - Uses ignore rules to skip irrelevant folders (for example `migrations`).
- **Embeddings + retrieval (`src/embeddings/`)**
  - Generates embeddings for chunks.
  - Builds and persists vector stores.
  - Loads vector stores and performs similarity search.
- **LLM tooling (`src/llm/`)**
  - Navigator prompt to constrain retrieval to likely-relevant paths.
  - Final prompt construction that attaches retrieved context.
- **Utilities (`src/utils/`)**
  - File iteration, text reading, ignore patterns, etc.

## Data and artifact layout
Artifacts are stored on disk so indexing is not repeated for every question.

- `repos/<repo_id>/...`
  - Local repository checkout.
- `embeddings/<repo_id>/...`
  - Index artifacts (one directory per repo).

Within `embeddings/<repo_id>/` you will typically see:
- `chunks.json`
  - Output of ingestion/chunking.
- `embeddings.json`
  - Chunk records with associated embedding vectors.
- `store_code/`
  - Vector store for code chunks.
- `store_structure/`
  - Vector store for structure chunks (file and directory summaries).
- `repo_map.json` (name may vary)
  - Repository map used by the navigator.
- `embedding_model.json`
  - Persists which embedding model was used to index this repo.

## Indexing pipeline
Indexing is designed to be run once (or explicitly re-run when desired).

High-level steps:
1. **Checkout**
   - Ensure `repos/<repo_id>` exists (clone if needed).
2. **Ingest / chunk**
   - Walk repository files.
   - Apply ignore rules.
   - Split content into overlapping chunks.
3. **Embed**
   - Use SentenceTransformers to embed each chunk.
   - Embedding runs in mini-batches to avoid OOM.
4. **Build vector stores**
   - Persist searchable stores to disk (FAISS preferred, NumPy fallback).
5. **Persist model metadata**
   - Save `embedding_model.json` so retrieval uses the same embedding space.

OOM resilience:
- Embedding uses a batch-size backoff strategy (decreasing batch size on OOM).
- Sequence length can be capped with `RAG_EMBED_MAX_SEQ_LEN`.

## Asking / retrieval pipeline
The ask flow should stay on a fast path (no full re-indexing):

1. Ensure stores exist.
2. (Optional) Refine / route the question.
   - Navigator: a lightweight LLM step that proposes likely relevant paths to constrain retrieval.
   - Structure heuristics: the backend can detect “structure questions” (e.g. repo layout, file locations) and rewrite/route accordingly.
3. Embed the (possibly refined) question once using the repo’s embedding model.
4. Retrieve top-K candidates from:
   - code store
   - structure store
   (Both stores share the same query embedding to keep latency low.)
5. Build a final prompt that attaches retrieved snippets.
6. Generate the answer using a local LLM backend.

Implementation note:
- The `/ask` endpoint computes the query embedding once, then performs retrieval against both vector stores.
- The navigator output is used as a path filter to improve precision and reduce irrelevant matches.

## Model choices
### Embeddings
- **Default embedding model**: `sentence-transformers/all-mpnet-base-v2`
- Runs locally via `sentence-transformers`.

Per-repo model consistency:
- At indexing time, the embedding model name is stored under `embeddings/<repo_id>/embedding_model.json`.
- At question time, the backend resolves the correct model for that repo.

### LLM (navigator + answer)
- Default backend: **Ollama**
- Default model id: `qwen2.5:7b`

The navigator and the answer generator can be configured independently via the API payload.

## Configuration (environment variables)
- `RAG_EMBED_MAX_SEQ_LEN`
  - Caps SentenceTransformer `max_seq_length`.
  - Default: `512`.
  - Lower this if you see CPU/GPU OOM during embedding.
- `RAG_LEGACY_EMBED_MODEL`
  - Fallback embedding model for older repos missing model metadata.

## Web UI behavior
- The UI is intended to **index before allowing chat**.
- Once indexed, `/ask` should not re-run indexing.
- The chat view auto-scrolls to new responses.

## API reference (summary)
- `GET /repos`
  - Lists repositories and whether each one is indexed.
- `POST /index`
  - Runs indexing for a repository.
  - When artifacts exist and `force_reindex` is false, the backend skips work.
- `POST /ask`
  - Runs the RAG pipeline.
  - If `ensure_index` is true (or artifacts are missing), indexing may be triggered.
- `DELETE /repos/{repo_id}`
  - Deletes local checkout and all indexing artifacts for that repo.

## Development
### Backend
- Install Python deps: `pip install -r requirements.txt`
- Run:
  - `uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000`

### Web
- From `web/`:
  - `npm install`
  - `npm run dev`

## Notes
- First run may download models; subsequent runs use cached models.
- For best retrieval quality, ensure the same embedding model is used for both indexing and querying (this project enforces that per repo).
