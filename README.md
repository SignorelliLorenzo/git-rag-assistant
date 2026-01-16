# Git Rag Assistant

Ask questions about a codebase with local RAG: fast retrieval, grounded answers, and a simple chat UI.

This runs **entirely on your own machine** (consumer-grade PC friendly):
- Local repo checkout
- Local embeddings
- Local vector search (FAISS)
- Local LLM inference (default via Ollama)

![git-rag-assistant UI](images/hero.png)

## Why
Reading a codebase is hard. Searching is not enough. You want:
- Answers that quote the exact code that matters.
- A workflow that works offline and doesn’t ship your repo to third parties.
- Something you can run on your own hardware.

## What it feels like
- Pick a repo
- Index it once
- Chat: “Where is authentication handled?”
- Get an answer grounded in the code + source snippets

![Chat with sources](images/chat-with-sources.png)

## Bring your repos securely
- Sign in with GitHub (built-in OAuth) to list your repositories.
- Choose one to clone locally—nothing leaves your machine.
- Every repo lives under `repos/<repo_id>` on disk so indexing and retrieval stay local.

## What’s inside
- **Next.js UI** (`web/`)
  - Repo selection + indexing + chat
- **FastAPI backend** (`src/api/`)
  - Orchestrates indexing and the RAG pipeline
- **Chunking + ingestion** (`src/loader/`)
  - Turns files into chunks (with ignore rules to skip noise)
- **Embeddings + vector stores** (`src/embeddings/`)
  - SentenceTransformers embeddings + persisted vector search
- **LLM layer** (`src/llm/`)
  - Navigator + answer prompt building

Technical deep dive:
- [Technical documentation](./TECHNICAL_README.md)

## Defaults (local)
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2`
- **LLM (navigator + answer):** Ollama `qwen2.5:7b`

## Quickstart
### 1) Backend API
1. `pip install -r requirements.txt`
2. `uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000`

### 2) Local LLM (Ollama)
1. Install Ollama
2. `ollama pull qwen2.5:7b`

### 3) Web UI
1. `cd web`
2. `npm install`
3. `npm run dev`
4. Open `http://localhost:3000`

## Notes for consumer-grade PCs
- Indexing is the heavy step; asking is designed to be the fast step.
- Embeddings are computed in batches to avoid OOM.
- If you hit memory issues, tune `RAG_EMBED_MAX_SEQ_LEN`.
