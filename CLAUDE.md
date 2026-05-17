# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

Always use `uv` instead of `pip`. Never use `pip install` — use `uv add` to add dependencies and `uv sync` to install.

## Commands

All commands run from `chat-rag/`:

```bash
# Install dependencies
uv sync

# Ingest data into ChromaDB (run once, or when messages.json changes)
uv run src/ingest/ingest.py
uv run src/ingest/ingest.py --limit 100  # ingest only first N messages

# Start the Gradio web UI
uv run src/query_app.py

# Interactive CLI query
uv run src/query.py

# Lint
uv run ruff check .
uv run ruff check --fix .

# Format
uv run ruff format .
uv run ruff format --check .
```

## Architecture

The system is a RAG pipeline over Telegram chat history, enriched with official embassy JSON. It uses a **LangGraph fan-out/fan-in graph** as the core query pipeline:

```
START
  ├──▶ retrieve_from_chat   (ChromaDB MMR search over embedded community messages)
  ├──▶ load_official_data   (reads germany_visa_official.json directly)
  └──▶ generate_answer      (waits for both branches, then prompts Mistral)
       └──▶ END
```

- `src/config.py` — `pydantic-settings` loads all config from `.env` (requires `MISTRAL_API_KEY`)
- `src/ingest/models.py` — Pydantic models: `Message`, `ChatExport` (input), `ChunkMetadata` (stored in ChromaDB metadata)
- `src/ingest/ingest.py` — Reads `data/messages.json`, groups by `topic`, creates sliding-window chunks (size=5, step=2), embeds with `mistral-embed`, stores in ChromaDB. Batched at 100 docs with 3s delay to respect Mistral rate limits.
- `src/graph.py` — Defines `GraphState` TypedDict and the compiled `visa_graph`. The `_build_official_context` helper recursively flattens the official JSON into a labeled string for the prompt.
- `src/query_app.py` — Gradio UI: two-column layout (chat + sources sidebar), invokes `visa_graph` directly.
- `src/query.py` — CLI wrapper around `visa_graph`.

## Data

- `data/messages.json` — Telegram export (single topic/thread per file). The `topic` field is used as `conversation_id` throughout the pipeline.
- `data/germany_visa_official.json` — Structured embassy data; any nested JSON is flattened at query time by `_build_official_context`.
- `chroma_db/` — Persisted vector index; committed to repo so the index is shared. Re-run `ingest/ingest.py` if `messages.json` changes.

## Environment

Copy `.env.example` to `.env` and set `MISTRAL_API_KEY`. The `.env` file is gitignored.

Scripts in `src/` import from `config` and `ingest.models` without the `src.` prefix — run them with `uv run src/<file>.py` from the `chat-rag/` directory so Python resolves `src/` as the working path.
