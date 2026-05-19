# Chat History RAG

A local RAG (Retrieval-Augmented Generation) system for querying visa-related Telegram chat history, enriched with official embassy data, using LangChain, LangGraph, ChromaDB, and Mistral AI. Embeddings are generated locally with Octen/Octen-Embedding-0.6B (CPU, 4 threads).

## Tech Stack

- **LangChain** — RAG pipeline
- **LangGraph** — graph with input/output guardrails, classification, and parallel retrieval (community + official sources)
- **[LangSmith](https://smith.langchain.com)** — optional tracing and observability
- **ChromaDB** — local vector store
- **Octen/Octen-Embedding-0.6B** — local embeddings via `sentence-transformers` (CPU, 4 threads, preloaded in background)
- **Mistral AI** — LLM (`mistral-small-latest` with `gemini-3.1-flash-lite` fallback)
- **Gradio** — web UI
- **Pydantic / pydantic-settings** — data models and settings validation
- **Python 3.14**
- **uv** — package manager
- **Ruff** — linting and formatting

## Project Structure

```
chat-rag/
├── .env
├── .env.example
├── .python-version
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── data/
│   ├── messages.json
│   ├── germany_visa_official.json
│   └── evaluations/           # embedding model comparison reports
├── src/
│   ├── __init__.py
│   ├── config.py              # settings via pydantic-settings
│   ├── llm.py                 # LLM + embeddings (Octen model preloaded in background thread, singleton)
│   ├── dashboard.py           # visa statistics from extracted data
│   ├── query.py               # interactive CLI
│   ├── query_app.py           # Gradio app entry point (assembles tabs, launches with Citrus theme)
│   ├── graph/
│   │   ├── graph.py           # LangGraph pipeline definition
│   │   ├── nodes.py           # node functions (input_guard, classify, retrieve, generate, output_guard, reject)
│   │   └── state.py           # GraphState TypedDict
│   ├── guardrails/
│   │   ├── __init__.py        # public API re-exports
│   │   ├── input.py           # run_input_guardrails: PII redaction + injection detection
│   │   ├── output.py          # run_output_guardrails: PII redaction + internal error scrubbing
│   │   ├── _injection.py      # prompt-injection heuristics
│   │   ├── _pii.py            # input PII regex redaction
│   │   ├── _output_tokens.py  # output PII token redaction
│   │   └── _internal_errors.py # internal error leak rewriting
│   ├── ingest/
│   │   ├── ingest.py          # ingestion script
│   │   └── models.py          # pydantic data models (Message, ChatExport, ChunkMetadata)
│   ├── prompts/
│   │   ├── answer.py          # ANSWER_PROMPT template
│   │   └── classify.py        # CLASSIFY_PROMPT template
│   └── ui/
│       ├── __init__.py
│       ├── chat_tab.py        # Chat tab layout and wiring (two-step: user message → LLM answer); sources rendered via gr.HTML + gr.State
│       ├── dashboard_tab.py   # Dashboard tab with tourist visa stats table
│       └── handlers.py        # Gradio event handlers (stage_user_message, complete_assistant_message, _sources_to_html)
└── chroma_db/                 # auto-created after ingestion
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/chat-rag.git
cd chat-rag
```

**2. Install uv** (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Install dependencies**
```bash
uv sync
```

**4. Create `.env` file from the template**
```bash
cp .env.example .env
```
Then fill in your values in `.env`. At minimum, set `MISTRAL_API_KEY`.

To enable LangSmith tracing, also set:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT=visa-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```
LangSmith tracing is off by default (`LANGSMITH_TRACING=false`).

**5. Add your chat export to `data/messages.json`**

**6. Add official embassy data to `data/germany_visa_official.json`** (see format below)

## Usage

**Index your data** (run once, or when `messages.json` changes):
```bash
uv run src/ingest/ingest.py
uv run src/ingest/ingest.py --limit 100  # ingest only first N messages
```

**Start the web UI:**
```bash
uv run src/query_app.py
```

**Or use the interactive CLI:**
```bash
uv run src/query.py
```

Example session:
```
🔍 Chat History RAG — type 'quit' to exit

Your question: Кто подавался на визу с боравком менее 6 месяцев?

💬 Answer:
[OFFICIAL] According to the German Embassy Belgrade...
[COMMUNITY] В чате обсуждался вопрос подачи на визу с боравком валидным менее 6 месяцев...

📎 Sources:
  • Topic: Германия | 2024-03-01T15:50:40 → 2024-03-01T15:55:10 | Senders: 1b4fcad569a1, ...
```

## How It Works

The system uses a **LangGraph pipeline** with input/output guardrails, a classification node, and two parallel retrieval branches:

```
START
  └──▶ input_guard
        ├──▶ [injection_flagged] reject → END
        └──▶ [clean] classify_question
              ├──▶ [relevant]  retrieve_from_chat   (ChromaDB MMR search over community messages)
              │    [relevant]  load_official_data   (German Embassy JSON)
              │                └──▶ generate_answer (waits for both, answers with labeled sources)
              │                     └──▶ output_guard → END
              └──▶ [off_topic] reject → END
```

1. `config.py` — loads settings (API key, paths, optional LangSmith config) from `.env` via `pydantic-settings`
2. `llm.py` — embeddings via local `Octen/Octen-Embedding-0.6B` (CPU, 4 threads), preloaded at module import in a background daemon thread; LLM uses `ChatMistralAI` with `ChatGoogleGenerativeAI` as fallback; both are singletons returned by `get_embeddings()` / `get_llm()`
3. `ingest/models.py` — pydantic models for `Message`, `ChatExport`, and `ChunkMetadata`
4. `ingest/ingest.py` — loads the JSON export, groups messages by `topic`, splits into sliding windows of 5 messages (step=2), embeds via Octen/Octen-Embedding-0.6B in batches of 100 and stores in ChromaDB
5. `graph/state.py` — `GraphState` TypedDict shared across all nodes
6. `graph/nodes.py` — node functions: `input_guard`, `classify_question`, `reject`, `retrieve_from_chat`, `load_official_data`, `generate_answer`, `output_guard`; `_build_official_context` flattens the embassy JSON for the LLM prompt
7. `graph/graph.py` — assembles and compiles the `StateGraph` with conditional routing after input guard and classification
8. `guardrails/input.py` — `run_input_guardrails`: redacts PII from user input and detects prompt-injection attempts
9. `guardrails/output.py` — `run_output_guardrails`: redacts PII tokens and rewrites internal error leaks from LLM output
10. `prompts/classify.py` — prompt for the topic classifier (relevant / off_topic)
11. `prompts/answer.py` — prompt template that structures `[OFFICIAL]` and `[COMMUNITY]` labeled sections
12. `query.py` — interactive CLI that invokes the graph and prints the answer with sources
13. `query_app.py` — Gradio entry point: assembles a `gr.Blocks` app with two tabs (Dashboard + Chat) and launches with the Citrus theme; initialises LangSmith tracing when enabled
14. `ui/chat_tab.py` — Chat tab layout: chatbot panel (scale=3) + sources sidebar (scale=1, min_width=240); uses a two-step event chain so the user message appears immediately before the LLM answer loads; sources are stored in a `gr.State` and rendered into a `gr.HTML` component via a `chatbot.change` listener
15. `ui/dashboard_tab.py` — Dashboard tab: renders a `gr.Dataframe` with per-country tourist visa statistics
16. `ui/handlers.py` — Gradio event handlers: `stage_user_message` appends the user turn instantly; `complete_assistant_message` invokes `visa_graph` and appends the assistant reply with sources; `_sources_to_html` renders retrieved `Document` objects as styled HTML cards showing topic, time range, senders, and a 280-character content preview
17. `dashboard.py` — reads extracted JSONL data under `data/extracted/` to compute per-country tourist visa statistics (wait times, approval rates, validity, multi-entry counts)

## Expected JSON Formats

### Chat export (`data/messages.json`)

The system expects a Telegram-style chat export:

```json
{
  "name": "Serbia: visas for other countries (chat)",
  "type": "private_supergroup",
  "id": 1608823685,
  "topic": "Германия 🇩🇪",
  "messages": [
    {
      "id": 131985,
      "date": "2024-03-01T15:50:40",
      "from": "1b4fcad569a1",
      "text": "Кто-нибудь подавался с боравком, валидным меньше чем 6 месяцев?"
    }
  ]
}
```

| Field | Description |
|---|---|
| `name` | Chat name |
| `topic` | Topic/thread name — used as `conversation_id` |
| `messages` | Array of messages |
| `messages[].date` | Message timestamp |
| `messages[].from` | Sender identifier |
| `messages[].text` | Message content |

### Official embassy data (`data/germany_visa_official.json`)

Structured JSON scraped or manually assembled from the embassy website:

```json
{
  "source": "https://belgrad.diplo.de/...",
  "title": "National Visa — German Embassy Belgrade",
  "last_updated": "2024-01-01",
  "requirements": { ... },
  "appointment": { ... }
}
```

The `_build_official_context` function in `graph/nodes.py` flattens any nested structure into labeled sections for the LLM prompt.

## Code Quality

**Check for linting issues:**
```bash
uv run ruff check .
```

**Auto-fix linting issues:**
```bash
uv run ruff check --fix .
```

**Format code:**
```bash
uv run ruff format .
```

**Check formatting without applying changes:**
```bash
uv run ruff format --check .
```

## Notes

- `chroma_db/` is committed to this repo so the index is shared — re-run `ingest/ingest.py` if `messages.json` changes
- The ChromaDB index was built with `Octen/Octen-Embedding-0.6B` embeddings (1024-dim); do not mix with `mistral-embed` (same dim but different vector space)
- `.env` is gitignored — never commit your API key
- Messages with empty `text` field are skipped during ingestion
- The embedding model loads in a background thread at import time — app startup is not blocked; first query waits for the model to be ready
- Off-topic questions are rejected before retrieval by the `classify_question` guardrail
