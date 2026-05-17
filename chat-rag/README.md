# Chat History RAG

A local RAG (Retrieval-Augmented Generation) system for querying visa-related Telegram chat history, enriched with official embassy data, using LangChain, LangGraph, ChromaDB, and Mistral AI.

## Tech Stack

- **LangChain** — RAG pipeline
- **LangGraph** — graph with classification guardrail + parallel retrieval (community + official sources)
- **ChromaDB** — local vector store
- **Mistral AI** — embeddings + LLM (`mistral-embed` + `mistral-small-latest`)
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
│   └── germany_visa_official.json
├── src/
│   ├── __init__.py
│   ├── config.py              # settings via pydantic-settings
│   ├── llm.py                 # LLM + embeddings factory functions
│   ├── dashboard.py           # visa statistics from extracted data
│   ├── query.py               # interactive CLI
│   ├── query_app.py           # Gradio web UI
│   ├── graph/
│   │   ├── graph.py           # LangGraph pipeline definition
│   │   ├── nodes.py           # node functions (classify, retrieve, generate, reject)
│   │   └── state.py           # GraphState TypedDict
│   ├── ingest/
│   │   ├── ingest.py          # ingestion script
│   │   └── models.py          # pydantic data models (Message, ChatExport, ChunkMetadata)
│   └── prompts/
│       ├── answer.py          # ANSWER_PROMPT template
│       └── classify.py        # CLASSIFY_PROMPT template
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
Then fill in your values in `.env`.

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

The system uses a **LangGraph pipeline** with a classification guardrail followed by two parallel retrieval branches:

```
START
  └──▶ classify_question
        ├──▶ [relevant]  retrieve_from_chat   (ChromaDB MMR search over community messages)
        │    [relevant]  load_official_data   (German Embassy JSON)
        │                └──▶ generate_answer (waits for both, answers with labeled sources)
        │                     └──▶ END
        └──▶ [off_topic] reject → END
```

1. `config.py` — loads settings (API key, paths) from `.env` via `pydantic-settings`
2. `llm.py` — factory functions for `ChatMistralAI` and `MistralAIEmbeddings`
3. `ingest/models.py` — pydantic models for `Message`, `ChatExport`, and `ChunkMetadata`
4. `ingest/ingest.py` — loads the JSON export, groups messages by `topic`, splits into sliding windows of 5 messages (step=2), embeds via Mistral in batches of 100 and stores in ChromaDB
5. `graph/state.py` — `GraphState` TypedDict shared across all nodes
6. `graph/nodes.py` — node functions: `classify_question`, `reject`, `retrieve_from_chat`, `load_official_data`, `generate_answer`; `_build_official_context` flattens the embassy JSON for the LLM prompt
7. `graph/graph.py` — assembles and compiles the `StateGraph` with conditional routing after classification
8. `prompts/classify.py` — prompt for the guardrail classifier (relevant / off_topic)
9. `prompts/answer.py` — prompt template that structures `[OFFICIAL]` and `[COMMUNITY]` labeled sections
10. `query.py` — interactive CLI that invokes the graph and prints the answer with sources
11. `query_app.py` — Gradio web UI with a chatbot panel and a sources sidebar
12. `dashboard.py` — reads extracted JSONL data under `data/extracted/` to compute per-country tourist visa statistics (wait times, approval rates, validity, multi-entry counts)

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
- `.env` is gitignored — never commit your API key
- Messages with empty `text` field are skipped during ingestion
- Ingestion is batched (100 docs / batch, 3 s delay) to stay within Mistral API rate limits
- Off-topic questions are rejected before retrieval by the `classify_question` guardrail
