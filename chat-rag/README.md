# Chat History RAG

A local RAG (Retrieval-Augmented Generation) system for querying visa-related Telegram chat history, enriched with official embassy data, using LangChain, LangGraph, ChromaDB, and Mistral AI. Embeddings are generated locally with Octen/Octen-Embedding-0.6B (CPU, 4 threads).

## Tech Stack

- **LangChain** — RAG pipeline
- **LangGraph** — graph with input/output guardrails, classification, country detection, and parallel retrieval (community + official sources)
- **[LangSmith](https://smith.langchain.com)** — optional tracing, observability, dataset management, and LLM-as-judge evaluation
- **ChromaDB** — local vector store
- **Octen/Octen-Embedding-0.6B** — local embeddings via `sentence-transformers` (CPU, 4 threads, preloaded in background)
- **Mistral AI** — LLM (`mistral-small-latest` with `gemini-3.1-flash-lite` fallback)
- - **OpenAI Agents SDK + LiteLLM** — used for LangSmith dataset population and demo agents
- **Gradio** — web UI
- **Pydantic / pydantic-settings** — data models and settings validation
- **Python 3.14**
- **uv** — package manager
- **Ruff** — linting and formatting
- **pre-commit** — git hooks for Ruff and common file checks

## Project Structure

```
chat-rag/
├── .env
├── .env.example
├── .python-version
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── deploy_modal.py
├── data/
│   ├── messages.json
│   ├── albania_visa_official.json
│   ├── austria_visa_official.json
│   ├── belgium_visa_official.json
│   ├── bulgaria_visa_official.json
│   ├── croatia_visa_official.json
│   ├── cyprus_visa_official.json
│   ├── france_visa_official.json
│   ├── germany_visa_official.json
│   ├── greece_visa_official.json
│   ├── hungary_visa_official.json
│   ├── italy_visa_official.json
│   ├── netherlands_visa_official.json
│   ├── poland_visa_official.json
│   ├── portugal_visa_official.json
│   ├── romania_visa_official.json
│   ├── slovenia_visa_official.json
│   ├── spain_visa_official.json
│   ├── sweden_visa_official.json
│   ├── switzerland_visa_official.json
│   ├── united_kingdom_visa_official.json
│   └── usa_visa_official.json
│   └── evaluations/           # embedding model comparison reports
├── src/
│   ├── __init__.py
│   ├── config.py              # settings via pydantic-settings
│   ├── llm.py                 # LLM + embeddings (Octen model preloaded in background thread, singleton)
│   ├── dashboard.py           # visa statistics from extracted data
│   ├── query.py               # interactive CLI
│   ├── query_app.py           # Gradio app entry point (assembles tabs, launches with Citrus theme)
│   ├── langsmith_eval.py      # LLM-as-judge evaluation runner against a LangSmith dataset
│   ├── langsmith_launch.py    # populates a LangSmith dataset from test_cases_germany.json
│   ├── langsmith_demo.py      # demo agent (OpenAI Agents SDK + LiteLLM) with LangSmith tracing
│   ├── graph/
│   │   ├── graph.py           # LangGraph pipeline definition
│   │   ├── nodes.py           # node functions (input_guard, classify, country detection, retrieve, generate, output_guard, reject)
│   │   └── state.py           # GraphState TypedDict (includes country field)
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
│   ├── logging_/
│   │   ├── __init__.py        # public API re-exports
│   │   ├── hierarchical.py    # HierarchicalFormatter + trace() context manager; indents nested node logs
│   │   └── debug.py           # debug_enter / debug_exit helpers (payload-aware structured debug logs)
│   ├── prompts/
│   │   ├── answer.py          # ANSWER_PROMPT template
│   │   ├── classify.py        # CLASSIFY_PROMPT template
│   │   └── detect.py          # COUNTRY_DETECT_PROMPT + KNOWN_COUNTRIES list
│   └── ui/
│       ├── __init__.py
│       ├── chat_tab.py        # Chat tab layout and wiring (two-step: user message → LLM answer); sources rendered via gr.HTML + gr.State
│       ├── dashboard_tab.py   # Dashboard tab with tourist visa stats table
│       └── handlers.py        # Gradio event handlers (stage_user_message, complete_assistant_message, _sources_to_html)
└── chroma_db/                 # downloaded via make chroma-db (gitignored)
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

**4. Download the ChromaDB vector index**
```bash
make chroma-db
```

**5. Create `.env` file from the template**
```bash
cp .env.example .env
```
Then fill in your values in `.env`. Required keys:

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Mistral AI API key (embeddings + LLM) |
| `GEMINI_API_KEY` | No | Google Gemini API key (alternative LLM backend) |
| `CHAT_RAG_DEBUG` | No | Set to `0` for INFO-only logs; `1` (default) enables DEBUG output |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing (default: `false`) |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name (e.g. `visa-agent`) |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |

**6. Add your chat export to `data/messages.json`**

**7. Add official embassy data** for each supported country as `data/{country}_visa_official.json` (see format below). Supported country slugs: `albania`, `austria`, `belgium`, `bulgaria`, `croatia`, `cyprus`, `france`, `germany`, `greece`, `hungary`, `italy`, `netherlands`, `poland`, `portugal`, `romania`, `slovenia`, `spain`, `sweden`, `switzerland`, `united_kingdom`, `usa`.

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

**LangSmith evaluation** (requires `LANGSMITH_API_KEY` and a populated dataset):
```bash
# Create / populate a LangSmith dataset from data/test_cases_germany.json
uv run src/langsmith_launch.py

# Run LLM-as-judge evaluation against the dataset
uv run src/langsmith_eval.py

# Re-score an existing experiment without re-running the RAG
uv run src/langsmith_eval.py --existing <experiment-name>
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

The system uses a **LangGraph pipeline** with input/output guardrails, a classification node, LLM-based country detection, and two parallel retrieval branches:

```
START
  └──▶ input_guard
        ├──▶ [injection_flagged] reject → END
        └──▶ [clean] classify_question
              ├──▶ [relevant]  retrieve_from_chat   (ChromaDB MMR search over community messages)
              │    [relevant]  load_official_data   (LLM detects country → loads {country}_visa_official.json)
              │                └──▶ generate_answer (waits for both, answers with labeled sources)
              │                     └──▶ output_guard → END
              └──▶ [off_topic] reject → END
```

1. `config.py` — loads settings (API key, `official_data_dir` path, optional LangSmith config) from `.env` via `pydantic-settings`
2. `llm.py` — factory functions for `ChatMistralAI` and `MistralAIEmbeddings`; LLM is configured with `max_retries=6` to handle transient API errors
3. `logging_/hierarchical.py` — `HierarchicalFormatter` and `trace()` context manager; wrapping a node in `trace("name")` emits an indented console tree that mirrors the LangSmith trace; `configure_logging()` wires the formatter to the root logger (idempotent). Verbosity is controlled by `CHAT_RAG_DEBUG` in `.env`.
4. `logging_/debug.py` — `debug_enter` / `debug_exit` helpers that emit structured JSON payloads at DEBUG level, summarizing large strings and collections to keep logs readable.
5. `ingest/models.py` — pydantic models for `Message`, `ChatExport`, and `ChunkMetadata`
6. `ingest/ingest.py` — loads the JSON export, groups messages by `topic`, splits into sliding windows of 5 messages (step=2), embeds via Mistral in batches of 100 and stores in ChromaDB
7. `graph/state.py` — `GraphState` TypedDict shared across all nodes; includes `country` field populated by `load_official_data`
8. `graph/nodes.py` — node functions: `input_guard`, `classify_question`, `reject`, `retrieve_from_chat`, `load_official_data`, `generate_answer`, `output_guard`; `_detect_country_llm` uses the LLM to identify the destination country from the question; `_build_official_context` flattens the embassy JSON for the LLM prompt
9. `graph/graph.py` — assembles and compiles the `StateGraph` with conditional routing after input guard and classification
10. `guardrails/input.py` — `run_input_guardrails`: redacts PII from user input and detects prompt-injection attempts
11. `guardrails/output.py` — `run_output_guardrails`: redacts PII tokens and rewrites internal error leaks from LLM output
12. `prompts/classify.py` — prompt for the topic classifier (relevant / off_topic)
13. `prompts/detect.py` — `COUNTRY_DETECT_PROMPT` instructs the LLM to return a country slug from `KNOWN_COUNTRIES`; used by `_detect_country_llm` in `nodes.py`
14. `prompts/answer.py` — prompt template that structures `[OFFICIAL]` and `[COMMUNITY]` labeled sections
15. `query.py` — interactive CLI that invokes the graph and prints the answer with sources
16. `query_app.py` — Gradio entry point: assembles a `gr.Blocks` app with two tabs (Dashboard + Chat) and launches with the Citrus theme; initializes LangSmith tracing when enabled
17. `ui/chat_tab.py` — Chat tab layout: chatbot panel (scale=3) + sources sidebar (scale=1, min_width=240); uses a two-step event chain so the user message appears immediately before the LLM answer loads; sources are stored in a `gr.State` and rendered into a `gr.HTML` component via a `chatbot.change` listener
18. `ui/dashboard_tab.py` — Dashboard tab: renders a `gr.Dataframe` with per-country tourist visa statistics
19. `ui/handlers.py` — Gradio event handlers: `stage_user_message` appends the user turn instantly; `complete_assistant_message` invokes `visa_graph` and appends the assistant reply with sources; `_sources_to_html` renders retrieved `Document` objects as styled HTML cards showing topic, time range, senders, and a 280-character content preview
20. `dashboard.py` — reads extracted JSONL data under `data/extracted/` to compute per-country tourist visa statistics (wait times, approval rates, validity, multi-entry counts)
21. `langsmith_launch.py` — creates a LangSmith dataset (`visa_qa_germany_3_v1`) from `data/test_cases_germany.json` using the OpenAI Agents SDK with LiteLLM; skips creation if the dataset already exists
22. `langsmith_eval.py` — runs LLM-as-judge evaluation over the dataset: `main()` invokes `visa_graph` on each example and scores the answer with a structured Mistral judge; `main_existing(experiment_name)` re-scores a previous experiment without re-running the RAG

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

### Official embassy data (`data/{country}_visa_official.json`)

One JSON file per supported country, named by slug (e.g. `germany_visa_official.json`, `france_visa_official.json`). Structured JSON scraped or manually assembled from the embassy website:

```json
{
  "source": "https://belgrad.diplo.de/...",
  "title": "National Visa — German Embassy Belgrade",
  "last_updated": "2024-01-01",
  "requirements": { ... },
  "appointment": { ... }
}
```

At query time, `_detect_country_llm` identifies the destination country from the user's question and loads the matching file. `_build_official_context` in `graph/nodes.py` then flattens any nested structure into labeled sections for the LLM prompt. If no country is detected, or no matching file exists, the official context is omitted gracefully.

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

**Install pre-commit hooks** (runs Ruff + common checks on every commit):
```bash
uv tool install pre-commit
pre-commit install
```

## Deploy to Modal

The app is deployed on [Modal](https://modal.com). A GitHub Actions workflow (`.github/workflows/deploy-modal.yml`) automatically deploys on every push to `master` that changes files under `chat-rag/`.

To deploy manually, make sure secrets are in your `.env` file, then run:

```bash
uv run modal deploy deploy_modal.py
```

Required GitHub secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `LANGSMITH_API_KEY`.

## Scrape Agent

The `scrape-embassy` Claude Code sub-agent automates fetching official visa information from an embassy website and saving it as a structured JSON file ready for ingestion.

### How to invoke

Open Claude Code and run:

```
/agent scrape-embassy France
```

Replace `France` with any supported country name. The agent resolves to `scrape-embassy.md` in `.claude/agents/`.

### What it does

1. **Finds the embassy** — searches for the official embassy/consulate of the requested country in Belgrade, Serbia (prefers `.gov` domains).
2. **Navigates to the visa page** — follows links titled "Visa", "Consular Services", "How to apply", etc. Translates non-English content when needed.
3. **Extracts structured data** — parses and maps all available information into a fixed JSON schema covering:
   - Embassy contact details and address
   - Visa eligibility (Serbian citizens + other nationals)
   - Processing times
   - Application location and booking instructions
   - Visa fees (adults, children, exemptions)
   - Required documents (general + purpose-specific + occupational)
   - Special provisions
4. **Saves the file** — writes `chat-rag/data/<country_slug>_visa_official.json` (e.g. `france_visa_official.json`).
5. **Reports** — confirms the URL(s) fetched, output path, key findings, and any fields it could not populate.

### Output format

```json
{
  "source": "https://...",
  "title": "Visa page title",
  "last_updated": "YYYY-MM-DD",
  "country": "France",
  "embassy_in_serbia": { "name": "...", "address": "...", "website": "...", "phone": null, "email": null },
  "visa_eligibility": { "serbian_citizens": "...", "other_nationals": "..." },
  "processing_times": { "minimum_calendar_days": null, "recommended_submission_weeks_before_travel": "4–6", "maximum_calendar_days": null, "notes": null },
  "application_location": { "center": "...", "address": "...", "booking": "..." },
  "visa_fees": { "adults_eur": null, "children_eur": null, "service_fee": null, "fee_exemptions": [] },
  "required_documents_all_applicants": ["..."],
  "additional_documents_by_nationality": [],
  "documents_for_minors": { "required": [], "note": null },
  "purpose_specific_requirements": { "tourism": [], "business": [], "family_or_private_visit": [], "medical_treatment": [], "transit": [] },
  "occupational_documentation": { "employees": [], "entrepreneurs": [], "students": [], "retirees": [] },
  "special_provisions": null,
  "contact_and_resources": { "embassy_location": "Belgrade, Serbia", "appointment_url": null, "consulate_website": "..." }
}
```

The file is immediately usable by the RAG pipeline — `_detect_country_llm` will pick it up for queries about that country.

## Docker

Build the image (chroma_db is downloaded automatically via the Makefile):

```bash
docker build -t chat-rag .
```

Run the Gradio web UI on http://localhost:7860:

```bash
docker run --rm -p 7860:7860 --env-file .env -e GRADIO_SERVER_NAME=0.0.0.0 chat-rag
```

**Interactive shell** (useful for debugging, running one-off scripts, or ad-hoc queries):

```bash
docker run --rm -it --env-file .env -e GRADIO_SERVER_NAME=0.0.0.0 -p 7860:7860 chat-rag bash
```

**Dev mode** — mount local `src/` and `data/` so edits on the host are reflected instantly inside the container (chroma_db is baked into the image). To get a shell in dev mode instead of launching the app:

```bash
docker run --rm -it \
  -p 7860:7860 \
  --env-file .env \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -v ./src:/app/src \
  -v ./data:/app/data \
  chat-rag bash
```

| Variable | Default | Description |
|---|---|---|
| `APP_MODE` | `query_app` | Entry point script (`query_app` or `langsmith_launch`) |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Set to `0.0.0.0` to access Gradio from the host |

## Notes

- `chroma_db/` is gitignored — download it with `make chroma-db`; re-run `ingest/ingest.py` if `messages.json` changes
- The ChromaDB index was built with `Octen/Octen-Embedding-0.6B` embeddings (1024-dim); do not mix with `mistral-embed` (same dim but different vector space)
- `.env` is gitignored — never commit your API key
- Messages with empty `text` field are skipped during ingestion
- The embedding model loads in a background thread at import time — app startup is not blocked; first query waits for the model to be ready
- Off-topic questions are rejected before retrieval by the `classify_question` guardrail
