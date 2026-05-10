# Chat History RAG

A local RAG (Retrieval-Augmented Generation) system for querying Telegram chat history using LangChain, ChromaDB, and Mistral AI.

## Tech Stack

- **LangChain** — RAG pipeline
- **ChromaDB** — local vector store
- **Mistral AI** — embeddings + LLM (`mistral-embed` + `mistral-small-latest`)
- **Gradio** — web UI
- **Python 3.11+**

## Project Structure

```
chat-rag/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── messages.json
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── rag.py
│   ├── query.py
│   └── query_app.py
└── chroma_db/             # auto-created after ingestion
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/chat-rag.git
cd chat-rag
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file from the template**
```bash
cp .env.example .env
```
Then fill in your values in `.env`.

**5. Add your chat export to `data/messages.json`**

## Usage

**Index your data** (run once, or when messages.json changes):
```bash
python src/ingest.py
```

**Start the web UI:**
```bash
python src/query_app.py
```

**Or use the interactive CLI:**
```bash
python src/query.py
```

Example session:
```
🔍 Chat History RAG — type 'quit' to exit

Your question: Кто подавался на визу с боравком менее 6 месяцев?

💬 Answer:
В чате обсуждался вопрос подачи на визу с боравком валидным менее 6 месяцев...

📎 Sources:
  • Conversation: Китай 🇨🇳 | 2024-03-01T15:50:40 → 2024-03-01T15:55:10 | Senders: 1b4fcad569a1, ...
```

## Expected JSON Format

The system expects a Telegram-style chat export:

```json
{
  "name": "Serbia: visas for other countries (chat)",
  "type": "private_supergroup",
  "id": 1608823685,
  "topic": "Китай 🇨🇳",
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

## How It Works

1. `ingest.py` — loads the JSON export, groups messages by `topic`, splits into sliding windows of 5 messages (step=2), embeds via Mistral and stores in ChromaDB
2. `rag.py` — loads ChromaDB, builds a retrieval chain with MMR search (k=6, fetch_k=20) and a custom prompt using `mistral-small-latest`
3. `query.py` — interactive CLI that takes your question, retrieves relevant chunks and returns an answer with sources
4. `query_app.py` — Gradio web UI with a chatbot panel and a sources sidebar

## Notes

- `chroma_db/` is gitignored — each user must run `ingest.py` locally
- `.env` is gitignored — never commit your API key
- Re-run `ingest.py` any time `messages.json` is updated
- Messages with empty `text` field are skipped during ingestion
