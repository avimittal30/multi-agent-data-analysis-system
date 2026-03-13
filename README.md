# CSV Multi-Agent System — Query-First Architecture

Raw data **never enters the LLM context window**. The LLM sees schema only,
generates code, the code runs against the actual data, and only the compact
result is returned for interpretation.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Orchestrator  (schema only → picks engine)          │
│  gpt-4o                                              │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    engine=sql                engine=pandas
          │                         │
    ┌─────▼──────┐           ┌──────▼──────┐
    │  SQL        │           │  Pandas      │
    │  CodeGen    │           │  CodeGen     │
    │  gpt-4o     │           │  gpt-4o      │
    └─────┬──────┘           └──────┬───────┘
          │                         │
    ┌─────▼─────────────────────────▼──────┐
    │           Executor                    │
    │   runs code against real data         │
    │   returns compact result string       │
    └─────────────────┬─────────────────────┘
                      │ error?
                      ├── yes (< 2 retries) ──► CodeGen (with error context)
                      │
                      └── no / max retries
                              │
                    ┌─────────▼──────────┐
                    │   Interpreter       │
                    │   gpt-4o-mini       │
                    │   result → answer   │
                    └────────────────────┘
```

### Why query-first?

| Old approach | Query-first |
|---|---|
| Loads all CSV rows into prompt | LLM sees schema only |
| Breaks on files > ~500 rows | Works on millions of rows |
| High token cost | Minimal tokens used |
| Hallucinated numbers | Numbers come from actual execution |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
```

## Usage

### CLI

```bash
python run_cli.py data/sales.csv data/orders.csv
```

CLI shows engine chosen, generated code, raw result, and final answer.

### REST API

```bash
uvicorn api.server:app --reload --port 8000
```

```bash
# Upload CSVs
curl -X POST http://localhost:8000/upload \
  -F "files=@data/sales.csv" \
  -F "files=@data/orders.csv"
# → { "session_id": "abc-123", "tables_loaded": ["sales", "orders"], "schemas": {...} }

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "What are the top 5 products by revenue?"}'

# Swagger UI
open http://localhost:8000/docs
```

## Project structure

```
csv_multiagent/
├── state.py              # AgentState TypedDict
├── graph.py              # LangGraph graph + retry loop
├── run_cli.py            # CLI runner
├── requirements.txt
├── agents/
│   ├── __init__.py
│   └── nodes.py          # orchestrator, codegen, executor, interpreter
├── tools/
│   ├── __init__.py
│   └── registry.py       # DataRegistry: loads CSVs → pandas + SQLite
└── api/
    ├── __init__.py
    └── server.py          # FastAPI
```

## Customisation

- **Change model**: edit `_llm` / `_llm_fast` in `agents/nodes.py`
- **More retries**: change `MAX_RETRIES` in `agents/nodes.py`
- **Persistent DB**: swap SQLite in-memory conn in `tools/registry.py` for a file-based or Postgres connection
- **Streaming**: replace `graph.invoke()` with `graph.astream()` + FastAPI `StreamingResponse`
