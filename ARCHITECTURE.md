# TMC-Optimal — Architecture & Tech Stack

## Overview

TMC-Optimal is a **query-first multi-agent data analysis system**. Users upload CSV files and ask natural-language questions. A pipeline of LLM agents decides how to answer, generates executable code, runs it against the data, and returns a human-readable interpretation — **raw data rows never enter the LLM context**.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Entry Points                                      │
│                                                                             │
│   ┌──────────────────────┐          ┌──────────────────────┐               │
│   │   FastAPI REST API   │          │   CLI (run_cli.py)   │               │
│   │   (api/server.py)    │          │                      │               │
│   │                      │          │  python run_cli.py   │               │
│   │  POST /upload        │          │  data/file.csv       │               │
│   │  POST /chat          │          └──────────┬───────────┘               │
│   │  GET  /session/{id}  │                     │                           │
│   │  DELETE /session/{id}│                     │                           │
│   └──────────┬───────────┘                     │                           │
│              │                                 │                           │
└──────────────┼─────────────────────────────────┼───────────────────────────┘
               │                                 │
               ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Session / Registry Layer                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    DataRegistry (tools/registry.py)                 │  │
│   │                                                                     │  │
│   │   CSV files ──► pandas DataFrames ──► in-memory SQLite DB           │  │
│   │                                                                     │  │
│   │   ┌──────────────────┐    ┌─────────────────────────────────────┐  │  │
│   │   │  DataFrame Store │    │  SQLite DB (:memory:)                │  │  │
│   │   │  { table: df }   │    │  (tables mirrored from DataFrames)  │  │  │
│   │   └──────────────────┘    └─────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   │   Exposes to LLM: schema only (column names, dtypes, row count,    │  │
│   │   5-row sample) — never full raw data                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  schema + user query
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LangGraph Agent Pipeline (graph.py)                  │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                                                                      │ │
│   │  START                                                               │ │
│   │    │                                                                 │ │
│   │    ▼                                                                 │ │
│   │  ┌─────────────────────────────────────────────────────┐            │ │
│   │  │  1. ORCHESTRATOR NODE          (gpt-4o)             │            │ │
│   │  │  • Receives schema + user question                  │            │ │
│   │  │  • Picks engine: "sql" | "pandas" | "direct"        │            │ │
│   │  │  • Refines/clarifies the query                      │            │ │
│   │  └──────────┬──────────────────────────────────────────┘            │ │
│   │             │                                                        │ │
│   │      ┌──────┴──────┐                                                │ │
│   │      │             │                                                │ │
│   │   engine=         engine=                                           │ │
│   │  "sql"|"pandas"  "direct"                                          │ │
│   │      │             │                                                │ │
│   │      ▼             │                                                │ │
│   │  ┌───────────────────────────────────────────────────────┐         │ │
│   │  │  2. CODEGEN NODE               (gpt-4o)               │         │ │
│   │  │  • Generates pandas code OR SQL query from schema     │         │ │
│   │  │  • On retry: receives previous error and fixes code   │         │ │
│   │  └──────────────┬────────────────────────────────────────┘         │ │
│   │                 │                                                   │ │
│   │                 ▼                                                   │ │
│   │  ┌───────────────────────────────────────────────────────┐         │ │
│   │  │  3. EXECUTOR NODE                                     │         │ │
│   │  │  • pandas: exec() in local namespace with DataFrames  │         │ │
│   │  │  • sql:    pd.read_sql_query() on SQLite :memory:     │         │ │
│   │  │  • Returns compact result string                      │         │ │
│   │  └──────────────┬────────────────────────────────────────┘         │ │
│   │                 │                                                   │ │
│   │         ┌───────┴────────┐                                         │ │
│   │         │                │                                         │ │
│   │      success         error + retries left                          │ │
│   │   (or max retries)        │                                        │ │
│   │         │                ▼                                         │ │
│   │         │      ┌──────────────────────┐                           │ │
│   │         │      │  increment_retry     │                           │ │
│   │         │      │  (bump counter)      │──────► back to CODEGEN    │ │
│   │         │      └──────────────────────┘   (max 2 retries)         │ │
│   │         │                                                          │ │
│   │         ▼◄──────────────────────────────────────── (direct) ──────┤ │
│   │  ┌───────────────────────────────────────────────────────┐         │ │
│   │  │  4. INTERPRETER NODE           (gpt-4o-mini)          │         │ │
│   │  │  • Receives result string (never raw data)            │         │ │
│   │  │  • Produces natural-language answer with insights     │         │ │
│   │  └──────────────┬────────────────────────────────────────┘         │ │
│   │                 │                                                   │ │
│   │               END                                                  │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Responsibilities

| Agent | Model | Input | Output |
|---|---|---|---|
| **Orchestrator** | `gpt-4o` | Schema + user question | Engine choice, refined query |
| **Codegen** | `gpt-4o` | Schema + refined query (+ error on retry) | Pandas code or SQL string |
| **Executor** | _(no LLM)_ | Generated code | Compact result string |
| **Interpreter** | `gpt-4o-mini` | Question + result string | Natural-language answer |

---

## Data Flow

```
User Query
    │
    ▼
Schema only ──► Orchestrator ──► Codegen ──► Executor ──► Result string ──► Interpreter ──► Answer
                                  ▲               │
                                  └── retry ──────┘ (on error, max 2x)
```

**Key principle:** The LLM at no point sees raw CSV rows — only column names, data types, row counts, and a 5-row sample. All heavy computation happens locally via pandas/SQLite.

---

## Tech Stack

### Core Framework
| Library | Version | Purpose |
|---|---|---|
| **LangGraph** | `>=0.2.0` | Agent pipeline orchestration — defines nodes, edges, and conditional routing as a stateful graph |
| **LangChain** | `>=0.3.0` | LLM abstraction layer, message types (`SystemMessage`, `HumanMessage`, `AIMessage`) |
| **LangChain-OpenAI** | `>=0.2.0` | OpenAI model bindings (`ChatOpenAI`) |

### LLM Models
| Model | Used For |
|---|---|
| `gpt-4o` | Orchestrator (engine selection) and Codegen (code generation) — high-accuracy tasks |
| `gpt-4o-mini` | Interpreter (natural-language summary) — lower cost, lower-stakes task |

### Data Layer
| Library | Purpose |
|---|---|
| **pandas** | `>=2.0.0` — DataFrame storage, CSV parsing, result formatting, pandas code execution |
| **sqlite3** | Built-in Python — in-memory relational database mirroring all loaded CSVs for SQL queries |
| **SQLAlchemy** | `>=2.0.0` — Used by pandas `to_sql` / `read_sql_query` as the DB adapter |

### API Layer
| Library | Version | Purpose |
|---|---|
| **FastAPI** | `>=0.111.0` | Async REST API with automatic OpenAPI/Swagger docs |
| **Uvicorn** | `>=0.30.0` | ASGI server to run the FastAPI app |
| **Pydantic** | `>=2.0.0` | Request/response validation and serialisation |
| **python-multipart** | `>=0.0.9` | Multipart file upload parsing for CSV ingestion |

### Utilities
| Library | Purpose |
|---|---|
| **tabulate** | `>=0.9.0` — Formats DataFrames as readable markdown tables for LLM output |
| **python-dotenv** | Loads `OPENAI_API_KEY` and other secrets from a `.env` file |

---

## Project Structure

```
TMC-optimal/
├── graph.py              # LangGraph graph definition — nodes, edges, routing
├── state.py              # AgentState TypedDict (shared state across all nodes)
├── run_cli.py            # Interactive CLI entry point
│
├── agents/
│   └── nodes.py          # All 4 agent node functions + routing helpers
│
├── api/
│   └── server.py         # FastAPI app — /upload, /chat, /session endpoints
│
├── tools/
│   └── registry.py       # DataRegistry — CSV loading, schema exposure, code execution
│
└── data/                 # Sample CSV datasets
```

---

## Session Model (API)

Each REST API session is isolated:

```
sessions = {
  "<uuid>": {
    "registry": DataRegistry,   # holds DataFrames + SQLite connection for this session
    "messages": [...]           # chat history (HumanMessage / AIMessage)
  }
}
```

Sessions are stored **in-memory** on the server (no database persistence). Each `/upload` call creates or reuses a session by `session_id`.

---

## Security Considerations

- LLM-generated code is executed via `exec()` in the same Python process (pandas path) — no sandboxing. Suitable for trusted-user internal deployments.
- Raw data rows are never sent to the OpenAI API — only schema metadata and result summaries.
- CORS is configured to `allow_origins=["*"]` — restrict this for production deployments.
- API keys are loaded from environment variables / `.env` file, not hardcoded.
