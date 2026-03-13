"""
api/server.py — FastAPI server.

Endpoints:
  POST /upload            — upload CSVs into a session
  POST /chat              — query the data
  GET  /session/{id}      — inspect session (schemas + history)
  DELETE /session/{id}    — delete session
  GET  /health
"""

import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage


load_dotenv()  # Load environment variables from .env file
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import graph
from tools.registry import DataRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CSV Multi-Agent System (Query-First)",
    description="LangGraph agents that generate & execute code against your CSVs — raw data never enters the LLM context.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Session store ─────────────────────────────────────────────────────────────
# { session_id: { "registry": DataRegistry, "messages": [...] } }
sessions: dict[str, dict] = {}


def _get_session(sid: str) -> dict:
    if sid not in sessions:
        raise HTTPException(404, f"Session '{sid}' not found")
    return sessions[sid]


# ── Schemas ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    tables_loaded: list[str]
    schemas: dict

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    engine_used: str
    generated_code: str
    execution_result: str
    final_response: str
    retries: int

class SessionInfo(BaseModel):
    session_id: str
    tables: list[str]
    message_count: int
    schemas: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    session_id: str | None = None,
):
    sid = session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = {"registry": DataRegistry(), "messages": []}

    registry: DataRegistry = sessions[sid]["registry"]
    schemas = {}

    for f in files:
        if not f.filename.endswith(".csv"):
            raise HTTPException(400, f"'{f.filename}' is not a CSV")
        content = await f.read()
        try:
            schema = registry.load_csv(f.filename, content)
            schemas[schema["table_name"]] = schema
            logger.info("Loaded '%s' → table '%s' (%d rows)", f.filename, schema["table_name"], schema["row_count"])
        except Exception as e:
            raise HTTPException(422, f"Could not parse '{f.filename}': {e}")

    return UploadResponse(
        session_id=sid,
        tables_loaded=list(schemas.keys()),
        schemas=schemas,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = _get_session(req.session_id)
    registry: DataRegistry = session["registry"]

    if not registry.get_table_names():
        raise HTTPException(400, "No CSV files loaded. Call /upload first.")

    initial_state = {
        "user_query": req.message,
        "schemas": registry.get_schemas(),
        "_registry": registry,        # passed by reference — never serialised
        "engine": "",
        "refined_query": "",
        "reasoning": "",
        "generated_code": "",
        "execution_result": "",
        "execution_error": "",
        "final_response": "",
        "_retry_count": 0,
        "messages": [
            *session["messages"],
            HumanMessage(content=req.message),
        ],
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Graph error")
        raise HTTPException(500, f"Agent error: {e}")

    # Persist message history (exclude internal keys)
    session["messages"] = [m for m in result.get("messages", []) if hasattr(m, "content")]

    return ChatResponse(
        session_id=req.session_id,
        user_message=req.message,
        engine_used=result.get("engine", ""),
        generated_code=result.get("generated_code", ""),
        execution_result=result.get("execution_result", ""),
        final_response=result.get("final_response", ""),
        retries=result.get("_retry_count", 0),
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
def session_info(session_id: str):
    session = _get_session(session_id)
    registry: DataRegistry = session["registry"]
    return SessionInfo(
        session_id=session_id,
        tables=registry.get_table_names(),
        message_count=len(session["messages"]),
        schemas=registry.get_schemas(),
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _get_session(session_id)
    del sessions[session_id]
    return {"message": f"Session '{session_id}' deleted"}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}
