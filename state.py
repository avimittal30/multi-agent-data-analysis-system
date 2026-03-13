"""
state.py — Shared LangGraph state.

Query-first design:
  1. Orchestrator sees schema only (column names + dtypes + row count)
  2. Code-gen agent writes pandas / SQL code against that schema
  3. Executor runs the code and returns a compact result
  4. Interpreter receives only the result (not raw data) and forms the answer
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Inputs ───────────────────────────────────────────────────────────────
    user_query: str

    # Schema-only context (never raw rows)
    # { filename: {"columns": [...], "dtypes": {...}, "row_count": int, "sample_rows": [...5 rows...]} }
    schemas: dict[str, Any]

    # ── Runtime registry (holds DataFrames + SQLite connection) ──────────────
    _registry: Any        # tools.registry.DataRegistry instance

    # ── Orchestrator ─────────────────────────────────────────────────────────
    engine: str           # "pandas" | "sql" | "direct"
    refined_query: str    # clarified query passed to code-gen
    reasoning: str

    # ── Code generation ───────────────────────────────────────────────────────
    generated_code: str   # pandas or SQL string produced by LLM

    # ── Execution ─────────────────────────────────────────────────────────────
    execution_result: str   # compact string result (e.g. markdown table, scalar)
    execution_error: str    # non-empty if execution failed

    # ── Retry counter ─────────────────────────────────────────────────────────
    _retry_count: int

    # ── Interpretation ────────────────────────────────────────────────────────
    final_response: str

    # ── Chat history ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
