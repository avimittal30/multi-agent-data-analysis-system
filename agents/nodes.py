"""
agents/nodes.py — LangGraph node functions.

Query-first flow:
  orchestrator → codegen → executor → [retry?] → interpreter
"""
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import AgentState

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MAX_RETRIES = 2   # how many times executor retries on error


# ─────────────────────────────────────────────────────────────────────────────
# 1. ORCHESTRATOR — sees schema only, picks engine + cleans query
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """\
You are the orchestrator of a query-first data analysis system.

You receive:
- A user question
- Database schemas (table names, columns, dtypes, row counts) — NO raw data

Your job: decide how to answer and clean up the query.

Engines available:
- "sql"    : best for aggregations, GROUP BY, JOINs across tables, filtering
- "pandas" : best for reshaping, complex transforms, multi-step logic, ML-style ops;
             REQUIRED when the user asks for a chart, graph, plot, or visualization
- "direct" : use ONLY if the question can be answered from schema alone
             (e.g. "how many columns?", "what tables do I have?")

IMPORTANT: If the user's question involves visualizing, charting, graphing, or plotting,
you MUST choose "pandas" — SQL cannot produce charts.

Respond ONLY with valid JSON, no markdown:
{
  "engine": "sql" | "pandas" | "direct",
  "refined_query": "<precise, unambiguous version of the user question>",
  "reasoning": "<one sentence>",
  "direct_answer": "<answer if engine=direct, else null>"
}"""


def orchestrator_node(state: AgentState) -> dict:
    registry = state["_registry"]
    schema_prompt = registry.get_schema_prompt()

    response = _llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=f"Schema:\n{schema_prompt}\n\nUser question: {state['user_query']}"),
    ])

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        plan = {"engine": "sql", "refined_query": state["user_query"], "reasoning": "fallback", "direct_answer": None}

    logger.info("Orchestrator → engine=%s | %s", plan.get("engine"), plan.get("reasoning"))

    updates = {
        "engine": plan.get("engine", "sql"),
        "refined_query": plan.get("refined_query", state["user_query"]),
        "reasoning": plan.get("reasoning", ""),
        "execution_result": "",
        "execution_error": "",
        "generated_code": "",
    }

    # Short-circuit: answer directly without code execution
    if plan.get("engine") == "direct" and plan.get("direct_answer"):
        updates["execution_result"] = plan["direct_answer"]

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# 2. CODE GENERATOR — writes pandas or SQL given schema + query
# ─────────────────────────────────────────────────────────────────────────────

PANDAS_CODEGEN_SYSTEM = """\
You are a pandas code generator. Write Python code to answer the user's question.

Rules:
- You have access to a dict `dfs` where keys are table names and values are DataFrames.
- Each table is ALSO available as a top-level variable (e.g. `sales`, `inventory`).
- Assign your final answer to a variable named `result`.
- `result` must be a DataFrame, Series, scalar, or list — NOT a print statement.
- Import nothing — pandas is available as `pd`, matplotlib.pyplot as `plt`.
- Write clean, efficient code. No comments needed.
- Do NOT read any files. Data is already loaded.
- Date columns originally in 'MMM-YY' format (e.g. 'Apr-25') have been pre-converted to
  datetime64 — sort them with `.sort_values()` directly; no manual conversion needed.
  To display them as 'Apr-25' again use `.dt.strftime('%b-%y')`.

For charts:
- Use matplotlib: create a figure with `fig, ax = plt.subplots(...)`, draw on `ax`,
  call `plt.tight_layout()`, then assign `result = fig`. Do NOT call `plt.show()`.

Example (table):
  result = sales.groupby("region")["revenue"].sum().reset_index()

Example (chart):
  df = sales.sort_values("month")
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.bar(df["month"].dt.strftime("%b-%y"), df["revenue"])
  ax.set_title("Revenue by Month")
  plt.tight_layout()
  result = fig
"""

SQL_CODEGEN_SYSTEM = """\
You are a SQL code generator for SQLite. Write a single SQL query to answer the question.

Rules:
- Output ONLY the SQL statement — no markdown, no explanation, no semicolons at end.
- Use only the tables and columns listed in the schema.
- SQLite syntax only (no window functions unavailable in SQLite 3.x).
- For string comparisons use LIKE or LOWER() for case-insensitivity.
- DATE columns (originally 'MMM-YY' strings) are stored as ISO-8601 text ('YYYY-MM-DD').
  A plain ORDER BY on these columns sorts chronologically — no special conversion needed.

Example:
  SELECT region, SUM(revenue) as total FROM sales GROUP BY region ORDER BY total DESC
"""


def codegen_node(state: AgentState) -> dict:
    if state.get("engine") == "direct":
        return {}   # no code needed

    registry = state["_registry"]

    if state["engine"] == "pandas":
        schema_prompt = registry.get_schema_prompt()
        system = PANDAS_CODEGEN_SYSTEM
        user_msg = f"Schema:\n{schema_prompt}\n\nQuestion: {state['refined_query']}"
    else:
        schema_prompt = registry.get_sql_schema_prompt()
        system = SQL_CODEGEN_SYSTEM
        user_msg = f"Schema:\n{schema_prompt}\n\nQuestion: {state['refined_query']}"

    # On retry: include the previous error so LLM can fix it
    if state.get("execution_error"):
        user_msg += f"\n\nPrevious attempt failed with error:\n{state['execution_error']}\n\nFix the code."

    response = _llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    code = response.content.strip()
    # Strip markdown fences if LLM wrapped it anyway
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])

    logger.info("CodeGen [%s]:\n%s", state["engine"], code)
    return {"generated_code": code.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXECUTOR — runs the generated code, returns compact result string
# ─────────────────────────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> dict:
    if state.get("engine") == "direct":
        return {}   # result already set by orchestrator

    if not state.get("generated_code"):
        return {"execution_error": "No code was generated."}

    registry = state["_registry"]

    if state["engine"] == "pandas":
        result, error = registry.execute_pandas(state["generated_code"])
    else:
        result, error = registry.execute_sql(state["generated_code"])

    if error:
        logger.warning("Executor error: %s", error[:300])
        return {"execution_error": error, "execution_result": ""}
    else:
        logger.info("Executor success. Result preview: %s", result[:200])
        return {"execution_result": result, "execution_error": ""}


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERPRETER — receives only the result, forms the final natural-language answer
# ─────────────────────────────────────────────────────────────────────────────

INTERPRETER_SYSTEM = """\
You are a data analyst communicating results to a business user.

You receive:
- The original user question
- The query result (a compact table or scalar — NOT raw CSV data)

Your job: interpret the result clearly and concisely.
- Lead with the direct answer to the question.
- Point out the most important numbers or patterns.
- Use markdown tables if the result is tabular.
- Add 1-2 sentence insight or recommendation where relevant.
- Be concise — no padding."""


def interpreter_node(state: AgentState) -> dict:
    query = state["user_query"]
    result = state.get("execution_result", "")
    error = state.get("execution_error", "")

    if error and not result:
        # Execution ultimately failed after retries
        final = (
            f"I was unable to execute the query for your question.\n\n"
            f"**Question:** {query}\n\n"
            f"**Error:**\n```\n{error[:600]}\n```\n\n"
            f"Please check that your CSV columns match what was expected."
        )
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
        }

    if state.get("engine") == "direct":
        # Orchestrator answered directly from schema
        final = result
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
        }

    response = _llm_fast.invoke([
        SystemMessage(content=INTERPRETER_SYSTEM),
        HumanMessage(content=f"Question: {query}\n\nQuery result:\n{result}"),
    ])

    final = response.content
    logger.info("Interpreter complete.")
    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers (used by graph.py)
# ─────────────────────────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """After executor: retry codegen if there was an error, else interpret."""
    error = state.get("execution_error", "")
    retries = state.get("_retry_count", 0)
    if error and retries < MAX_RETRIES:
        logger.info("Retrying code generation (attempt %d)", retries + 1)
        return "retry"
    return "interpret"


def increment_retry(state: AgentState) -> dict:
    """Bump retry counter before looping back to codegen."""
    return {"_retry_count": state.get("_retry_count", 0) + 1}
