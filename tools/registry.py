"""
tools/registry.py — Central data registry.

Responsibilities:
  - Load CSV files into both a pandas DataFrame store and a SQLite database
  - Expose SCHEMA ONLY to the LLM (column names, dtypes, row counts, 5-row sample)
  - Provide execute_pandas() and execute_sql() for safe code execution
"""
import os
import re
import io
import sqlite3
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# Matches month strings like "Apr-25", "Dec-24"
_MONTH_YY_RE = re.compile(r'^[A-Za-z]{3}-\d{2}$')


def _try_parse_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'MMM-YY' string columns (e.g. 'Apr-25') to datetime64 so that
    ORDER BY / sort_values() produces chronological order, not alphabetical."""
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if len(sample) > 0 and sample.apply(lambda v: bool(_MONTH_YY_RE.match(str(v)))).all():
            try:
                df[col] = pd.to_datetime(df[col], format="%b-%y")
            except Exception:
                pass
    return df


class DataRegistry:
    """Holds loaded DataFrames and a shared SQLite connection."""

    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}        # table_name → DataFrame
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_csv(self, filename: str, content: bytes | str | Path) -> dict:
        """Load a CSV from bytes, string content, or a file path."""
        if isinstance(content, Path):
            df = pd.read_csv(content)
        elif isinstance(content, bytes):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_csv(io.StringIO(content))

        # Auto-parse 'MMM-YY' month columns so date sorting works correctly
        df = _try_parse_month_columns(df)

        # Sanitise table name: lowercase, spaces → underscores, strip .csv
        table_name = re.sub(r"[^\w]", "_", Path(filename).stem.lower())
        self._frames[table_name] = df

        # Persist into SQLite so SQL agent can query it
        df.to_sql(table_name, self._conn, if_exists="replace", index=False)

        return self._build_schema(table_name, df)

    def get_table_names(self) -> list[str]:
        return list(self._frames.keys())

    # ── Schema (what the LLM is allowed to see) ───────────────────────────────

    def get_schemas(self) -> dict[str, Any]:
        """Return schema-only dict for all loaded tables."""
        return {name: self._build_schema(name, df) for name, df in self._frames.items()}

    def get_schema_prompt(self) -> str:
        """
        Human-readable schema block for LLM prompts.
        Contains NO data rows — only structure + 5-row sample for type inference.
        Low-cardinality text columns include their distinct values so filters are exact.
        """
        parts = []
        for name, info in self.get_schemas().items():
            df = self._frames[name]
            col_lines = []
            for c, t in info["dtypes"].items():
                line = f"    {c} ({t})"
                # Append distinct values for low-cardinality text/category columns
                if str(t) in ("object", "category", "string"):
                    vals = df[c].dropna().unique()
                    if 0 < len(vals) <= 20:
                        vals_str = ", ".join(repr(str(v)) for v in sorted(vals, key=str))
                        line += f"  — values: {vals_str}"
                col_lines.append(line)
            sample = pd.DataFrame(info["sample_rows"]).to_markdown(index=False)
            parts.append(
                f"Table: {name}\n"
                f"Rows: {info['row_count']:,}\n"
                f"Columns:\n" + "\n".join(col_lines) + "\n"
                f"Sample (5 rows):\n{sample}"
            )
        return "\n\n---\n\n".join(parts)

    def get_sql_schema_prompt(self) -> str:
        """CREATE TABLE statements with distinct value hints for text columns."""
        parts = []
        for name, df in self._frames.items():
            cols = []
            comments = []
            for col, dtype in df.dtypes.items():
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "REAL"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "DATE"
                else:
                    sql_type = "TEXT"
                cols.append(f"    {col} {sql_type}")
                # Add distinct value hint for low-cardinality TEXT columns
                if sql_type == "TEXT":
                    vals = df[col].dropna().unique()
                    if 0 < len(vals) <= 20:
                        vals_str = ", ".join(repr(str(v)) for v in sorted(vals, key=str))
                        comments.append(f"-- {col} values: {vals_str}")
            table_def = f"CREATE TABLE {name} (\n" + ",\n".join(cols) + "\n);"
            if comments:
                table_def += "\n" + "\n".join(comments)
            parts.append(table_def)
        return "\n\n".join(parts)

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute_pandas(self, code: str) -> tuple[str, str]:
        """
        Execute LLM-generated pandas code.
        The code has access to a `dfs` dict: { table_name: DataFrame }.
        Returns (result_string, error_string).
        """
        # Inject dataframes into execution namespace
        local_ns = {"dfs": dict(self._frames), "pd": pd}
        if _MPL_AVAILABLE:
            local_ns["plt"] = plt
        # Also expose each table directly by name for convenience
        local_ns.update(self._frames)

        try:
            exec(textwrap.dedent(code), local_ns)  # noqa: S102
            result = local_ns.get("result", None)
            if result is None:
                # Try to find any DataFrame or scalar assigned in the code
                for key, val in local_ns.items():
                    if key.startswith("_") or key in ("dfs", "pd"):
                        continue
                    if key not in self._frames:
                        result = val
                        break
            return _format_result(result), ""
        except Exception:
            return "", traceback.format_exc()

    def execute_sql(self, sql: str) -> tuple[str, str]:
        """
        Execute LLM-generated SQL against the in-memory SQLite database.
        Returns (result_string, error_string).
        """
        try:
            df = pd.read_sql_query(sql, self._conn)
            return _format_result(df), ""
        except Exception:
            return "", traceback.format_exc()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_schema(table_name: str, df: pd.DataFrame) -> dict:
        return {
            "table_name": table_name,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
            "numeric_summary": df.describe(include="number").to_dict() if not df.select_dtypes("number").empty else {},
        }


# ── Result formatter ──────────────────────────────────────────────────────────

def _format_result(result: Any, max_rows: int = 50) -> str:
    """Convert any execution result to a compact, LLM-readable string."""
    if result is None:
        return "Query returned no output. Make sure your code assigns to `result`."
    # matplotlib Figure → save to a temp PNG and return the path
    if _MPL_AVAILABLE and isinstance(result, plt.Figure):
        chart_path = os.path.join(tempfile.gettempdir(), "tmc_chart.png")
        result.savefig(chart_path, bbox_inches="tight", dpi=150)
        plt.close(result)
        return f"[CHART:{chart_path}]"
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Empty DataFrame (0 rows)."
        truncated = result.head(max_rows)
        out = truncated.to_markdown(index=False)
        if len(result) > max_rows:
            out += f"\n\n_(showing {max_rows} of {len(result):,} rows)_"
        return out
    if isinstance(result, pd.Series):
        return result.to_frame().to_markdown()
    # Scalar / list / dict
    return str(result)
