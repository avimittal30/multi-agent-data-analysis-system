"""
run_cli.py — Interactive CLI for the query-first multi-agent system.

Usage:
    python run_cli.py data/sales.csv data/inventory.csv
"""
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file
sys.path.insert(0, str(Path(__file__).parent))

from graph import graph
from tools.registry import DataRegistry
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.WARNING)   # quiet in CLI mode


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_cli.py file1.csv [file2.csv ...]")
        sys.exit(1)

    registry = DataRegistry()

    print("\n── Loading files ─────────────────────────────")
    for path_str in sys.argv[1:]:
        p = Path(path_str)
        if not p.exists():
            print(f"  ✗ Not found: {p}")
            continue
        info = registry.load_csv(p.name, p)
        print(f"  ✓ {p.name}  →  table '{info['table_name']}'  ({info['row_count']:,} rows, {len(info['columns'])} cols)")

    if not registry.get_table_names():
        print("No files loaded. Exiting.")
        sys.exit(1)

    print("\n── Schema ────────────────────────────────────")
    print(registry.get_schema_prompt())
    print("\nType your question ('quit' to exit).\n")

    messages = []

    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        state = {
            "user_query": query,
            "schemas": registry.get_schemas(),
            "_registry": registry,
            "engine": "",
            "refined_query": "",
            "reasoning": "",
            "generated_code": "",
            "execution_result": "",
            "execution_error": "",
            "final_response": "",
            "_retry_count": 0,
            "messages": [*messages, HumanMessage(content=query)],
        }

        result = graph.invoke(state)
        messages = [m for m in result.get("messages", []) if hasattr(m, "content")]

        print(f"\n[engine: {result.get('engine')}]  [retries: {result.get('_retry_count', 0)}]")
        if result.get("generated_code"):
            print(f"[code]\n{result['generated_code']}\n")
        raw = result.get("execution_result", "")
        if raw:
            print(f"[raw result]\n{raw[:500]}\n")
        # Auto-open chart if one was generated
        if raw.startswith("[CHART:"):
            import re as _re
            m = _re.search(r'\[CHART:(.+?)\]', raw)
            if m:
                import os as _os
                chart_path = m.group(1)
                print(f"[chart saved → {chart_path}]")
                _os.startfile(chart_path)
        print(f"Assistant >\n{result.get('final_response', '')}\n")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
