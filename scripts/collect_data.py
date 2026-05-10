"""
collect_data.py
================
Phase 1 data-collection pipeline for the
AutoML-Powered Learned Query Optimizer.

For each query in the workload it:
    1. Runs   EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)  in PostgreSQL
    2. Captures the JSON execution plan
    3. Extracts headline metrics (estimated cost, actual time, rows, ...)
    4. Persists everything as JSON files inside  data/raw/
    5. Appends a one-line summary record to    data/raw/_index.jsonl

The files written here become the training set for later ML phases.
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Make `config/` importable regardless of where the script is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
RAW_DIR    = PROJECT_ROOT / "data" / "raw"
INDEX_FILE = RAW_DIR / "_index.jsonl"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Workload: a small, varied set of queries that exercise different plan
# shapes (seq scan, index scan, hash join, aggregate, sort, ...).
# Add more queries here as the project grows.
# ---------------------------------------------------------------------------
WORKLOAD: list[dict] = [
    {
        "id":  "q01_customers_by_country",
        "tag": "filter",
        "sql": "SELECT * FROM customers WHERE country = 'India';",
    },
    {
        "id":  "q02_orders_recent",
        "tag": "filter+index",
        "sql": "SELECT * FROM orders WHERE order_date > DATE '2024-01-01';",
    },
    {
        "id":  "q03_join_customer_orders",
        "tag": "join",
        "sql": """
            SELECT c.customer_id, c.name, o.order_id, o.amount
            FROM customers c
            JOIN orders   o ON c.customer_id = o.customer_id
            WHERE c.country = 'USA'
              AND o.amount > 500;
        """,
    },
    {
        "id":  "q04_agg_total_per_country",
        "tag": "join+agg",
        "sql": """
            SELECT c.country,
                   COUNT(o.order_id) AS num_orders,
                   SUM(o.amount)     AS total_revenue
            FROM customers c
            JOIN orders   o ON c.customer_id = o.customer_id
            GROUP BY c.country
            ORDER BY total_revenue DESC;
        """,
    },
    {
        "id":  "q05_top_customers",
        "tag": "join+agg+sort+limit",
        "sql": """
            SELECT c.customer_id, c.name, SUM(o.amount) AS spent
            FROM customers c
            JOIN orders   o ON c.customer_id = o.customer_id
            WHERE o.status = 'DELIVERED'
            GROUP BY c.customer_id, c.name
            ORDER BY spent DESC
            LIMIT 25;
        """,
    },
    {
        "id":  "q06_status_breakdown",
        "tag": "agg",
        "sql": """
            SELECT status, COUNT(*) AS n, AVG(amount) AS avg_amt
            FROM orders
            GROUP BY status;
        """,
    },
    {
        "id":  "q07_self_subquery",
        "tag": "subquery",
        "sql": """
            SELECT name
            FROM customers
            WHERE customer_id IN (
                SELECT customer_id
                FROM orders
                WHERE amount > 900
            );
        """,
    },
]

EXPLAIN_PREFIX = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def extract_summary(plan_json: list) -> dict:
    """
    EXPLAIN (FORMAT JSON) returns a list with a single dict that looks like:
        [{
            "Plan": {... root node ...},
            "Planning Time": 0.123,
            "Execution Time": 4.567,
            ...
        }]
    Pull the headline metrics out so they're easy to query later.
    """
    root         = plan_json[0]
    plan_node    = root.get("Plan", {})
    return {
        "estimated_total_cost": plan_node.get("Total Cost"),
        "estimated_startup_cost": plan_node.get("Startup Cost"),
        "estimated_rows":        plan_node.get("Plan Rows"),
        "actual_total_time_ms":  plan_node.get("Actual Total Time"),
        "actual_rows":           plan_node.get("Actual Rows"),
        "planning_time_ms":      root.get("Planning Time"),
        "execution_time_ms":     root.get("Execution Time"),
        "root_node_type":        plan_node.get("Node Type"),
    }


def collect_one(cur, query: dict) -> dict:
    sql = query["sql"].strip()
    print(f"[•] Running {query['id']} ...", flush=True)

    wall_start = time.perf_counter()
    cur.execute(EXPLAIN_PREFIX + sql)
    plan_json = cur.fetchone()[0]
    wall_ms   = (time.perf_counter() - wall_start) * 1000.0

    summary = extract_summary(plan_json)

    record = {
        "query_id":     query["id"],
        "tag":          query.get("tag"),
        "sql":          sql,
        "sql_hash":     short_hash(sql),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round(wall_ms, 3),
        "summary":      summary,
        "plan":         plan_json,
    }

    out_path = RAW_DIR / f"{query['id']}__{record['sql_hash']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(
        f"    saved -> {out_path.name}  "
        f"(est_cost={summary['estimated_total_cost']}, "
        f"exec_ms={summary['execution_time_ms']})"
    )
    return record


def append_index(record: dict) -> None:
    """Append a compact, one-line summary to data/raw/_index.jsonl."""
    line = {
        "query_id":     record["query_id"],
        "tag":          record["tag"],
        "sql_hash":     record["sql_hash"],
        "collected_at": record["collected_at"],
        **record["summary"],
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(
        f"[i] Connecting to postgres://{DB_CONFIG['user']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(f"[!] Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        return 1

    conn.autocommit = True
    ok, failed = 0, 0
    try:
        with conn.cursor() as cur:
            for query in WORKLOAD:
                try:
                    record = collect_one(cur, query)
                    append_index(record)
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"[!] {query['id']} failed: {exc}", file=sys.stderr)
    finally:
        conn.close()

    print(f"\n[✓] Done. {ok} succeeded, {failed} failed.")
    print(f"[✓] Plans written to: {RAW_DIR}")
    print(f"[✓] Index file:       {INDEX_FILE}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
