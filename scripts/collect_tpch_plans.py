"""
collect_tpch_plans.py
=====================
Phase 2A plan-collection pipeline.

For every TPC-H query in  db/tpch_queries.sql  this script captures
multiple execution plans under different optimizer configurations:

    variant         optimizer knobs set per session
    --------------  -------------------------------
    default         (all join methods enabled)
    no_hashjoin     SET enable_hashjoin  = off
    no_mergejoin    SET enable_mergejoin = off
    no_nestloop    SET enable_nestloop  = off

Each (query, variant) pair becomes one JSON file in
   data/tpch/plans/{query_id}__{variant}__{sql_hash}.json
and one row in
   data/tpch/plans/_index.jsonl

This multi-variant capture is the whole point of Phase 2A: by forcing
PostgreSQL to consider non-default join strategies on the *same*
query, we build a dataset where many plan shapes exist per query —
exactly the signal a learned cost model needs to learn "plan A is
faster than plan B for this query".

Helper functions (short_hash, extract_summary) are reused from the
Phase 1 collector so plan records have a consistent schema across
the project.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.errors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.db_config import DB_CONFIG          # noqa: E402
from collect_data import short_hash, extract_summary  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUERIES_FILE = PROJECT_ROOT / "db" / "tpch_queries.sql"
PLANS_DIR    = PROJECT_ROOT / "data" / "tpch" / "plans"
INDEX_FILE   = PLANS_DIR / "_index.jsonl"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Optimizer variants we want plans for
# ---------------------------------------------------------------------------
# Note: setting `enable_<join> = off` only *discourages* a join method;
# PostgreSQL will still use it if no other legal plan exists. That's a
# feature, not a bug — the resulting plan tells us "this is the best
# plan PG can build while penalising hash joins", which is exactly
# what we want to learn from.
VARIANTS: dict[str, list[str]] = {
    "default":      [],
    "no_hashjoin":  ["SET enable_hashjoin  = off"],
    "no_mergejoin": ["SET enable_mergejoin = off"],
    "no_nestloop":  ["SET enable_nestloop  = off"],
}

EXPLAIN_PREFIX = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "

# Per-statement timeout (ms). TPC-H queries with bad plan choices can
# run for tens of minutes — we'd rather record the timeout than hang.
STATEMENT_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes


# ---------------------------------------------------------------------------
# Query file parser
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(
    r"^--\s*@QUERY:\s*(?P<id>\S+)\s*\|\s*tag:\s*(?P<tag>.+?)\s*$",
    re.MULTILINE,
)


def parse_queries(path: Path) -> list[dict]:
    """
    Split tpch_queries.sql on the `-- @QUERY: <id> | tag: <tags>` markers.
    Returns a list of {"id", "tag", "sql"} dicts in file order.
    """
    text = path.read_text(encoding="utf-8")

    headers = list(HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"No '-- @QUERY:' markers found in {path}")

    queries: list[dict] = []
    for i, m in enumerate(headers):
        start = m.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body  = text[start:end].strip()
        # Strip any trailing leading-comment lines that aren't part of the SQL.
        # We keep everything else verbatim so the planner sees exactly what
        # the analyst wrote.
        queries.append({
            "id":  m.group("id"),
            "tag": m.group("tag").strip(),
            "sql": body,
        })
    return queries


# ---------------------------------------------------------------------------
# One (query, variant) measurement
# ---------------------------------------------------------------------------
def collect_one(cur, query: dict, variant_name: str, settings: list[str]) -> dict | None:
    """
    Run EXPLAIN (ANALYZE, ...) on `query` under the given optimizer
    `settings`. Returns the JSON record on success, or None on timeout /
    error (which has already been logged).
    """
    sql = query["sql"].rstrip().rstrip(";")  # COPY/EXPLAIN can't take trailing ';'
    label = f"{query['id']}/{variant_name}"

    cur.execute("RESET ALL;")
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
    for stmt in settings:
        cur.execute(stmt + ";")

    print(f"[•] Running {label:<28} ...", end="", flush=True)

    wall_start = time.perf_counter()
    try:
        cur.execute(EXPLAIN_PREFIX + sql)
        plan_json = cur.fetchone()[0]
    except psycopg2.errors.QueryCanceled:
        print(f"  TIMEOUT (>{STATEMENT_TIMEOUT_MS/1000:.0f}s)")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc.__class__.__name__}: {exc}")
        return None
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    summary = extract_summary(plan_json)
    record = {
        "query_id":      query["id"],
        "variant":       variant_name,
        "variant_knobs": settings,
        "tag":           query["tag"],
        "sql":           sql,
        "sql_hash":      short_hash(sql),
        "collected_at":  datetime.now(timezone.utc).isoformat(),
        "wall_time_ms":  round(wall_ms, 3),
        "summary":       summary,
        "plan":          plan_json,
    }

    out_path = (
        PLANS_DIR
        / f"{query['id']}__{variant_name}__{record['sql_hash']}.json"
    )
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"  est={summary['estimated_total_cost']!s:>10}  "
        f"exec_ms={summary['execution_time_ms']!s:>10}  "
        f"-> {out_path.name}"
    )
    return record


def append_index(record: dict) -> None:
    line = {
        "query_id":      record["query_id"],
        "variant":       record["variant"],
        "tag":           record["tag"],
        "sql_hash":      record["sql_hash"],
        "collected_at":  record["collected_at"],
        **record["summary"],
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    queries = parse_queries(QUERIES_FILE)
    print(f"[i] Loaded {len(queries)} TPC-H queries from "
          f"{QUERIES_FILE.relative_to(PROJECT_ROOT)}")
    print(f"[i] Variants: {', '.join(VARIANTS.keys())}")
    print(f"[i] Per-query timeout: {STATEMENT_TIMEOUT_MS/1000:.0f}s")
    print(
        f"[i] Connecting to postgres://{DB_CONFIG['user']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}\n"
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
            for query in queries:
                for variant_name, settings in VARIANTS.items():
                    record = collect_one(cur, query, variant_name, settings)
                    if record is None:
                        failed += 1
                        continue
                    append_index(record)
                    ok += 1
    finally:
        conn.close()

    total = len(queries) * len(VARIANTS)
    print(f"\n[✓] Done. {ok}/{total} plans captured, {failed} failed/timed-out.")
    print(f"[✓] Plans written to: {PLANS_DIR}")
    print(f"[✓] Index file:       {INDEX_FILE}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
