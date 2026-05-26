"""
setup_tpch.py
=============
Phase 2A: TPC-H data generation + PostgreSQL load.

Pipeline:
    1. Use DuckDB's bundled `tpch` extension to generate the standard
       TPC-H tables at a chosen scale factor (default SF=1).
    2. Export each table to CSV under  data/tpch/raw_data/  .
    3. Drop + recreate the TPC-H schema in PostgreSQL via
       db/tpch_schema.sql.
    4. Bulk-load each CSV into its PostgreSQL table using COPY ... FROM
       STDIN (fast, single-transaction-per-table).
    5. Run ANALYZE so the planner has accurate statistics before
       plan collection begins.

Why DuckDB?
-----------
The official TPC-H data generator is a C program (`dbgen`) that's
painful to build on Windows. DuckDB ships an in-memory implementation
of the same generator that produces *identical* row content. So we get
real, spec-compliant TPC-H data with a single `pip install duckdb`.

Usage:
    python scripts/setup_tpch.py                # SF=1 (default)
    python scripts/setup_tpch.py --sf 0.1       # smaller (faster)
    python scripts/setup_tpch.py --sf 1 --skip-gen  # reuse existing CSVs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

SCHEMA_FILE = PROJECT_ROOT / "db" / "tpch_schema.sql"
DATA_DIR    = PROJECT_ROOT / "data" / "tpch" / "raw_data"

# TPC-H tables in dependency / load order (parents first).
TPCH_TABLES: list[str] = [
    "region", "nation", "part", "supplier",
    "partsupp", "customer", "orders", "lineitem",
]


# ---------------------------------------------------------------------------
# Step 1+2: generate TPC-H data with DuckDB and export to CSV
# ---------------------------------------------------------------------------
def generate_tpch_csvs(scale_factor: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Generating TPC-H data at SF={scale_factor} via DuckDB ...")

    t0 = time.perf_counter()
    con = duckdb.connect(":memory:")
    con.execute("INSTALL tpch;")
    con.execute("LOAD tpch;")
    con.execute(f"CALL dbgen(sf={scale_factor});")

    for table in TPCH_TABLES:
        csv_path = out_dir / f"{table}.csv"
        # Use HEADER + DELIMITER '|' to mirror the canonical *.tbl format
        # (TPC-H uses '|' as its native field separator).
        con.execute(
            f"COPY {table} TO '{csv_path.as_posix()}' "
            f"(HEADER, DELIMITER '|', FORMAT csv)"
        )
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"    exported {table:<10} -> {csv_path.name}  ({size_mb:.1f} MB)")

    con.close()
    print(f"[✓] DuckDB generation finished in {time.perf_counter() - t0:.1f}s\n")


# ---------------------------------------------------------------------------
# Step 3: (re)create schema in PostgreSQL
# ---------------------------------------------------------------------------
def apply_schema(conn) -> None:
    print(f"[i] Applying schema from {SCHEMA_FILE.relative_to(PROJECT_ROOT)} ...")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[✓] Schema created.\n")


# ---------------------------------------------------------------------------
# Step 4: bulk-load CSVs via COPY ... FROM STDIN
# ---------------------------------------------------------------------------
def copy_table(conn, table: str, csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path} — run without --skip-gen first")

    print(f"[•] Loading {table:<10} from {csv_path.name} ...", end="", flush=True)
    t0 = time.perf_counter()
    # Binary mode: psycopg2's copy_expert wants a file-like object, and on
    # Windows opening in text mode would translate line endings, which
    # corrupts the byte counts COPY expects.
    with csv_path.open("rb") as f, conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '|')",
            f,
        )
    conn.commit()
    print(f"  done in {time.perf_counter() - t0:.1f}s")


def load_all(conn, data_dir: Path) -> None:
    print("[i] Bulk-loading CSVs into PostgreSQL ...")
    for table in TPCH_TABLES:
        copy_table(conn, table, data_dir / f"{table}.csv")
    print()


# ---------------------------------------------------------------------------
# Step 5: refresh planner statistics
# ---------------------------------------------------------------------------
def analyze_all(conn) -> None:
    print("[i] Running ANALYZE on all TPC-H tables ...")
    with conn.cursor() as cur:
        for table in TPCH_TABLES:
            cur.execute(f"ANALYZE {table};")
    conn.commit()
    print("[✓] Statistics refreshed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPC-H setup: generate + load.")
    p.add_argument("--sf", type=float, default=1.0,
                   help="TPC-H scale factor (default: 1.0)")
    p.add_argument("--skip-gen", action="store_true",
                   help="skip CSV generation; reuse files already in data/tpch/raw_data/")
    p.add_argument("--skip-load", action="store_true",
                   help="generate CSVs only; do not touch PostgreSQL")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_gen:
        generate_tpch_csvs(args.sf, DATA_DIR)
    else:
        print("[i] --skip-gen: reusing existing CSVs in", DATA_DIR)

    if args.skip_load:
        print("[i] --skip-load: stopping before PostgreSQL load. Done.")
        return 0

    print(
        f"[i] Connecting to postgres://{DB_CONFIG['user']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(f"[!] Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        return 1

    try:
        apply_schema(conn)
        load_all(conn, DATA_DIR)
        analyze_all(conn)
    finally:
        conn.close()

    print("[✓] TPC-H setup complete. You can now run:")
    print("    python scripts/collect_tpch_plans.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
