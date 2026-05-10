# AutoML-Powered Learned Query Optimizer

> **Phase 1 — Data Collection & Foundation**

Traditional databases like PostgreSQL pick query plans using static, rule-based
cost models. These estimates can be wildly off for complex queries on
real-world data, leading to slow execution.

The long-term goal of this project is to **learn** a better cost / plan-selection
model from real execution traces, and eventually bring AutoML into the loop so
the system improves over time.

This repository currently implements **only Phase 1**: a clean pipeline that
runs SQL queries on PostgreSQL, captures their `EXPLAIN (ANALYZE, FORMAT JSON)`
output, and stores it as a dataset that future ML phases will consume.

---

## What Phase 1 does (and does *not*) do

✔ Set up a PostgreSQL schema with sample data
✔ Execute a configurable workload of SQL queries
✔ Capture full execution plans + estimated cost + actual runtime as JSON
✔ Persist everything under `data/raw/` for later feature extraction

✘ No ML models
✘ No APIs / web UI / microservices
✘ No plan rewriting yet

---

## Project structure

```
auto-ml-query-optimizer/
├── data/
│   └── raw/                 # JSON execution plans land here
├── scripts/
│   └── collect_data.py      # runs EXPLAIN ANALYZE and saves plans
├── db/
│   └── schema.sql           # tables + sample data + sample queries
├── config/
│   └── db_config.py         # DB connection settings (env-var overridable)
├── README.md
└── requirements.txt
```

---

## Prerequisites

- Python **3.9+**
- PostgreSQL **13+** running locally (or reachable over the network)

---

## Setup

### 1. Create a database

```bash
createdb automl_qo
```

### 2. Load the schema and sample data

```bash
psql -d automl_qo -f db/schema.sql
```

This creates `customers` (10,000 rows) and `orders` (100,000 rows) using
`generate_series`, plus a couple of helper indexes.

### 3. Install Python dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. (Optional) Configure connection settings

Defaults live in `config/db_config.py`. You can override any of them via
environment variables — no code changes required:

```
PGHOST       (default: localhost)
PGPORT       (default: 5432)
PGDATABASE   (default: automl_qo)
PGUSER       (default: postgres)
PGPASSWORD   (default: postgres)
```

---

## Run the data collector

```bash
python scripts/collect_data.py
```

You should see something like:

```
[i] Connecting to postgres://postgres@localhost:5432/automl_qo
[•] Running q01_customers_by_country ...
    saved -> q01_customers_by_country__a1b2c3d4.json  (est_cost=205.0, exec_ms=3.812)
...
[✓] Done. 7 succeeded, 0 failed.
```

Each run writes:

- One `*.json` file per query in `data/raw/` containing
  - The original SQL
  - The full `EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)` plan
  - A `summary` block with estimated cost, actual time, planning time, etc.
- A single appended line per query in `data/raw/_index.jsonl` for quick scanning.

---

## Output format (per file)

```json
{
  "query_id": "q03_join_customer_orders",
  "tag": "join",
  "sql": "SELECT ...",
  "sql_hash": "a1b2c3d4",
  "collected_at": "2026-05-05T18:00:00+00:00",
  "wall_time_ms": 12.345,
  "summary": {
    "estimated_total_cost": 1234.56,
    "estimated_rows": 8421,
    "actual_total_time_ms": 9.87,
    "actual_rows": 8390,
    "planning_time_ms": 0.412,
    "execution_time_ms": 10.901,
    "root_node_type": "Hash Join"
  },
  "plan": [ /* raw EXPLAIN JSON tree */ ]
}
```

---

## Adding more queries

Edit the `WORKLOAD` list in `scripts/collect_data.py` and add new entries:

```python
{
    "id":  "q08_my_new_query",
    "tag": "agg",
    "sql": "SELECT ...;",
}
```

Re-run the collector. Each query gets its own JSON file, keyed by `query_id`
plus a hash of its SQL, so re-collecting after edits keeps history rather
than overwriting it.

---

## What's next (future phases — not in this repo yet)

1. **Feature extraction** — flatten plan trees into ML-ready features
2. **Learned cost model** — train a regressor on `(features → actual_time_ms)`
3. **Plan selection** — choose between candidate plans using the learned model
4. **AutoML loop** — continuously retrain as the workload evolves
