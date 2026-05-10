# Phase 1 — Deep Dive Reference

> A line-by-line, advanced-level walkthrough of the
> **AutoML-Powered Learned Query Optimizer** Phase 1 codebase.
> Use this as your single source of truth when you come back to the
> project later.

---

## 0. Mental model — what Phase 1 actually is

Before touching code, lock this picture in your head:

```
   ┌──────────────────────┐        ┌──────────────────────────┐
   │     PostgreSQL       │ <────► │  collect_data.py (Python)│
   │ (customers, orders)  │        │  - opens psycopg2 conn   │
   └──────────────────────┘        │  - runs EXPLAIN ANALYZE  │
                                   │  - parses JSON plan      │
                                   │  - writes per-query JSON │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
                                       data/raw/*.json
                                       data/raw/_index.jsonl
                                                │
                                                ▼
                                  (Phase 2: features + ML)
```

Phase 1 is a **dataset builder**. Nothing more, nothing less.
It exists so that Phase 2 has clean, structured (`features → cost/time`)
training data without you ever having to talk to PostgreSQL again at
training time.

The two design rules behind every file in this phase:

1. **Capture everything Postgres knows about the plan** — never throw
   information away at collection time, because we don't yet know which
   features the ML model will want.
2. **Make every record self-describing** — a single JSON file should be
   enough to reconstruct what query was run, when, against what plan
   shape, and what it cost. No external lookups required.

---

## 1. Directory layout — why each folder exists

```
auto-ml-query-optimizer/
├── data/
│   └── raw/                 # immutable raw outputs (one JSON per run)
├── scripts/
│   └── collect_data.py      # the pipeline entry point
├── db/
│   └── schema.sql           # reproducible database state
├── config/
│   └── db_config.py         # connection settings (env-overridable)
├── docs/
│   └── PHASE1_EXPLAINED.md  # ← this file
├── README.md
└── requirements.txt
```

A few non-obvious choices worth remembering:

- **`data/raw/` is sacred.** Anything written here is treated as
  immutable. Phase 2 will produce `data/processed/`, `data/features/`,
  etc., but those are derived; if they get corrupted you can always
  rebuild from `data/raw/`. This is the same pattern used by Kaggle
  pipelines, dbt, and most ML systems in production.
- **`config/` is its own folder, not a flag.** Even though we only have
  one config file today, separating "how to connect" from "what to do"
  means you can later add `config/workload.py`,
  `config/logging.py`, etc., without touching the collector.
- **`scripts/` (not `src/`) on purpose.** The code in here is an
  *executable pipeline*, not a reusable library. Calling the folder
  `scripts/` is a hint to future-you: "run these from the CLI; don't
  `import` them as a package."
- **`db/schema.sql` is checked in.** The database is part of the
  experiment. If you can't reproduce the schema and seed data, your
  collected plans are scientifically useless — different statistics
  ⇒ different plans ⇒ different costs.

---

## 2. `db/schema.sql` — the synthetic universe

### 2.1 Idempotency block

```sql
DROP TABLE IF EXISTS orders    CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
```

- Dropped in **child-first** order because `orders.customer_id`
  references `customers`. Reversing this would fail without `CASCADE`.
- `CASCADE` is added defensively in case future indexes / views /
  foreign keys hang off these tables.
- The whole file is now safe to re-run; you can iterate on the
  schema and just re-load with `psql -f`.

### 2.2 Table design

```sql
CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    country       TEXT NOT NULL,
    signup_date   DATE NOT NULL,
    age           INT  NOT NULL
);
```

- `SERIAL` = `INTEGER` + a sequence + `DEFAULT nextval(...)`. It's
  enough for 10k rows; for billions you'd use `BIGSERIAL`.
- All columns are `NOT NULL` to keep the planner's selectivity
  estimates simple — null fractions throw extra knobs into the cost
  formulas, which would muddy our learning signal.
- `country` is `TEXT` (not an enum) on purpose: enums force the planner
  to use exact equality stats only, while `TEXT` lets us later
  experiment with `LIKE` predicates and trigram indexes.

```sql
CREATE TABLE orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INT          NOT NULL REFERENCES customers(customer_id),
    order_date   DATE         NOT NULL,
    amount       NUMERIC(10,2) NOT NULL,
    status       TEXT         NOT NULL
);
```

- `REFERENCES customers(customer_id)` — gives the planner a foreign-key
  hint. PG 12+ uses FK information for **join cardinality** estimates
  (it knows the right side is unique), which directly shapes the plans
  we'll learn from.
- `NUMERIC(10,2)` — exact decimal. Using `FLOAT` would introduce
  rounding noise in `SUM(amount)` aggregates, making actual-row counts
  unstable across runs.

### 2.3 Seed data via `generate_series`

```sql
INSERT INTO customers (name, email, country, signup_date, age)
SELECT
    'Customer_' || g,
    'user_' || g || '@example.com',
    (ARRAY['USA','India','UK',...])[1 + (g % 8)],
    DATE '2018-01-01' + (g % 2000),
    18 + (g % 60)
FROM generate_series(1, 10000) AS g;
```

What `generate_series` is doing here:

- `generate_series(1, 10000)` is a **set-returning function** — it
  emits 10,000 rows in a single `SELECT`, much faster than 10,000
  individual `INSERT`s (one transaction, one plan, one tuple stream).
- `g % 8` and `g % 60` produce **deterministic, uniform** distributions
  across countries / ages. This is intentional: in Phase 1 we want a
  *known* statistical landscape so we can later check whether the
  planner's row estimates match reality.
- `DATE '2018-01-01' + (g % 2000)` — date arithmetic in days, giving a
  ~5.5-year spread.

Orders are similar but ten times bigger and reference `customer_id`s in
`[1, 10000]` so every customer has ~10 orders on average:

```sql
1 + (g % 10000)            -- customer fan-out
ROUND((random() * 1000)::NUMERIC, 2)   -- amount in [0, 1000]
(ARRAY[...])[1 + (g % 4)]              -- status, 4-way uniform
```

**Why `random()` for amount but `g % N` everywhere else?**
- Modulo gives perfect uniformity → predictable cardinality.
- `random()` for `amount` introduces realistic noise on a *non-key*
  column. This is what lets queries like `WHERE amount > 500` produce
  ~50 % selectivity that the planner has to *estimate* from histograms,
  not derive analytically.

### 2.4 Indexes — intentionally minimal

```sql
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date  ON orders(order_date);
```

- Two indexes on `orders`, none on `customers.country` / `amount`.
  This *forces* the planner into interesting choices:
  - Joins on `customer_id` can use either a hash join (full seq scan +
    hash) or an index nested loop. The planner will pick based on cost
    — exactly the decision we want to learn.
  - `WHERE amount > 500` has no index, so it's always a filter on a seq
    scan. That's a useful baseline for the model.
- We do **not** add an index per query, because then the planner would
  have only one obvious choice for everything and our dataset would be
  boring (all index scans, all the time).

### 2.5 `ANALYZE`

```sql
ANALYZE customers;
ANALYZE orders;
```

`ANALYZE` rebuilds `pg_statistic` (per-column histograms,
most-common-values lists, n_distinct estimates, correlation, ...).
Without this the planner uses pessimistic defaults and you'd see
totally different plans on the first run vs. later runs. We run it
explicitly so the dataset is **reproducible from a cold database**.

---

## 3. `config/db_config.py` — connection plumbing

```python
DB_CONFIG = {
    "host":     os.getenv("PGHOST",     "localhost"),
    "port":     int(os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PGDATABASE", "automl_qo"),
    "user":     os.getenv("PGUSER",     "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}
```

Three deliberate decisions:

1. **Env-var overrides with sane defaults.** This is the standard
   12-factor pattern. Local dev "just works"; CI / cloud can inject
   secrets without code changes. Notice we use the *same* names
   (`PGHOST`, `PGPORT`, ...) that `psql` and `libpq` already
   recognise — so the same environment can drive both.
2. **`int(os.getenv("PGPORT", "5432"))`.** Env vars are always strings;
   `psycopg2` wants `port` as an `int`. Casting at config time, not at
   connect time, means you fail fast with a clear `ValueError` if
   somebody sets `PGPORT=abc`.
3. **`get_dsn()` helper.** Some Postgres tooling (e.g. `psql`,
   SQLAlchemy URL builders, third-party drivers) prefers a DSN string
   over a kwargs dict. Building both formats from a single source of
   truth avoids drift.

```python
def get_dsn() -> str:
    return (
        f"host={DB_CONFIG['host']} "
        f"port={DB_CONFIG['port']} "
        f"dbname={DB_CONFIG['dbname']} "
        f"user={DB_CONFIG['user']} "
        f"password={DB_CONFIG['password']}"
    )
```

> ⚠️ **Production caveat:** `password=...` in a DSN ends up in process
> listings and logs. For real deployments, prefer a `~/.pgpass` file or
> mount the password as a secret.

---

## 4. `scripts/collect_data.py` — the pipeline, line by line

### 4.1 Imports and path bootstrap

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.db_config import DB_CONFIG
```

Why this dance instead of a normal package import?

- The folder isn't a Python package (no `__init__.py`, no
  `pyproject.toml`). Treating it as one would be over-engineering for
  Phase 1.
- `__file__` is the script's path; `.resolve()` converts symlinks and
  `..` segments to an absolute, canonical path. So no matter where you
  launch from — `python scripts/collect_data.py`,
  `python /abs/path/.../collect_data.py`, or via cron — the import
  still finds `config/`.
- Inserting at index `0` (not `append`) ensures our local config wins
  over any `config` package that might exist on the system path.

### 4.2 Output paths

```python
RAW_DIR    = PROJECT_ROOT / "data" / "raw"
INDEX_FILE = RAW_DIR / "_index.jsonl"
RAW_DIR.mkdir(parents=True, exist_ok=True)
```

- `parents=True` lets you check out the repo without `data/raw/`
  existing and still have the script work.
- `exist_ok=True` makes re-runs idempotent.
- The leading underscore in `_index.jsonl` is a convention: anything
  starting with `_` is metadata, not a data record. When Phase 2 globs
  `data/raw/*.json` it can safely skip files matching `_*`.

### 4.3 The `WORKLOAD` list

This is the **experiment design**, not just a list of queries. Each
entry maps to a specific plan shape we want represented in the dataset:

| ID | Tag | Plan shape we expect |
|----|-----|----------------------|
| `q01_customers_by_country` | `filter` | `Seq Scan` with text equality |
| `q02_orders_recent` | `filter+index` | `Index Scan` or `Bitmap Heap Scan` on `order_date` |
| `q03_join_customer_orders` | `join` | `Hash Join` (small build-side) |
| `q04_agg_total_per_country` | `join+agg` | `HashAggregate` over a join |
| `q05_top_customers` | `join+agg+sort+limit` | `Sort` + `Limit` after agg |
| `q06_status_breakdown` | `agg` | Pure aggregate, no join |
| `q07_self_subquery` | `subquery` | `Semi Join` or `Hash Join` w/ unique inner |

Why this matters for ML:

- A learned cost model is only as expressive as the plan-shape diversity
  in its training data. If 100 % of your training queries are Hash
  Joins, the model will silently never learn anything about Nested
  Loops.
- The `tag` field is your future grouping key for stratified train/test
  splits and per-shape error analysis.

Each entry is a plain dict (not a class) on purpose: easy to append,
easy to JSON-dump for reproducibility, easy to read into pandas later
with `pd.DataFrame(WORKLOAD)`.

### 4.4 The `EXPLAIN` prefix

```python
EXPLAIN_PREFIX = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "
```

Each option is doing something specific:

- **`ANALYZE`** — actually executes the query and records real
  per-node timing and row counts. Without it you'd only get the
  planner's *estimates*, which is exactly what we *don't* trust.
- **`BUFFERS`** — adds shared/local/temp block hit / read counts per
  node. This is the only way to detect cache-vs-disk effects, which
  matter enormously for runtime prediction.
- **`VERBOSE`** — adds `Output` columns, schema-qualified relation
  names, and function signatures. Cheap to capture, expensive to
  recover later if you want to know "which columns flowed through
  this node?".
- **`FORMAT JSON`** — gives us a structured tree we can `json.loads`
  directly, instead of the human-readable text format we'd have to
  regex-parse.

> ⚠️ **`ANALYZE` actually runs the query.** Don't point the collector
> at a production database, and don't `ANALYZE` an `INSERT`/`UPDATE`/
> `DELETE` unless you wrap it in `BEGIN; ... ROLLBACK;`. Phase 1's
> workload is read-only, so we're safe.

### 4.5 `short_hash`

```python
def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
```

- We hash the **SQL text**, not the query ID, so that if you edit
  `q03`'s SQL the hash changes and you keep both the old and new
  recordings on disk. This is critical for tracking how plan choices
  evolve as queries change.
- 8 hex chars = 32 bits. Birthday-collision probability with even a
  few thousand queries is astronomically low for our purposes; we're
  not using this for security.
- SHA-1 (vs MD5) — cryptographic strength is irrelevant here, but SHA-1
  is the default we already pay for in most stdlibs and produces a
  uniform hex string.

### 4.6 `extract_summary`

```python
root      = plan_json[0]
plan_node = root.get("Plan", {})
return {
    "estimated_total_cost":   plan_node.get("Total Cost"),
    "estimated_startup_cost": plan_node.get("Startup Cost"),
    "estimated_rows":         plan_node.get("Plan Rows"),
    "actual_total_time_ms":   plan_node.get("Actual Total Time"),
    "actual_rows":            plan_node.get("Actual Rows"),
    "planning_time_ms":       root.get("Planning Time"),
    "execution_time_ms":      root.get("Execution Time"),
    "root_node_type":         plan_node.get("Node Type"),
}
```

You need to internalise the JSON shape Postgres returns:

```
[                          ← outer list, always length 1 for one stmt
  {
    "Plan": {              ← root operator
      "Node Type": ...,
      "Startup Cost": ...,
      "Total Cost": ...,
      "Plan Rows": ...,
      "Actual Startup Time": ...,
      "Actual Total Time":  ...,
      "Actual Rows": ...,
      "Plans": [ ...child operators... ]   ← recursive
    },
    "Planning Time":  ...,
    "Execution Time": ...,
    "Triggers":       [ ... ]
  }
]
```

Important nuances:

- **`Total Cost` ≠ wall-clock seconds.** It's an abstract cost in PG's
  internal units (mostly tuned to "the cost of a sequential page
  read"). The whole point of Phase 2 is to learn a function from this
  abstract number (and other features) to *real* time.
- **`Actual Total Time` is per *loop*, in ms, including children.**
  For a node that ran with `Actual Loops = 5`, total wall time is
  `Actual Total Time × Actual Loops`. Phase-2 feature extraction
  must respect this.
- `.get(...)` (not `[...]`) everywhere — older PG versions and certain
  node types omit some fields. Returning `None` is better than
  crashing the whole batch.

The summary is duplicated info — it's already inside `plan` — but
having it pre-flattened means the eventual `_index.jsonl` is greppable
without a JSON parser, and `pandas.read_json("_index.jsonl",
lines=True)` immediately gives you a usable DataFrame for sanity
checks.

### 4.7 `collect_one`

```python
wall_start = time.perf_counter()
cur.execute(EXPLAIN_PREFIX + sql)
plan_json  = cur.fetchone()[0]
wall_ms    = (time.perf_counter() - wall_start) * 1000.0
```

- **`time.perf_counter()`** — monotonic, highest available resolution
  on your platform. We use it instead of `time.time()` so NTP clock
  adjustments can't make `wall_ms` negative.
- **`wall_ms` vs `Execution Time`** — these are intentionally
  different signals:
  - `Execution Time` is what Postgres measures *inside the server*.
  - `wall_ms` includes network round-trip, psycopg2 deserialisation,
    JSON parsing, and the cursor lifecycle.
  - Their *difference* is a useful client-side overhead feature later.
- **`cur.fetchone()[0]`** — `EXPLAIN ... FORMAT JSON` returns a single
  row, single column whose value is already a parsed Python `list[dict]`
  (psycopg2 maps PG's `json` type via its default adapter). No
  `json.loads` needed.

Then we build the final record:

```python
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
```

- **Timezone-aware UTC.** Always. Naive datetimes are a future bug
  waiting to happen when this dataset gets shared across machines or
  is rebuilt months later from logs.
- **Sql is stored verbatim**, including indentation, so future you can
  diff two recordings of the same `query_id` and see exactly what
  changed.

Filename convention:

```python
out_path = RAW_DIR / f"{query['id']}__{record['sql_hash']}.json"
```

The double underscore is a deliberate visual separator that's both
filesystem-safe on Windows/macOS/Linux and easy to split on (`name,
hash = stem.split("__")`). If you edit a query, the new file lands
alongside the old one — no overwrites, no lost history.

### 4.8 `append_index`

```python
line = {
    "query_id":     record["query_id"],
    "tag":          record["tag"],
    "sql_hash":     record["sql_hash"],
    "collected_at": record["collected_at"],
    **record["summary"],
}
with INDEX_FILE.open("a", encoding="utf-8") as f:
    f.write(json.dumps(line) + "\n")
```

Why **JSON Lines** (`.jsonl`) instead of a giant single JSON array?

- **Append-only friendly.** No need to read-modify-write the file; each
  run just `O_APPEND`s a line. Safe across crashes.
- **Streamable.** You can process it with `awk`, `jq -c`, or
  `pandas.read_json(..., lines=True, chunksize=...)` without loading
  the whole file into memory.
- **Robust to partial writes.** If a row is corrupted you lose one
  line, not the whole index.

### 4.9 `main` — connection lifecycle

```python
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True
```

- **`autocommit = True`.** `EXPLAIN ANALYZE` is a `SELECT`, so
  technically a transaction would be fine. But a long-running
  transaction holds snapshot locks and prevents `VACUUM` on a busy
  database; autocommit keeps each measurement independent and
  isolated, which is also better for *reproducibility* — every query
  runs against the latest committed state.
- **`with conn.cursor() as cur:`** — guarantees cursor close even on
  exception. Cursors hold server-side resources; leaking them is a
  classic psycopg2 footgun.
- **Per-query try/except.** A single bad query (typo, missing table,
  permission error) doesn't kill the whole batch — we record the
  failure count and move on.

```python
finally:
    conn.close()
```

Always close in `finally`, never inside the `try` body, so a mid-batch
exception still releases the TCP socket and the backend process.

Exit code semantics:

```python
return 0 if failed == 0 else 2
```

- `0` — perfect run.
- `1` — couldn't connect at all (returned earlier).
- `2` — connected, ran, but at least one query failed.

This makes the script CI-friendly: a wrapping shell can distinguish
"infrastructure broken" from "data problem".

---

## 5. The output, decoded

For each query you get a file like
`q03_join_customer_orders__c266d5b5.json`. Re-read your real recording
in `data/raw/`; here's what every block means:

### Top-level metadata

| Key | Meaning |
|-----|---------|
| `query_id` | Stable handle (`q03_join_customer_orders`). Use this for grouping. |
| `tag` | Plan-shape category. Use for stratified splits. |
| `sql` | Verbatim SQL that was executed. |
| `sql_hash` | First 8 hex chars of `SHA-1(sql)`. Changes ⇒ query changed. |
| `collected_at` | UTC ISO-8601 timestamp. |
| `wall_time_ms` | Client-side wall clock around `cur.execute`. |
| `summary.*` | Pre-flattened headline numbers, see below. |
| `plan` | Full raw `EXPLAIN` JSON tree. |

### `summary` block

| Key | Source in plan | What it tells you |
|-----|----------------|-------------------|
| `estimated_total_cost`   | root `Total Cost` | Planner's cost units (abstract). |
| `estimated_startup_cost` | root `Startup Cost` | Cost before first row is returned. |
| `estimated_rows`         | root `Plan Rows` | How many rows planner thought it'd return. |
| `actual_total_time_ms`   | root `Actual Total Time` | Real wall-clock at the root. |
| `actual_rows`            | root `Actual Rows` | Real rows produced. |
| `planning_time_ms`       | top-level `Planning Time` | How long the planner deliberated. |
| `execution_time_ms`      | top-level `Execution Time` | How long execution took, server-side. |
| `root_node_type`         | root `Node Type` | One-shot indicator of plan shape. |

### `plan` tree — operator vocabulary

The full tree is what Phase 2 will mine. The recurring fields per node:

- **`Node Type`** — `Seq Scan`, `Index Scan`, `Bitmap Heap Scan`,
  `Hash Join`, `Merge Join`, `Nested Loop`, `Hash`, `HashAggregate`,
  `Sort`, `Limit`, `Aggregate`, `Materialize`, ...
- **`Startup Cost` / `Total Cost`** — planner's units, *cumulative*
  including children.
- **`Plan Rows`** — planner's row estimate.
- **`Actual Startup Time` / `Actual Total Time`** — real ms, **per
  loop**.
- **`Actual Rows`** — real rows, per loop.
- **`Actual Loops`** — how many times this node was executed (e.g.
  inner side of a Nested Loop).
- **`Filter`, `Hash Cond`, `Index Cond`, `Merge Cond`, ...** — the
  textual predicate, which Phase 2 can vectorise (one-hot the operator,
  hash the column name, etc.).
- **`Rows Removed by Filter`** — cardinality lost to predicates;
  surprisingly predictive of runtime.
- **`Shared Hit Blocks` / `Shared Read Blocks`** — pages found in
  buffer cache vs. read from disk. Huge feature for runtime modelling.
- **`Plans`** — list of child operator nodes (recursive).

The estimation **error** at each node — `Plan Rows` vs
`Actual Rows × Actual Loops` — is the single most important
quantity in this entire dataset. It's the planner's blind spot, and
it's where the learned model adds value.

---

## 6. `requirements.txt`

```
psycopg2-binary==2.9.9
```

- `psycopg2-binary` ships precompiled wheels — no need for a system
  `libpq` build chain on your laptop. For *production* deployments
  the official advice is to switch to `psycopg2` (source) compiled
  against a known-good `libpq`, but for Phase 1 the binary distribution
  is the right call.
- The version is **pinned** (`==`), not floated (`>=`). Pinning is what
  makes "I ran this in May" reproducible in November when transitive
  deps have moved. When you upgrade, do it deliberately.

---

## 7. Reproducibility checklist

To rebuild the exact dataset on a fresh machine:

1. `createdb automl_qo`
2. `psql -d automl_qo -f db/schema.sql`
3. `pip install -r requirements.txt`
4. `python scripts/collect_data.py`

If the JSON files look meaningfully different from a previous run,
suspect (in order):

1. PostgreSQL **major version** changed — planner heuristics evolve.
2. `random_page_cost`, `seq_page_cost`, `effective_cache_size`, or
   `work_mem` differ from defaults. Check `SHOW` for each.
3. Statistics are stale — re-run `ANALYZE`.
4. The OS page cache state differs (cold vs. warm). Run the collector
   twice and use the second run for "warm" measurements.

---

## 8. Common gotchas you'll hit later

- **Plans differ between runs even with the same data.** That's
  expected — `random_page_cost` interacts with what's currently in the
  buffer cache. The dataset is *intentionally* noisy on this axis;
  don't try to "fix" it.
- **`Actual Rows` is a float, not an int.** PG averages over
  `Actual Loops`. The example file shows `6170.0`, not `6170`.
- **Subqueries can produce `InitPlan` / `SubPlan` nodes** that don't
  appear under `Plans` — they hang off other keys. Phase-2 feature
  extractors must walk the tree generically (any value that's a list
  of dicts with `Node Type`).
- **`EXPLAIN ANALYZE` on `INSERT/UPDATE/DELETE` mutates data.**
  Phase-1 workload is `SELECT`-only so this is safe; if you ever add a
  write query, wrap it: `BEGIN; EXPLAIN (ANALYZE) UPDATE ...; ROLLBACK;`.
- **Don't run the collector concurrently with itself.** `_index.jsonl`
  appends are fine, but two collectors writing to the same JSON file
  name is a race. If you parallelise later, shard by `query_id`.

---

## 9. What Phase 2 will plug into

The contract this phase exposes to the rest of the project:

- **Per-query record:** `data/raw/{query_id}__{sql_hash}.json` with the
  exact schema documented in §5.
- **Index of records:** `data/raw/_index.jsonl`, one JSON object per
  line, fields = top-level metadata + summary keys.
- **Stable identifiers:**
  - `query_id` for grouping.
  - `tag` for stratification.
  - `sql_hash` for "this is the same SQL".
  - `collected_at` for time-based splits.

A Phase-2 feature extractor should:

1. Glob `data/raw/*.json` (skip `_*`).
2. For each file, walk `record["plan"][0]["Plan"]` recursively.
3. For each node emit a feature row keyed by
   `(query_id, sql_hash, node_path)`.
4. The supervised target is `Actual Total Time × Actual Loops` (or
   the top-level `Execution Time` for query-level prediction).

If we've designed Phase 1 right, Phase 2 should never have to talk to
PostgreSQL again.

---

## 10. TL;DR

- **`db/schema.sql`** = a deterministic, reproducible mini-universe
  with enough data and enough index sparsity that the planner has
  *interesting* choices to make.
- **`config/db_config.py`** = 12-factor connection settings, no
  secrets in code.
- **`scripts/collect_data.py`** = run `EXPLAIN (ANALYZE, BUFFERS,
  VERBOSE, FORMAT JSON)`, capture the full tree, flatten the headline
  metrics, write one JSON per query and one index line, never lose
  history.
- **`data/raw/`** = the immutable training corpus. Treat it like
  source code: never edited in place, only appended to.
- **No ML, no API, no microservices** — that's the point. Phase 1's
  job is to make Phase 2 trivial.
