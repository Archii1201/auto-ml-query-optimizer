"""
Database connection configuration for the
AutoML-Powered Learned Query Optimizer (Phase 1).

You can override any of these values via environment variables
so credentials never have to be hard-coded for real deployments:

    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import os

DB_CONFIG = {
    "host":     os.getenv("PGHOST",     "localhost"),
    "port":     int(os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PGDATABASE", "automl_qo"),
    "user":     os.getenv("PGUSER",     "postgres"),
    "password": os.getenv("PGPASSWORD", "Archi@1201"),
}


def get_dsn() -> str:
    """Return a libpq-style DSN string built from DB_CONFIG."""
    return (
        f"host={DB_CONFIG['host']} "
        f"port={DB_CONFIG['port']} "
        f"dbname={DB_CONFIG['dbname']} "
        f"user={DB_CONFIG['user']} "
        f"password={DB_CONFIG['password']}"
    )
