## databse login and analytics

import os
import sqlite3
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
DB_PATH = f"{DATA_PATH}/usage_logs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    found INTEGER NOT NULL,
    top_score REAL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_query_logs_query ON query_logs(query);
CREATE INDEX IF NOT EXISTS idx_query_logs_found ON query_logs(found);
"""


@contextmanager
def get_connection():
    """Yield a sqlite3 connection, ensuring it's closed after use."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the query_logs table and indexes if they don't exist. Idempotent."""
    os.makedirs(DATA_PATH, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def log_query(
    query: str, answer: str, found: bool, top_score: float | None, latency_ms: float
) -> None:
    """Insert one row into query_logs."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO query_logs (query, answer, found, top_score, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, answer, int(found), top_score, latency_ms),
        )
        conn.commit()


def get_most_frequent_questions(limit: int = 10) -> list[dict]:
    """Return the most frequently asked questions, ranked by count."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT query, COUNT(*) AS count FROM query_logs "
            "GROUP BY query ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"query": row[0], "count": row[1]} for row in rows]


def get_no_answer_queries(limit: int = 50) -> list[dict]:
    """Return distinct queries where no answer was found in context, ranked by count."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT query, COUNT(*) AS count FROM query_logs "
            "WHERE found = 0 GROUP BY query ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"query": row[0], "count": row[1]} for row in rows]


def get_average_latency_ms() -> float:
    """Return the average /ask response latency in milliseconds. 0.0 if no rows yet."""
    with get_connection() as conn:
        row = conn.execute("SELECT AVG(latency_ms) FROM query_logs").fetchone()
        return row[0] if row[0] is not None else 0.0
