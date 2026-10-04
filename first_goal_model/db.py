"""Shared SQLite connection helper for the first-goal model."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "understat.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
