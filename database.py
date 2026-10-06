"""
database.py
===========
Stores analysed complaints in a small SQLite database (data/complaints.db).

SQLite ships with Python, so there is nothing extra to install.

Table columns
-------------
complaint_id, complaint_text, category, sentiment, priority, urgency,
keywords, summary, recommended_action, created_at, source

`source` is "user" for complaints typed into the app and "sample" for the
demo rows loaded from data/complaints.csv.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from complaint_analyzer import DATA_PATH, get_recommended_action

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "complaints.db"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Column names shown to the user (database name -> display name)
DISPLAY_COLUMNS = {
    "complaint_id": "Complaint ID",
    "complaint_text": "Complaint",
    "category": "Category",
    "sentiment": "Sentiment",
    "priority": "Priority",
    "urgency": "Urgency",
    "keywords": "Keywords",
    "summary": "Summary",
    "recommended_action": "Recommended Action",
    "created_at": "Date/Time",
}


class DatabaseError(Exception):
    """A friendly database error that is safe to show to normal users."""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the database file (creating the folder if needed)."""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(DB_PATH)
    except (sqlite3.Error, OSError) as exc:
        logger.error("Cannot open database: %s", exc)
        raise DatabaseError("The complaint database could not be opened.") from exc


@contextmanager
def connection():
    """Open the database, commit on success, and ALWAYS close it afterwards."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the complaints table if it does not exist yet."""
    try:
        with connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    complaint_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_text     TEXT NOT NULL,
                    category           TEXT,
                    sentiment          TEXT,
                    priority           TEXT,
                    urgency            TEXT,
                    keywords           TEXT,
                    summary            TEXT,
                    recommended_action TEXT,
                    created_at         TEXT,
                    source             TEXT DEFAULT 'user'
                )
                """
            )
    except sqlite3.Error as exc:
        logger.error("init_db failed: %s", exc)
        raise DatabaseError("The complaint database could not be prepared.") from exc


def save_complaint(result: dict, source: str = "user", created_at: str | None = None) -> int:
    """Insert one analysed complaint and return its new ID."""
    created_at = created_at or datetime.now().strftime(TIME_FORMAT)
    try:
        with connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO complaints
                (complaint_text, category, sentiment, priority, urgency,
                 keywords, summary, recommended_action, created_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["complaint_text"],
                    result["category"],
                    result["sentiment"],
                    result["priority"],
                    result["urgency"],
                    ", ".join(result["keywords"]),
                    result["summary"],
                    result["recommended_action"],
                    created_at,
                    source,
                ),
            )
            return cursor.lastrowid
    except (sqlite3.Error, KeyError) as exc:
        logger.error("save_complaint failed: %s", exc)
        raise DatabaseError("Your complaint was analysed but could not be saved.") from exc


def load_complaints(include_sample: bool = True) -> pd.DataFrame:
    """Return all complaints as a DataFrame (newest first)."""
    query = "SELECT * FROM complaints"
    if not include_sample:
        query += " WHERE source = 'user'"
    query += " ORDER BY datetime(created_at) DESC, complaint_id DESC"
    try:
        with connection() as conn:
            df = pd.read_sql_query(query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        logger.error("load_complaints failed: %s", exc)
        raise DatabaseError("Complaint history could not be loaded.") from exc

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def count_complaints() -> int:
    try:
        with connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    except sqlite3.Error as exc:
        raise DatabaseError("The complaint database could not be read.") from exc


def clear_user_complaints() -> None:
    """Delete only the complaints typed by the user (sample data is kept)."""
    try:
        with connection() as conn:
            conn.execute("DELETE FROM complaints WHERE source = 'user'")
    except sqlite3.Error as exc:
        raise DatabaseError("Complaints could not be deleted.") from exc


def seed_sample_data(analyzer, csv_path: Path = DATA_PATH) -> int:
    """Load data/complaints.csv into the database (only if the table is empty).

    The CSV supplies Category / Priority / Sentiment. The analyzer fills in the
    other columns (urgency, keywords, summary). Dates are spread randomly over
    the last 30 days so that the "Complaint trends" chart has something to show.
    Returns the number of rows added.
    """
    if count_complaints() > 0:
        return 0
    if not Path(csv_path).exists():
        raise DatabaseError("The sample dataset (data/complaints.csv) is missing.")

    try:
        sample = pd.read_csv(csv_path)
        required = {"Complaint", "Category", "Priority", "Sentiment"}
        if not required.issubset(sample.columns):
            raise ValueError("missing columns")
    except Exception as exc:
        raise DatabaseError("The sample dataset could not be read.") from exc

    rng = random.Random(42)  # fixed seed -> same dates every time
    now = datetime.now()
    added = 0
    for row in sample.itertuples(index=False):
        try:
            result = analyzer.analyze(row.Complaint)
        except Exception as exc:  # skip a bad row instead of failing everything
            logger.warning("Skipping sample row: %s", exc)
            continue
        result["category"] = row.Category
        result["priority"] = row.Priority
        result["sentiment"] = row.Sentiment
        result["recommended_action"] = get_recommended_action(row.Category, row.Priority)

        moment = now - timedelta(
            days=rng.randint(0, 29), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        save_complaint(result, source="sample", created_at=moment.strftime(TIME_FORMAT))
        added += 1
    return added
