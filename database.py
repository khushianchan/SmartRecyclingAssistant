
import sqlite3

DB_NAME = "recycling_history.db"


def initialize_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT
                    (datetime('now', 'localtime'))
            )
        """)


def save_prediction(category, confidence):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            INSERT INTO predictions (category, confidence)
            VALUES (?, ?)
            """,
            (category, confidence)
        )


def get_dashboard_data():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        categories = conn.execute("""
            SELECT category, COUNT(*) AS count
            FROM predictions
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        recent = conn.execute("""
            SELECT category, confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

    return total, categories, recent
