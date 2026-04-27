import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "degas.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            assigned_to TEXT DEFAULT '',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id        INTEGER NOT NULL,
            filename          TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            status            TEXT DEFAULT 'uploaded',
            error_message     TEXT,
            style             TEXT DEFAULT '1',
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()
