import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "/data/degas.db")


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
            client_id   INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        conn.execute("ALTER TABLE projects ADD COLUMN client_id INTEGER")
    except sqlite3.OperationalError:
        pass
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
    # Standalone audio transcription jobs (Studio's Podcast Page Generator
    # tab, Ben's ask 2026-09-17) -- not tied to any project/clip. This has
    # to be a DB table, not an in-process dict: gunicorn runs 2 workers
    # (separate processes, separate memory -- see the fcntl lock comment in
    # app.py for why that distinction already bit us once, 2026-08-18). A
    # POST that lands on worker A and a status GET that lands on worker B
    # need to see the same job state, which only a shared store (SQLite,
    # here) actually guarantees.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS podcast_jobs (
            id            TEXT PRIMARY KEY,
            status        TEXT NOT NULL DEFAULT 'transcribing',
            transcript    TEXT,
            error_message TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
