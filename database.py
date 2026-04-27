import os
import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL", "")


class DBWrapper:
    """
    Thin wrapper making psycopg2 behave like our sqlite3 usage:
    - db.execute(sql, params) returns a cursor with fetchone/fetchall
    - db.commit() / db.close()
    - Automatically converts ? placeholders to %s
    - Returns RealDictRow objects (support both ["col"] and .col access in Jinja2)
    """

    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)

    def execute(self, sql, params=()):
        sql = sql.replace("?", "%s")
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or None)
        return cur

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def get_db():
    return DBWrapper()


def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            assigned_to TEXT DEFAULT '',
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id                SERIAL PRIMARY KEY,
            project_id        INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            filename          TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            status            TEXT DEFAULT 'uploaded',
            error_message     TEXT,
            style             TEXT DEFAULT '1',
            created_at        TIMESTAMP DEFAULT NOW()
        )
    """)
    db.commit()
    db.close()
