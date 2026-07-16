"""
SQLite data access layer for the attendance system.
Tables:
    users       -> registered people (the CNN's output classes)
    attendance  -> one check-in row per user per day
    settings    -> single-row key/value config (login deadline, grace period)
"""
import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label_id    INTEGER UNIQUE,   -- index used by the CNN's softmax output
            name        TEXT NOT NULL,
            roll_no     TEXT,
            created_at  TEXT NOT NULL,
            sample_count INTEGER DEFAULT 0,
            trained     INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL,
            status      TEXT NOT NULL,       -- 'ON-TIME' or 'LATE'
            confidence  REAL,
            UNIQUE(user_id, date),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # sensible defaults, only inserted the first time
    cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('login_deadline', '09:30')")
    cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('grace_minutes', '5')")
    cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('recognition_threshold', '0.75')")
    conn.commit()
    conn.close()


# ---------------- users ----------------

def create_user(name, roll_no):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(label_id), -1) + 1 AS next_id FROM users")
    next_label_id = cur.fetchone()["next_id"]
    cur.execute(
        "INSERT INTO users (label_id, name, roll_no, created_at) VALUES (?, ?, ?, ?)",
        (next_label_id, name, roll_no, datetime.now().isoformat())
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id, next_label_id


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_label(label_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE label_id = ?", (label_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_sample_count(user_id, count):
    conn = get_connection()
    conn.execute("UPDATE users SET sample_count = sample_count + ? WHERE id = ?", (count, user_id))
    conn.commit()
    conn.close()


def mark_all_trained():
    conn = get_connection()
    conn.execute("UPDATE users SET trained = 1")
    conn.commit()
    conn.close()


def delete_user(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


# ---------------- settings ----------------

def get_setting(key, default=None):
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_connection()
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
    conn.commit()
    conn.close()


def get_all_settings():
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


# ---------------- attendance ----------------

def mark_attendance(user_id, status, confidence):
    """One row per user per day. Returns 'already_marked' if today's row exists."""
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")
    conn = get_connection()
    existing = conn.execute(
        "SELECT * FROM attendance WHERE user_id = ? AND date = ?", (user_id, today)
    ).fetchone()
    if existing:
        conn.close()
        return {"already_marked": True, "record": dict(existing)}
    conn.execute(
        "INSERT INTO attendance (user_id, date, time, status, confidence) VALUES (?, ?, ?, ?, ?)",
        (user_id, today, now_time, status, confidence)
    )
    conn.commit()
    record = conn.execute(
        "SELECT * FROM attendance WHERE user_id = ? AND date = ?", (user_id, today)
    ).fetchone()
    conn.close()
    return {"already_marked": False, "record": dict(record)}


def get_attendance_for_date(target_date=None):
    target_date = target_date or date.today().isoformat()
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.*, u.name, u.roll_no
        FROM attendance a JOIN users u ON a.user_id = u.id
        WHERE a.date = ?
        ORDER BY a.time ASC
    """, (target_date,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_summary(target_date=None):
    target_date = target_date or date.today().isoformat()
    rows = get_attendance_for_date(target_date)
    total_users = len(list_users())
    present = len(rows)
    on_time = len([r for r in rows if r["status"] == "ON-TIME"])
    late = len([r for r in rows if r["status"] == "LATE"])
    absent = total_users - present
    return {
        "date": target_date,
        "total_users": total_users,
        "present": present,
        "on_time": on_time,
        "late": late,
        "absent": absent,
        "records": rows
    }
