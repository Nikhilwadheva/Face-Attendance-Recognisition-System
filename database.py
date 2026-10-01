from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

from flask import current_app, g

ATTENDANCE_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    attendance_date TEXT NOT NULL,
    checked_in_at TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'General',
    UNIQUE(employee_id, attendance_date, subject)
);
"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
""" + ATTENDANCE_SQL


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        connection = sqlite3.connect(current_app.config["DATABASE"])
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_app(app) -> None:
    database_path = Path(app.config["DATABASE"])
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        # Migrate v1 databases (no subject column) without losing history.
        columns = [row[1] for row in connection.execute("PRAGMA table_info(attendance)")]
        if "subject" not in columns:
            connection.executescript(
                "ALTER TABLE attendance RENAME TO attendance_old;"
                + ATTENDANCE_SQL
                + "INSERT INTO attendance (employee_id, attendance_date, checked_in_at, subject) "
                "SELECT employee_id, attendance_date, checked_in_at, 'General' FROM attendance_old;"
                "DROP TABLE attendance_old;"
            )
    app.teardown_appcontext(close_db)


def save_employee_and_samples(employee_code: str, name: str, image_paths: list[str]) -> list[str]:
    connection = get_db()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection:
        connection.execute(
            "INSERT INTO employees (employee_code, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(employee_code) DO UPDATE SET name = excluded.name",
            (employee_code, name, now),
        )
        employee_id = connection.execute(
            "SELECT id FROM employees WHERE employee_code = ?", (employee_code,)
        ).fetchone()["id"]
        old_paths = [
            row["image_path"]
            for row in connection.execute(
                "SELECT image_path FROM face_samples WHERE employee_id = ?", (employee_id,)
            ).fetchall()
        ]
        connection.execute("DELETE FROM face_samples WHERE employee_id = ?", (employee_id,))
        connection.executemany(
            "INSERT INTO face_samples (employee_id, image_path, created_at) VALUES (?, ?, ?)",
            [(employee_id, image_path, now) for image_path in image_paths],
        )
    return old_paths


def load_training_samples() -> list[tuple[int, str]]:
    rows = get_db().execute(
        "SELECT employee_id, image_path FROM face_samples ORDER BY employee_id, id"
    ).fetchall()
    return [(row["employee_id"], row["image_path"]) for row in rows]


def find_employee(employee_id: int) -> tuple[str, str] | None:
    row = get_db().execute(
        "SELECT employee_code, name FROM employees WHERE id = ?", (employee_id,)
    ).fetchone()
    return (row["employee_code"], row["name"]) if row else None


def record_attendance(employee_id: int, subject: str = "General") -> bool:
    cursor = get_db().execute(
        "INSERT OR IGNORE INTO attendance (employee_id, attendance_date, checked_in_at, subject) "
        "VALUES (?, ?, ?, ?)",
        (employee_id, date.today().isoformat(),
         datetime.now().astimezone().isoformat(timespec="seconds"), subject),
    )
    get_db().commit()
    return cursor.rowcount == 1


def list_attendance(selected_date: str, subject: str | None = None) -> list[dict[str, str]]:
    query = (
        "SELECT employees.employee_code, employees.name, attendance.attendance_date, "
        "attendance.checked_in_at, attendance.subject FROM attendance "
        "JOIN employees ON employees.id = attendance.employee_id WHERE attendance.attendance_date = ?"
    )
    params: list[str] = [selected_date]
    if subject:
        query += " AND attendance.subject = ?"
        params.append(subject)
    rows = get_db().execute(query + " ORDER BY attendance.checked_in_at, employees.name", params).fetchall()
    return [dict(row) for row in rows]


def list_absent(selected_date: str, subject: str) -> list[dict[str, str]]:
    rows = get_db().execute(
        "SELECT employee_code, name FROM employees WHERE id NOT IN "
        "(SELECT employee_id FROM attendance WHERE attendance_date = ? AND subject = ?) ORDER BY name",
        (selected_date, subject),
    ).fetchall()
    return [dict(row) for row in rows]


def list_subjects() -> list[str]:
    rows = get_db().execute("SELECT DISTINCT subject FROM attendance ORDER BY subject").fetchall()
    return [row["subject"] for row in rows]
