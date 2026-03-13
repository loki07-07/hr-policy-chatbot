"""Simple SQLite-backed user store for signup / login."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any


DB_PATH = Path(__file__).resolve().parent.parent / "users.db"


def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'Employee'
            )
            """
        )
        try:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'Employee'")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(name: str, email: str, password: str, role: str = "Employee") -> bool:
    role = role if role in ("Employee", "Manager", "HR") else "Employee"
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name.strip(), email.strip().lower(), _hash_password(password), role),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    with _get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, email, password_hash, role FROM users WHERE email = ?",
            (email.strip().lower(),),
        )
        row = cur.fetchone()
    if not row:
        return None
    user_id, name, stored_email, password_hash = row[0], row[1], row[2], row[3]
    role = row[4] if len(row) > 4 and row[4] else "Employee"
    if password_hash != _hash_password(password):
        return None
    return {"id": user_id, "name": name, "email": stored_email, "role": role}


def list_users() -> list[Dict[str, Any]]:
    """Return all users with id, name, email, role."""
    with _get_connection() as conn:
        cur = conn.execute("SELECT id, name, email, role FROM users ORDER BY id ASC")
        rows = cur.fetchall()
    return [
        {"id": r[0], "name": r[1], "email": r[2], "role": r[3] or "Employee"}
        for r in rows
    ]


def update_user_role(user_id: int, role: str) -> bool:
    if role not in ("Employee", "Manager", "HR"):
        return False
    with _get_connection() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()
    return True


