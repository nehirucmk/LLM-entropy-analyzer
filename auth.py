import sqlite3
import bcrypt
from typing import Optional, Tuple
import time

DB_PATH = "database.db"
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 300

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)

def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            failed_attempts INTEGER DEFAULT 0,
            locked_until REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    if "failed_attempts" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN failed_attempts INTEGER DEFAULT 0")
    if "locked_until" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN locked_until REAL DEFAULT 0")
    conn.commit()
    conn.close()

def register_user(email: str, password: str, db_path: str = DB_PATH) -> bool:
    init_db(db_path)
    email_clean = email.strip().lower()
    if not email_clean or not password:
        return False

    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash, failed_attempts_ locked_until) VALUES (?, ?, 0, 0)",
            (email_clean, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(email: str, password: str, db_path: str = DB_PATH) -> Tuple[bool, str]:
    init_db(db_path)
    email_clean = email.strip().lower()
    if not email_clean or not password:
        return False, "Email and password cannot be empty."

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, failed_attempts, locked_until FROM users WHERE email = ?", (email_clean,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False, "Invalid email or password."

    stored_hash, failed_attempts, locked_until = row
    current_time = time.time()

    if locked_until and current_time < locked_until:
        remaining_sec = int(locked_until-current_time)
        conn.close()
        return False, f"Account locked due to too many failed attempts. Try again in {remaining_sec} seconds."

    
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
        cursor.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE email = ?", (email_clean,))
        conn.commit()
        conn.close()
        return True, "Authentication successful."
    else:
        new_attempts = (failed_attempts or 0)+1
        new_lock = 0
        if new_attempts >= MAX_FAILED_ATTEMPTS:
            new_lock = current_time + LOCKOUT_DURATION_SECONDS
            status_msg = f"Account locked for 5 minutes due to {MAX_FAILED_ATTEMPTS} consecutive failed attempts."
        else:
            remaining_tries = MAX_FAILED_ATTEMPTS - new_attempts
            status_msg = f"Invalid email or password. {remaining_tries} attempts remaining before lockout."
        cursor.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE email = ?",
            (new_attempts,new_lock,email_clean)
        )       
        conn.commit()
        conn.close()
        return False, status_msg

def delete_user(email: str, password: str, db_path: str = DB_PATH) -> bool:
    # verify user credentials before executing account deletion
    if not authenticate_user(email, password, db_path):
        return False

    email_clean = email.strip().lower()
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE email = ?", (email_clean,))
    conn.commit()
    conn.close()
    return True

