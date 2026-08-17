import sqlite3
import bcrypt
from typing import Optional

DB_PATH = "database.db"

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email_clean, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(email: str, password: str, db_path: str = DB_PATH) -> bool:
    init_db(db_path)
    email_clean = email.strip().lower()
    if not email_clean or not password:
        return False

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE email = ?", (email_clean,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    stored_hash = row[0]
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))


if __name__ == "__main__":
    init_db()
    test_email = "intern@test.com"
    test_pass = "secure123"

    print("***** Database & Auth Test *****")
    registered = register_user(test_email, test_pass)
    print(f"Register '{test_email}': {'Success' if registered else 'Failed or already exists'}")

    valid_login = authenticate_user(test_email, test_pass)
    print(f"Login with correct password: {'Success' if valid_login else 'Failed'}")

    invalid_login = authenticate_user(test_email, "wrongpass")
    print(f"Login with wrong password: {'Rejected (Expected)' if not invalid_login else 'Vulnerability!'}")