import hashlib
import secrets
import hmac
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "civicdex.db"


def initialize_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            citizen_name TEXT NOT NULL,
            email TEXT NOT NULL,
            district TEXT NOT NULL,
            language TEXT NOT NULL,
            complaint_text TEXT NOT NULL,
            category TEXT NOT NULL,
            department TEXT NOT NULL,
            confidence REAL,
            status TEXT DEFAULT 'Registered',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS citizens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS staff_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    initialize_db()
    print("CivicDex database initialized successfully!")


def register_complaint(
    complaint_id,
    citizen_name,
    email,
    district,
    language,
    complaint_text,
    category,
    department,
    confidence
):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO complaints (
            complaint_id,
            citizen_name,
            email,
            district,
            language,
            complaint_text,
            category,
            department,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        citizen_name,
        email,
        district,
        language,
        complaint_text,
        category,
        department,
        confidence
    ))

        

    connection.commit()
    connection.close()

    return complaint_id

def get_complaint_by_id(complaint_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM complaints WHERE complaint_id = ?",
        (complaint_id,)
    )

    complaint = cursor.fetchone()
    conn.close()

    return complaint

def get_complaints_by_department(department):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM complaints
        WHERE department = ?
        ORDER BY created_at DESC
        """,
        (department,)
    )

    complaints = cursor.fetchall()
    conn.close()

    return complaints

def get_all_department_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT department
        FROM complaints
    """)

    departments = cursor.fetchall()
    conn.close()

    return departments

def update_complaint_status(complaint_id, new_status):
    allowed_statuses = [
        "Registered",
        "Forwarded",
        "In Progress",
        "Resolved"
    ]

    if new_status not in allowed_statuses:
        raise ValueError("Invalid complaint status")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE complaints
        SET status = ?
        WHERE complaint_id = ?
        """,
        (new_status, complaint_id)
    )

    updated = cursor.rowcount
    conn.commit()
    conn.close()

    return updated > 0


def create_citizen_account(name, email, password):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()

    try:
        cursor.execute("""
            INSERT INTO citizens (
                name, email, password_hash, salt
            )
            VALUES (?, ?, ?, ?)
        """, (
            name.strip(),
            email.strip().lower(),
            password_hash,
            salt
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def verify_citizen_login(email, password):


    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM citizens WHERE email = ?",
        (email.strip().lower(),)
    )

    citizen = cursor.fetchone()
    conn.close()

    if citizen is None:
        return None

    entered_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(citizen["salt"]),
        310000
    ).hex()

    if hmac.compare_digest(
        entered_hash,
        citizen["password_hash"]
    ):
        return {
            "id": citizen["id"],
            "name": citizen["name"],
            "email": citizen["email"]
        }

    return None

def create_staff_account(name, email, password, role, department=None):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()

    try:
        cursor.execute("""
            INSERT INTO staff_accounts (
                name, email, password_hash, salt, role, department
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name.strip(),
            email.strip().lower(),
            password_hash,
            salt,
            role,
            department
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()

def verify_staff_login(email, password):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM staff_accounts WHERE email = ?",
        (email.strip().lower(),)
    )

    staff = cursor.fetchone()
    conn.close()

    if staff is None:
        return None

    entered_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(staff["salt"]),
        310000
    ).hex()

    if hmac.compare_digest(
        entered_hash,
        staff["password_hash"]
    ):
        return {
            "id": staff["id"],
            "name": staff["name"],
            "email": staff["email"],
            "role": staff["role"],
            "department": staff["department"]
        }

    return None

def get_complaints_by_email(email):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM complaints
        WHERE LOWER(email) = LOWER(?)
        ORDER BY created_at DESC
        """,
        (email.strip(),)
    )

    complaints = cursor.fetchall()
    conn.close()

    return complaints