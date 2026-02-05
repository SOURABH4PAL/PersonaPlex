import sqlite3
import bcrypt

DB_NAME = "personaplex.db"


# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------

def get_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


# -------------------------------------------------
# USERS TABLE
# -------------------------------------------------

def create_users_table():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password BLOB
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# CHAT TABLES
# -------------------------------------------------

def create_chat_tables():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# SIGNUP
# -------------------------------------------------

def signup(email, password):
    conn = get_db()
    c = conn.cursor()

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        c.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hashed)
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# -------------------------------------------------
# LOGIN
# -------------------------------------------------

def login(email, password):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT id, password FROM users WHERE email = ?",
        (email,)
    )

    user = c.fetchone()
    conn.close()

    if not user:
        return None

    user_id, hashed_password = user

    if bcrypt.checkpw(password.encode(), hashed_password):
        return (user_id, email)

    return None


# -------------------------------------------------
# CHAT MANAGEMENT
# -------------------------------------------------

def create_chat(user_id, title):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "INSERT INTO chats (user_id, title) VALUES (?, ?)",
        (user_id, title)
    )

    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id


def get_chats(user_id):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT id, title FROM chats WHERE user_id=? ORDER BY id DESC",
        (user_id,)
    )

    rows = c.fetchall()
    conn.close()
    return rows


def save_message(chat_id, role, content):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )

    conn.commit()
    conn.close()


def load_chat(chat_id):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT role, content FROM messages WHERE chat_id=?",
        (chat_id,)
    )

    messages = c.fetchall()
    conn.close()
    return messages
