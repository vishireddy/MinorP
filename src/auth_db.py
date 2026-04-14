import sqlite3
import hashlib
import os

DB_PATH = "data/app_db.sqlite"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT
        )
    ''')
    # Create chats table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, is_admin=False):
    role = "admin" if is_admin else "citizen"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  (username, hash_password(password), role))
        conn.commit()
        return True, "✅ Registration successful! You can now log in."
    except sqlite3.IntegrityError:
        return False, "⚠️ Username already exists. Please choose another."
    except Exception as e:
        return False, f"⚠️ Database error: {e}"
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", 
              (username, hash_password(password)))
    result = c.fetchone()
    conn.close()
    if result:
        return True, result[0] # Returns the role string
    return False, None

def save_chat(username, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (username, role, content) VALUES (?, ?, ?)", 
              (username, role, content))
    conn.commit()
    conn.close()

def get_chat_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE username = ? ORDER BY id ASC", (username,))
    results = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in results]

def purge_system_chats():
    """Deletes all chat records from the database for privacy purging."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats")
    conn.commit()
    conn.close()
