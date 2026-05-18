"""
Migration script to add the 'meta' column to the wallets table.

Run with: python -m database.migrate_add_wallet_meta
Or:       python alphapilot_ai/database/migrate_add_wallet_meta.py
"""
import sqlite3
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_path():
    """Find the database file."""
    # Check common locations
    candidates = [
        "alphapilot.db",
        "alphapilot_ai/alphapilot.db",
        "../alphapilot.db",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "alphapilot.db"  # Default

def migrate():
    """Add meta column to wallets table if it doesn't exist."""
    db_path = get_db_path()
    print(f"[MIGRATE] Using database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(wallets)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "meta" in columns:
        print("[MIGRATE] 'meta' column already exists in wallets table")
    else:
        print("[MIGRATE] Adding 'meta' column to wallets table...")
        cursor.execute("ALTER TABLE wallets ADD COLUMN meta TEXT DEFAULT '{}'")
        conn.commit()
        print("[MIGRATE] Successfully added 'meta' column")
    
    conn.close()
    print("[MIGRATE] Migration complete")

if __name__ == "__main__":
    migrate()
