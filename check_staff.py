import sqlite3
from database import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT name, email, role, department
    FROM staff_accounts
""")

staff = cursor.fetchall()

print("===== REGISTERED STAFF ACCOUNTS =====")

for person in staff:
    print(person)

conn.close()