import sqlite3


database_path = "reconstruct_3d/database.db"


db_conn = sqlite3.connect(database_path)
db_cursor = db_conn.cursor()


db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = db_cursor.fetchall()

for table in tables:

    print(f"Table: {table[0]}")
    db_cursor.execute(f"PRAGMA table_info('{table[0]}');")

    rows = db_cursor.fetchall()
    for row in rows:
        print(f"  {row[1]} ({row[2]}) ")
db_conn.close()