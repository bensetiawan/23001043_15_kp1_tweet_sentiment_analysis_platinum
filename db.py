import sqlite3

conn = sqlite3.connect("db_alay.sqlite")

cursor = conn.cursor()


sql_alay = """CREATE TABLE replace_alay(
    ALAY text,
    TIDAK_ALAY text
)"""

cursor.execute("DROP TABLE IF EXISTS replace_alay")
