import sqlite3

# Connect to the database
conn = sqlite3.connect('valorant.sqlite')
cursor = conn.cursor()

# Retrieve the column names
cursor.execute("PRAGMA table_info(Game_Scoreboard)")
columns = cursor.fetchall()

# Display column names to confirm
for column in columns:
    print(column)

conn.close()
