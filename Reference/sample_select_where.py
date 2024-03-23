import sqlite3

connection = sqlite3.connect('database.db')

cur = connection.cursor()

data = cur.execute("SELECT * FROM characters WHERE c_name == 'Archibald'").fetchall()
print(data)

connection.close()