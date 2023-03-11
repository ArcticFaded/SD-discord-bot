import sqlite3

def connect_db():
    return sqlite3.connect("bot.db")

def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXIST share (id INTEGER PRIMARY KEY AUTOINCREMENT, image_url TEXT NOT NULL, prompt TEXT NOT NULL, author TEXT NOT NULL, model TEXT NOT NULL, processed INTEGER NOT NULL, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_generation(image_url, prompt, author, model):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO share(image_url, prompt, author, model, processed) VALUES (?,?,?,?,?)",
            (image_url, prompt, author, model, 0))
    conn.commit()
    conn.close()

