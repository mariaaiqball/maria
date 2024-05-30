import sqlite3
import hashlib

conn = sqlite3.connect("userdata.db")
cur = conn.cursor()

cur.execute(""" 
CREATE TABLE IF NOT EXISTS userdata(
    id INTEGER PRIMARY KEY, 
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL
)
""")
cur.execute("DELETE FROM userdata")
username1, password1 = "mariaiqbal", hashlib.sha512("mariaspassword".encode()).hexdigest()
username2, password2 = "utopia", hashlib.sha512("LaFl@me".encode()).hexdigest()
username3, password3 = "champagnepapi", hashlib.sha512("richbabydaddy".encode()).hexdigest()
username4, password4 = "carti", hashlib.sha512("atldraco".encode()).hexdigest()
username5, password5 = "abel", hashlib.sha512("65spencerave".encode()).hexdigest()
username6, password6 = "advaitpatil", hashlib.sha512("rutgers2004".encode()).hexdigest()
username7, password7 = "shanikapaul", hashlib.sha512("forallthedogs".encode()).hexdigest()


cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username1, password1))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username2, password2))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username3, password3))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username4, password4))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username5, password5))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username6, password6))
cur.execute("INSERT INTO userdata(username, password) VALUES (?, ?)", (username7, password7))

conn.commit()

