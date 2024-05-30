import threading # two things happening at once
import sqlite3
import hashlib
import socket # used to establish the connection between client and server

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.gethostbyname(socket.gethostname())
server.bind((s, 5000))
server.listen()


def handle_connection(c):
    c.send("Username: ".encode())
    username = c.recv(1024).decode()
    c.send("Password: ".encode())
    password = c.recv(1024)
    password = hashlib.sha512(password).hexdigest()

    #checks if the password is correct
    conn = sqlite3.connect("userdata.db")
    cur = conn.cursor()


    cur.execute("SELECT * FROM userdata WHERE username = ? AND password = ?", (username, password))

    if cur.fetchall() == []:
        c.send("Login failed! Please Try Again.".encode())
    else:
        c.send("Login successful!".encode())
    
    
while True:
       client, addr = server.accept()
       threading.Thread(target=handle_connection, args=(client,)).start()

    
 