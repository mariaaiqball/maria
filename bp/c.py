import threading
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.gethostbyname(socket.gethostname())
client.connect((s, 5000))

print("Hi, Welcome to your login.")
message = client.recv(1024).decode()
client.send(input(message).encode())
message = client.recv(1024).decode()
client.send(input(message).encode())
print(client.recv(1024).decode())                
