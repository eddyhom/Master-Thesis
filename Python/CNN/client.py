import socket
from time import sleep

PORT = 1026

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), PORT))


with open('anger_disgust.png', 'rb') as f:
    print("File opened")

    l = f.read()
    s.sendall(l)
f.close()

msg = ""
s.send(msg.decode("utf-8"))


while True:
    msg = s.recv(1024)

    if not msg:
        break
    else:
        print(msg)

sleep(1)
s.close()

'''while True:
    msg = s.recv(7)
    if len(msg) <= 0:
        break
    complete_info += msg.decode("utf-8")

print complete_info'''
