#!/usr/bin/env python

import socket
import glob

emotions = {'-1': "Negative", '0': "NoFace", '1': "Positive"}


def contactServer(sock, picture):
	with open(picture, 'rb') as f:
		l = f.read()
		string = bytes('%-10s' % (len(l),))   


		m = string.encode("utf-8") + l

		sock.sendall(m)
		print picture, "Sent!"
	f.close()



	msg = sock.recv(1024)
	print emotions[msg]
	return msg



if __name__ == '__main__':
	pictures_dir = "/home/peter/Desktop/*.png"
	PORT = 1026


	CLIENT_IP = '192.168.1.36'
	SERVER_IP = '192.168.1.33'


	pic_list = glob.glob(pictures_dir)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print "Connecting to Server!"
	s.connect((SERVER_IP, PORT))

	for picture in pic_list:
		prediction = contactServer(s, picture)
		print emotions[prediction]


	s.close()




