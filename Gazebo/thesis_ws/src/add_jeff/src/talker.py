#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2, pi, sqrt, floor
import pickle
import Gridworld #We use the training code for the class GridWorld
import client
import glob
import os
import socket
import time
import numpy as np
import matplotlib.pyplot as plt
from random import choice


PORT = 1026
CLIENT_IP = '192.168.1.36'
SERVER_IP = '192.168.1.33'
Q_LINK = 'Qtable14.pkl'

class setRobotPosition(object):
	def __init__(self, pub, s):
		w = 9
		h = 9
		self.graphX = []
		self.graphY = []

		self.pub = pub #Publisher handle
		self.sock = s
		self.rob = [0,0,0]

		self.jeff = [0, 0] #x, y position
		self.jeff_name = 'jeff-happy' #Person's name
		
		self.speed = Twist()

           
	def getModelIndex(self, msg, model_name):
		return msg.name.index(model_name)



	def getDist(self):
		dx = self.jeff[0] - self.rob[0]
		dy = self.jeff[1] - self.rob[1]	
		
		return sqrt(dx**2 + dy**2)	


	def getAngle(self):
		inc_x = self.jeff[0] - self.rob[0]
		inc_y = self.jeff[1] - self.rob[1]	

		return atan2(inc_y, inc_x)
		

	def setSpeed(self):
		ang = self.getAngle() - self.rob[2]

		
		self.speed.linear.x = -0.12
		self.speed.angular.z = 0.05 if ang > 0 else -0.05
		self.pub.publish(self.speed)


	def findFace(self):
		
		list_of_files = glob.glob('/home/peter/thesis_ws/src/add_jeff/src/camera_save_tutorial/*.jpg')
		latest_file = max(list_of_files, key = os.path.getctime)
		
		face = client.contactServer(self.sock, latest_file)

		return face
	


	def findJeff(self):
		freq = rospy.Rate(4)


		while not rospy.is_shutdown():
			self.setSpeed()

			dist = self.getDist()
			face = self.findFace()

			print "Distance: ", dist, " Message: ", face
			if face != '-1':
				self.graphX.append(dist)
				self.graphY.append(int(face))
		
			if dist < 0.8:
				break	

			freq.sleep()
		
		self.speed.linear.x = 0.0
		self.speed.angular.z = 0.0
		self.pub.publish(self.speed)

		'''with open('distance-face-jeff-happy.pkl', 'wb') as f:
			pickle.dump([self.graphX, self.graphY],f)

		plt.step(self.graphX, self.graphY)
		plt.xlim(13, 0.3)
		plt.ylim(-0.1, 1.1)

		plt.xlabel("Distance (m)")
		plt.ylabel("Face-Detected")
		#plt.savefig('distance-face-jeff-happy.png')
		plt.show()'''



	def callback(self, msg):
		jeff_ind = self.getModelIndex(msg, self.jeff_name)

		self.jeff[0] = msg.pose[jeff_ind].position.x
		self.jeff[1] = msg.pose[jeff_ind].position.y


	def callback2(self, msg):
		self.rob[0] = msg.pose.pose.position.x
		self.rob[1] = msg.pose.pose.position.y

		rob_ori = msg.pose.pose.orientation	
		(roll, pitch, theta) = euler_from_quaternion([rob_ori.x, rob_ori.y, rob_ori.z, rob_ori.w ])
		self.rob[2] = theta



def listener():
	rospy.init_node('listener', anonymous=True)
	pub = rospy.Publisher("/my_vehicle2/cmd_vel", Twist, queue_size=1)

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((SERVER_IP, PORT))


	setPos = setRobotPosition(pub, s)
	

	rospy.Subscriber("/gazebo/model_states", ModelStates, setPos.callback)
	rospy.Subscriber("/my_vehicle2/odom", Odometry, setPos.callback2)
	time.sleep(1)
	
	setPos.findJeff() #angleToJeff()

	s.close()



if __name__ == '__main__':
	listener()
