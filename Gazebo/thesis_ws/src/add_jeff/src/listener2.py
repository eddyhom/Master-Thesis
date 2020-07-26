#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2, pi, sqrt, floor
import numpy as np
import matplotlib.pyplot as plt
import pickle
import QLearning_2Persons as Gridworld#We use the training code for the class GridWorld
import client
import glob
import os
import socket
import time

PORT = 1026
CLIENT_IP = '192.168.1.36'
SERVER_IP = '192.168.1.33'
Q_LINK = 'Qtable_2Emotions_500k_fversion.pkl'
BG_PIC = '/home/peter/Desktop/gazebo_map2.png'
FLAG = False

class setRobotPosition(object):
	def __init__(self, pub, s):
		w = 9
		h = 9
		self.env = Gridworld.GridWorld(w, h)
		self.pub = pub #Publisher handle
		self.sock = s
		self.rob = [0,0,0]

		self.jeff = [[7, 7], [9, 9]] #x, y position
		self.jeff_name = ['jeff-happy', 'jeff-angry']#Person's name - jeff2 = Happy, jeff3 = Angry

		self.jeff_found = []
		self.faces = ""
		self.jeffChecker = [False, False]
		
		self.speed = Twist()

		self.discreteDist, self.discreteDist2  = self.discDist()
		self.discreteAng = self.discAng()

		self.Q = self.getQ()
		self.state = []
		
		self.actionSpace = {'R': 0, 'UR': 1, 'U': 2,  'UL': 3, 
				    'L': 4 ,'DL': 5, 'D': 6, 'DR': 7}

		self.path = []
		self.pathWithFace = []

                             
                                                 
	def getModelIndex(self, msg, model_name):
		return msg.name.index(model_name)

	def getQ(self):
		f = open(Q_LINK, 'rb') #Open Qtable file to use what was learned
		Q, reward = pickle.load(f) #Set Qtable to Q, ignore reward its not needed here
		return Q

	def discDist(self):#Change distance to discrete distance
		w = 9
		h = 9
		l_straight = []
		l_diagonal = []
		for i in range(w+1):
			for j in range(h+1):
				dx = w - i
				dy = h - j
				if dx == 0 or dy == 0:
					l_straight.append(round(sqrt(dx**2+dy**2),2))
				else:
					l_diagonal.append(round(sqrt(dx**2+dy**2),2))

		l_diagonal.append(0.0)
		return l_straight, l_diagonal

	def discAng(self): #Change angles to discrete angles
		return [[-22, 22],[22, 68],[68, 112],[112, 158],[-158, -112],[-112, -68], [-68, -22]]

	def approxDist(self, dist, odd): #Approximate distance to closest from QTable
		if dist < 1.0:
			return 0
		if odd == 0:		
			return min(self.discreteDist, key=lambda x:abs(x-dist))
		else:
			return min(self.discreteDist2, key=lambda x:abs(x-dist))


	def approxAngle(self, angle): #Approximate angle to closest from QTable
		angle = round(angle*(180/pi))

		if angle >= 158 or angle <= -158:
			return 4

		for i in range(len(self.discreteAng)):
			if angle in range(self.discreteAng[i][0],self.discreteAng[i][1]):
				return i+1 if i > 3 else i

		return 0


	def getDist(self, jeff): #Get distance to jeff
		dx = jeff[0] - self.rob[0]
		dy = jeff[1] - self.rob[1]	
		
		return sqrt(dx**2 + dy**2)	

	def getAngle(self, jeff): #Get angle to jeff
		inc_x = jeff[0] - self.rob[0]
		inc_y = jeff[1] - self.rob[1]	

		return atan2(inc_y, inc_x)

	def maxAction(Q, state, actions): #Choose best action from current state
		values = np.array([Q[state, a] for a in actions])
		action = np.argmax(values)
		return actions[action]

	def rotateCCW(self, rotTo, rotFrom):
		cw = rotFrom
		ccw = rotFrom

		while True:
			ccw += 1
			cw -= 1
			if ccw % 8 == rotTo:
				return True
			elif cw % 8 == rotTo:
				return False
				 
	def getSpeed(self, redZone, action, rob):
		angle = self.getAngle(self.jeff[0])
		ang = self.approxAngle(angle)
		angle = round(angle*(180/pi))


		if redZone and action != ang:
			angleRedZone = {0: 0.1, 1: 45, 2: 90,  3: 135, 
				        4: 179.9, 5: -135, 6: -90, 7: -45}
			angle2 = angleRedZone[action]
			print angle, angle2, action, ang
			angle = angle2

		robAng = round(self.rob[2]*(180/pi))
		dist = self.getDist(self.jeff[0])

		maxAngle = 180
		maxDist = 11.5
		
		linVel = dist/maxDist
		angVel = (angle - robAng)/maxAngle
		if redZone:
			linVel = 0.1
			angVel = 0.1

		return linVel, angVel
		
		
	def setSpeed(self, action, state, rob, old_state):
		linVel, angVel = self.getSpeed(state[2], action, rob)
		if state[0] == 0:
			self.speed.linear.x = 0.0
			self.speed.angular.z = 0.0
		elif action == rob: #Go Straight	
			self.speed.linear.x = -0.2 * linVel - 0.03
			self.speed.angular.z = 0.1 * angVel + 0.02 if angVel > 0 else 0.1 * angVel - 0.02 #Negative clockwise, positive counter-clockwise
			if not self.jeffChecker[0]:
				self.path.append([self.rob[0], self.rob[1]])
			else:
				self.pathWithFace.append([self.rob[0], self.rob[1]])
		else:
			ccw = self.rotateCCW(action, rob)
			self.speed.linear.x = 0.0
			if ccw:
				self.speed.angular.z = 0.3 * abs(angVel) + 0.02
			else:
				self.speed.angular.z = -0.3 * abs(angVel) - 0.02
	

	def getState(self):
		angPos = self.approxAngle(self.getAngle(self.jeff[0]))
		angNeg = self.approxAngle(self.getAngle(self.jeff[1]))

		distPos = self.approxDist(self.getDist(self.jeff[0]), angPos % 2)#Send if angle is odd or even
		distNeg = self.approxDist(self.getDist(self.jeff[1]), angNeg % 2)#Send if angle is odd or even


		nearby = False
		quadrant2 = 0

		if distNeg < 2.0:#If closer than 1.42mts ~ sqrt(2)
			'''dx = self.jeff[1][0] - self.rob[0]
			dy = self.jeff[1][1] - self.rob[1]	
			if dx == 0 or dy == 0:
				if distNeg <= 1.0:
					nearby = True
					quadrant2 = angNeg
			else:'''
			nearby = True
			quadrant2 = angNeg

		if distPos == 0:
			return tuple([0, 0, False, 0])
		else:
			return tuple([distPos, angPos, nearby, quadrant2])
				
	def angleToJeff(self):
		freq = rospy.Rate(10)
		state = [0, 0, False, 0]
		old_state = [0, 0, False, 0]
		count = 0


		while not rospy.is_shutdown():
			if not all(i == True for i in self.jeffChecker):
				self.jeff_found = self.faceInPic()
				self.faces = self.findFace()

			state = self.getState()

			if state[0] == 0:
				if count > 0:
					break
				count += 1
				print "It is true, we should finish"


			pos = [4, 4]
			if self.rob[0] <= 1.0:
				pos[0] = 0
			elif self.rob[0] >= 8.0:
				pos[0] = 9

			if self.rob[1] <= 1.0:
				pos[1] = 0
			elif self.rob[1] >= 8.0:
				pos[1] = 9

			robAng = self.approxAngle(self.rob[2]) 
			action = Gridworld.maxAction(self.Q, state, self.env.possibleActions, pos)


			print "State: ", state, " Action-Angle: ", [action, self.actionSpace[action], robAng], self.jeffChecker, self.jeff_found
			print "Jeffs' Positions: ", self.jeff

			self.setSpeed(self.actionSpace[action], state, robAng, old_state)
			
			self.pub.publish(self.speed)


			old_state = state
			freq.sleep()

	def findFace(self):
		
		list_of_files = glob.glob('/home/peter/thesis_ws/src/add_jeff/src/camera_save_tutorial/*.jpg')
		if len(list_of_files) > 0:
			latest_file = max(list_of_files, key = os.path.getctime)
		
		face = client.contactServer(self.sock, latest_file)

		return face
	
	def faceModulo(self, z):

		if z > 180:
			z =-360 + z 
		elif z < -180:
			z = 360 + z
		return z


	def faceInPic(self):
		facePos = [] #Both Zeros no face.
		ro = self.rob[2]*(180/pi) #Robots Angle

		angPos = self.getAngle(self.jeff[0])
		angNeg = self.getAngle(self.jeff[1])

		anglePos = self.faceModulo(angPos*(180/pi) - ro)
		angleNeg = self.faceModulo(angNeg*(180/pi) - ro)


		if (30 > anglePos > 0) and (30 > angleNeg > 0): #if both to the left
			if angleNeg < anglePos:
				return [self.jeff_name[1], self.jeff_name[0]]
			else:
				return [self.jeff_name[0], self.jeff_name[1]]
		elif (-30 < anglePos < 0) and (-30 < angleNeg < 0): #if both to the right
			if angleNeg < anglePos:
				return [self.jeff_name[0], self.jeff_name[1]]
			else:
				return [jeff_name[1], jeff_name[0]]

		if 30 > anglePos > -30: #If only Pos to the left
			facePos.append(self.jeff_name[0])


		if 30 > angleNeg > 0: #To the left
			facePos.insert(0, self.jeff_name[1])

		elif -30 < angleNeg < 0: #To the right
			facePos.append(self.jeff_name[1])


		return facePos

		

	def findJeff(self):
		freq = rospy.Rate(4)


		while not rospy.is_shutdown():
			self.jeff_found = self.faceInPic()

			self.speed.linear.x = 0.0
			self.speed.angular.z = 0.1
			self.pub.publish(self.speed)

			self.faces = self.findFace()

			if face:
				break 


			freq.sleep()


	def callback(self, msg):
		d = [21, 21]
		for ind in range(len(self.jeff_found)):
			if 0 <= ind < len(self.faces):
				jeff_ind = self.getModelIndex(msg, self.jeff_found[ind])
				if self.jeffChecker[self.jeff_name.index(self.jeff_found[ind])]:
					continue

				if len(self.faces) == 1:
					d[ind] = self.getDist([msg.pose[jeff_ind].position.x, msg.pose[jeff_ind].position.y])

				else:
					if self.faces[ind] is "1":
						self.jeff[0][0] = msg.pose[jeff_ind].position.x
						self.jeff[0][1] = msg.pose[jeff_ind].position.y
						self.jeffChecker[0] = True
					elif self.faces[ind] is "2":
						self.jeff[1][0] = msg.pose[jeff_ind].position.x
						self.jeff[1][1] = msg.pose[jeff_ind].position.y
						self.jeffChecker[1] = True
			

		if not all(i == 21 for i in d):
			ind = d.index(min(d))
			if not self.jeffChecker[self.jeff_name.index(self.jeff_found[ind])]:
				jeff_ind = self.getModelIndex(msg, self.jeff_found[ind])
				self.jeff[int(self.faces)-1][0] = msg.pose[jeff_ind].position.x
				self.jeff[int(self.faces)-1][1] = msg.pose[jeff_ind].position.y
			
		'''for ind, jef in enumerate(self.jeff_found):
			jeff_ind = self.getModelIndex(msg, jef)


			self.jeff[ind][0] = msg.pose[jeff_ind].position.x
			self.jeff[ind][1] = msg.pose[jeff_ind].position.y'''

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
	setPos.angleToJeff()#setPos.findJeff() #
	print "We are outside"

	#img = plt.imread(BG_PIC)
	#fig, ax = plt.subplots()
	#ax.imshow(img, extent=[0,9,0,9])

	plt.scatter(setPos.jeff[0][0], setPos.jeff[0][1], s=1000, color='#008000', marker='o')
	plt.scatter(setPos.jeff[1][0], setPos.jeff[1][1], s=1000, color='#8B0000', marker='o')
	xaxis = [i[0] for i in setPos.path]
	yaxis = [i[1] for i in setPos.path]
	xaxis2 = [i[0] for i in setPos.pathWithFace]
	yaxis2 = [i[1] for i in setPos.pathWithFace]

	plt.scatter(xaxis[0::30], yaxis[0::30], s=25, color='#000000')
	plt.scatter(xaxis2[0::30], yaxis2[0::30], s=25, color='#FFFF00')

	plt.scatter(xaxis[-1], yaxis[-1], s=25, color='#000000')
	plt.scatter(setPos.path[0][0], setPos.path[0][1], s=1000, color='#0000FF', marker='o')

	plt.xlim(0, 9)
	plt.ylim(0, 9)

	plt.savefig('QLearning_path.png')  # Save plot
	plt.show()

	#s.close()



if __name__ == '__main__':
	listener()
