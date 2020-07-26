#!/usr/bin/env python2
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2, pi, sqrt, floor
import pickle
import QLearning_2Persons #We use the training code for the class GridWorld
import client
import glob
import os
import socket
import time

PORT = 1026
CLIENT_IP = '192.168.1.36'
SERVER_IP = '192.168.1.33'
Q_LINK = 'Qtable_2Emotions_500k_fversion.pkl'

class setRobotPosition(object):
	def __init__(self, pub, s):
		w = 9
		h = 9
		self.env = Gridworld.GridWorld(w, h)
		self.pub = pub #Publisher handle
		self.sock = s
		self.rob = [0,0,0]

		self.jeff = [[0, 0], [0, 0]] #x, y position
		self.jeff_name = ['jeff2', 'jeff3']#Person's name - jeff2 = Happy, jeff3 = Angry
		
		self.speed = Twist()

		self.discreteDist, self.discreteDist2  = self.discDist()
		self.discreteAng = self.discAng()

		self.Q = self.getQ()
		self.state = []
		
		self.actionSpace = {'R': 0, 'UR': 1, 'U': 2,  'UL': 3, 
				    'L': 4 ,'DL': 5, 'D': 6, 'DR': 7}

                             
                                                 
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
		#print "Rotate to: ", rotTo, " rotate from: ", rotFrom
		while True:
			ccw += 1
			cw -= 1
			if ccw % 8 == rotTo:
				return True
			elif cw % 8 == rotTo:
				return False
				 
	def getSpeed(self):
		angle = self.getAngle()
		angle = round(angle*(180/pi))
		robAng = round(self.rob[2]*(180/pi))
		dist = self.getDist()

		maxAngle = 180
		maxDist = 11.5
		
		linVel = dist/maxDist
		angVel = (angle - robAng)/maxAngle


		return linVel, angVel
		
		
		

	def setSpeed(self, action, state, rob, old_state):
		linVel, angVel = self.getSpeed()
		if state[0] == 0:
			self.speed.linear.x = 0.0
			self.speed.angular.z = 0.0
		elif action == rob: #Go Straight
			
			self.speed.linear.x = -0.2 * linVel - 0.03
			
			self.speed.angular.z = 0.1 * angVel + 0.01 if angVel > 0 else 0.1 * angVel - 0.01#Negative clockwise, positive counter-clockwise
		else:
			ccw = self.rotateCCW(action, rob)
			self.speed.linear.x = 0.0
			if ccw:
				self.speed.angular.z = 0.3 * abs(angVel) + 0.02
			else:
				self.speed.angular.z = -0.3 * abs(angVel) - 0.02
	

	def getState(self):
		ang = self.approxAngle(self.getAngle())
		dist = self.getDist() #Send if angle is odd or even
		dist = self.approxDist(dist, ang % 2)
		if dist == 0:
			return tuple([0, 0])
		else:
			return tuple([dist, ang])
				
	def angleToJeff(self):
		freq = rospy.Rate(10)
		state = [0, 0]
		old_state = [0, 0]


		while not rospy.is_shutdown():
			state = self.getState()

			robAng = self.approxAngle(self.rob[2])
			action = Gridworld.maxAction(self.Q, state, self.env.possibleActions)


			print "State: ", state, " Action-Angle: ", [self.actionSpace[action], robAng], " Velocities: ", [round(self.speed.linear.x,4), round(self.speed.angular.z,4)]

			self.setSpeed(self.actionSpace[action], state, robAng, old_state)
			
			self.pub.publish(self.speed)


			old_state = state
			freq.sleep()

	def findFace(self):
		
		list_of_files = glob.glob('/home/peter/thesis_ws/src/add_jeff/src/camera_save_tutorial/*.jpg')
		latest_file = max(list_of_files, key = os.path.getctime)
		

		#face = client.contactServer(self.sock, latest_file)
		#print face

		return False
	


	def findJeff(self):
		freq = rospy.Rate(4)


		while not rospy.is_shutdown():

			self.speed.linear.x = 0.0
			self.speed.angular.z = 0.0
			self.pub.publish(self.speed)

			face = self.findFace()

			print self.jeff

			if face:
				break 


			freq.sleep()


	def callback(self, msg):
		for ind, jef in enumerate(self.jeff_name):
			jeff_ind = self.getModelIndex(msg, jef)


			self.jeff[ind][0] = msg.pose[jeff_ind].position.x
			self.jeff[ind][1] = msg.pose[jeff_ind].position.y

	def callback2(self, msg):
		self.rob[0] = msg.pose.pose.position.x
		self.rob[1] = msg.pose.pose.position.y

		rob_ori = msg.pose.pose.orientation	
		(roll, pitch, theta) = euler_from_quaternion([rob_ori.x, rob_ori.y, rob_ori.z, rob_ori.w ])
		self.rob[2] = theta



def listener():
	rospy.init_node('listener', anonymous=True)
	pub = rospy.Publisher("/my_vehicle2/cmd_vel", Twist, queue_size=1)

	#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	#s.connect((SERVER_IP, PORT))
	s = 0

	setPos = setRobotPosition(pub, s)
	

	rospy.Subscriber("/gazebo/model_states", ModelStates, setPos.callback)
	rospy.Subscriber("/my_vehicle2/odom", Odometry, setPos.callback2)
	setPos.findJeff() #setPos.angleToJeff()#

	#s.close()



if __name__ == '__main__':
	listener()
