#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2, pi, sqrt
import pickle
import Gridworld #We use the training code for the class GridWorld

class setRobotPosition(object):
	def __init__(self, pub):
		w = 9
		h = 9
		self.env = Gridworld.GridWorld(w, h)
		self.pub = pub #Publisher handle
		self.rob = [0,0,0]

		self.jeff = [0, 0] #x, y position
		self.jeff_name = 'jeff' #Person's name
		
		self.speed = Twist()

		self.discreteDist = self.discDist()
		self.discreteAng = self.discAng()

		self.Q = self.getQ()
		self.state = []
		
		self.actionSpace = {'R': 0, 'UR': 1, 'U': 2,  'UL': 3, 
				    'L': 4 ,'DL': 5, 'D': 6, 'DR': 7}
                             
                            
                             


	def getModelIndex(self, msg, model_name):
		return msg.name.index(model_name)

	def getQ(self):
		f = open('Qtable12.pkl', 'rb') #Open Qtable file to use what was learned
		Q, reward = pickle.load(f) #Set Qtable to Q, ignore reward its not needed here
		return Q

	def discDist(self):
		w = 9
		h = 9
		l = []
		for i in range(w+1):
			for j in range(h+1):
				dx = w - i
				dy = h - j
				l.append(round(sqrt(dx**2+dy**2),2))
		return l

	def discAng(self):
		return [[-15, 15],[15, 75],[75, 105],[105, 165],[-165, -105],[-105, -75], [-75, -15]]

	def approxDist(self, dist):
		return min(self.discreteDist, key=lambda x:abs(x-dist))


	def approxAngle(self, angle):
		angle = round(angle*(180/pi))

		if angle > 165 or angle < -165:
			return 4

		for i in range(len(self.discreteAng)):
			if angle in range(self.discreteAng[i][0],self.discreteAng[i][1]):
				return i+1 if i > 3 else i
		return 0


	def getDist(self):
		inc_x = self.jeff[0] - self.rob[0]
		inc_y = self.jeff[1] - self.rob[1]	
		return self.approxDist(sqrt(inc_x**2 + inc_y**2))		

	def getAngle(self):
		inc_x = self.jeff[0] - self.rob[0]
		inc_y = self.jeff[1] - self.rob[1]	

		angle_to_jeff = atan2(inc_y, inc_x)
		return self.approxAngle(angle_to_jeff)

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
				 
			
		

	def setSpeed(self, action, state, rob, old_state):
		if old_state == state:
			if state[0] == 0:
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.0
			elif action == rob: #Go Straight
				self.speed.linear.x = -0.15
				self.speed.angular.z = 0.0
			else:
				ccw = self.rotateCCW(action, rob)
				self.speed.linear.x = 0.0
				if ccw:
					self.speed.angular.z = 0.15
				else:
					self.speed.angular.z = -0.15
		else:
			if state[0] == 0:
				self.speed.linear.x = self.speed.linear.x * 0.5
				self.speed.angular.z = self.speed.angular.z * 0.5
			elif action == rob: #Go Straight
				self.speed.linear.x = self.speed.linear.x * 0.5
				self.speed.angular.z = self.speed.linear.z * 0.5
			else:
				ccw = self.rotateCCW(action, rob)
				self.speed.linear.x = 0.0
				if ccw:
					self.speed.angular.z = self.speed.linear.z * 0.5
				else:
					self.speed.angular.z = self.speed.linear.z * 0.5
	

	def getState(self):
		if self.getDist() == 0:
			return tuple([0, 0])
		else:
			return tuple([self.getDist(), self.getAngle()])
				
	def angleToJeff(self):
		freq = rospy.Rate(2)
		state = [0, 0]
		old_state = [0, 0]


		while not rospy.is_shutdown():
			state = self.getState()
			robAng = self.approxAngle(self.rob[2])
			action = Gridworld.maxAction(self.Q, state, self.env.possibleActions)


			print "Taken action", self.actionSpace[action], " Action that should be taken: ", state[1], " angle where we are: ", robAng

			self.setSpeed(self.actionSpace[action], state, robAng, old_state)
			
			self.pub.publish(self.speed)

			#observation_, _, done, _ = self.env.step(action)

			'''inc_x = self.jeff[0] - self.rob[0] ###Remove
			inc_y = self.jeff[1] - self.rob[1] ###Remove
			angle_to_jeff = atan2(inc_y, inc_x) ##Remove

			final_angle = angle_to_jeff - self.rob[2] ##Remove

			print("Current state is: ", state, " and total angle is: ", angle_to_jeff*(180/pi))

			if final_angle > 0.1: #Turn clockwise
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.3
			elif final_angle < -0.1: #Turn counter-clockwise
				self.speed.linear.x = 0.0
				self.speed.angular.z = -0.3
			else: #Go straight
				self.speed.linear.x = -0.2
				self.speed.angular.z = 0.0

			
			if sqrt(inc_x**2 + inc_y**2) < 1.5:
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.0
				self.pub.publish(self.speed)
			else:
				self.pub.publish(self.speed)'''

			old_state = state
			freq.sleep()


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
	setPos = setRobotPosition(pub)
	

	rospy.Subscriber("/gazebo/model_states", ModelStates, setPos.callback)
	rospy.Subscriber("/my_vehicle2/odom", Odometry, setPos.callback2)
	setPos.angleToJeff()



if __name__ == '__main__':
	listener()
