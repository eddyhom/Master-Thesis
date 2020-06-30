#!/usr/bin/env python2
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
		self.env = Gridworld.Gridworld(w, h)
		self.pub = pub #Publisher handle
		self.rob = [0,0,0]

		self.jeff = [0, 0] #x, y position
		self.jeff_name = 'jeff' #Person's name
		
		self.speed = Twist()

		self.discreteDist = self.discDist()
		self.discreteAng = self.discAng()

		self.Q = self.getQ()
		self.state = []


	def getModelIndex(self, msg, model_name):
		return msg.name.index(model_name)

	def getQ(self):
		f = open('Qtable11.pkl', 'rb') #Open Qtable file to use what was learned
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
		return [[-5, 5],[5, 85],[85, 95],[95, 175],[-175, -95],[-95, -85], [-85, -5]]

	def approxDist(self, dist):
		return min(self.discreteDist, key=lambda x:abs(x-dist))


	def approxAngle(self, angle):
		angle = round(angle*(180/pi))

		if angle > 175 or angle < -175:
			return 4

		for i in range(len(self.discreteAng)):
			if angle in range(self.discreteAng[i][0],self.discreteAng[i][1]):
				return i+1 if i > 3 else i


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

	def angleToJeff(self):
		freq = rospy.Rate(4)
		state = [0, 0]
		old_state = [0, 0]


		while not rospy.is_shutdown():
			state = [self.getDist(), self.getAngle()]
			action = self.maxAction(self.Q, state, self.env.possibleActions)

			print "Take action", action

			#observation_, _, done, _ = self.env.step(action)

			'''inc_x = self.jeff[0] - self.rob[0] ###Remove
			inc_y = self.jeff[1] - self.rob[1] ###Remove
			angle_to_jeff = atan2(inc_y, inc_x) ##Remove

			final_angle = angle_to_jeff - self.rob[2] ##Remove

			print("Current state is: ", state, " and total angle is: ", angle_to_jeff*(180/pi))

			if final_angle > 0.1: #Turn clockwise
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.0#0.3
			elif final_angle < -0.1: #Turn counter-clockwise
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.0#-0.3
			else: #Go straight
				self.speed.linear.x = 0.0#-0.2
				self.speed.angular.z = 0.0

			
			if sqrt(inc_x**2 + inc_y**2) < 1.5:
				self.speed.linear.x = 0.0
				self.speed.angular.z = 0.0
				self.pub.publish(self.speed)
			else:
				self.pub.publish(self.speed)

			old_state = state'''
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
