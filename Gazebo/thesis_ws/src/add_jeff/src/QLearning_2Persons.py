#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from math import floor, sqrt, ceil
import pickle
from random import randint, random

width = 100
steppedOnNeg = -100

class GridWorld(object):
    def __init__(self, w, h):  # w and h are the width and height of the space
        self.w = w  # Width
        self.h = h  # Height
        self.gl = 0
        self.history = []

        self.agentPosition = [0, 0]  # Robots position in x, y coordinates
        self.goal = [self.w, self.h]  # Positive Person's position in x, y coordinates
        x, y = ceil(self.w / 2), ceil(self.h / 2)
        self.negative = [x, y]  # Negative person's position in x and y

        states = []  # All the discrete states
        for i in range(self.w + 1):  # Create states from all the distances along the width
            for j in range(self.h + 1):  # Create states from all the distances along the height
                dx = self.goal[0] - i  # Distance to goal from the i:th position
                dy = self.goal[1] - j  # Distance to goal from the j:th position
                for quadrant in range(0,
                                      8):  # Angle to goal from robot according to the x-plane and -y-plane (negative y is positive here) -See getQuadrant()
                    for quadrant2 in range(0, 8):
                        dist = round(sqrt(dx ** 2 + dy ** 2), 2)  # Calculate distance with pythagoras
                        for distance in range(2):
                            near = True
                            if distance != 1:
                                near = False

                            if (dist, quadrant, near,
                                quadrant2) not in states and dist > 0:  # Add the states to state list, if they're not repeated and if dist is not 0.
                                if dx == 0 or dy == 0:
                                    if quadrant % 2 == 0:  # If pair
                                        if near:
                                            states.append((dist, quadrant, near,
                                                           quadrant2))  # State space distance to goal (for now every meter)
                                        else:
                                            states.append((dist, quadrant, near, 0))
                                else:
                                    if quadrant % 2 == 1:
                                        if near:
                                            states.append((dist, quadrant, near,
                                                           quadrant2))  # State space distance to goal (for now every meter)
                                        else:
                                            states.append((dist, quadrant, near, 0))

        self.stateSpace = states  # Here are all the states saved
        self.state = states[0]  # Initial state

        self.possibleActions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']  # Possible actions that robot can take
        self.actionSpace = {'U': [0, -1], 'D': [0, 1],
                            'L': [-1, 0], 'R': [1, 0],
                            'UL': [-1, -1], 'UR': [1, -1],
                            'DL': [-1, 1], 'DR': [1, 1]}  # How to take each action

    def isTerminalState(self, state):  # Return True if is final state, else return False
        if abs(state[0] - 0) <= 0.01:  # if distance to goal is ~0, robot is in goal
            return True
        else:
            return False

    def setAgentPosition(self, action, newState):
        # Takes the new action as input and changes robots position and current state based on new position
        self.agentPosition = [self.agentPosition[0] + self.actionSpace[action][0],
                              self.agentPosition[1] + self.actionSpace[action][1]]

        self.state = newState

    def getQuadrant(self, newPosition, person):  # Get angle to goal from new position
        # Angles are divided in 8 options, same options as possibleActions Right = R, Up-Right = UR, etc..
        # The numbers below represent the Directions.
        # 0 = R, 1 = UR, 2 = U,  3 = UL,  4 = L,   5 = DL,  6 = D,   7 = DR  --- In actions
        # 0 = 0, 1 = 45, 2 = 90, 3 = 135, 4 = 180, 5 = 225, 6 = 270, 7 = 315 --- In Angles

        pos = [person[0] - newPosition[0], person[1] - newPosition[1]]

        if all([v == 0 for v in pos]):
            return 0

        if pos[0] > 0:
            if pos[1] > 0:
                return 7
            elif pos[1] < 0:
                return 1
            else:
                return 0
        elif pos[0] < 0:
            if pos[1] > 0:  # [-1, 1] => DL = 5 ?
                return 5
            elif pos[1] < 0:
                return 3
            else:
                return 4
        else:
            if pos[1] > 0:
                return 6
            elif pos[1] < 0:
                return 2

    def getDist(self, position, person):  # Get distance from robot to goal with pythagoras
        dx = person[0] - position[0]
        dy = person[1] - position[1]

        return round(sqrt(dx ** 2 + dy ** 2), 2)

    def getState(self, newState):  # Calculate new state by calculating distance and angle.
        resultingPosition = [self.agentPosition[0] + self.actionSpace[newState][0],
                             self.agentPosition[1] + self.actionSpace[newState][1]]

        negative_dist = self.getDist(resultingPosition, self.negative)
        dx = self.negative[0] - resultingPosition[0]
        dy = self.negative[1] - resultingPosition[1]

        nearby = False
        quadrant2 = 0

        if negative_dist < 1.42:
            if dx == 0 or dy == 0:
                if negative_dist <= 1.0:
                    nearby = True
                    quadrant2 = self.getQuadrant(resultingPosition, self.negative)
            else:
                nearby = True
                quadrant2 = self.getQuadrant(resultingPosition, self.negative)

        return tuple([self.getDist(resultingPosition, self.goal),
                      self.getQuadrant(resultingPosition, self.goal), nearby, quadrant2])

    def setReward(self, newState, newQuadrant, resultingPosition):
        beenThere = 1 if resultingPosition not in self.history else 2
        if newState[1] == newQuadrant:  # If we are going in the right direction - give small punishment
            return -1 * beenThere  # Best Case Scenario where it goes a straight direction
        else:  # If we're not going in the wrong direction - give bigger punishment
            if self.getDist(resultingPosition, self.negative) <= self.getDist(self.agentPosition, self.negative):
                return -2 * beenThere  # Wrong direction but getting closer
            else:  # Wrong direction and getting farther away
                return -3 * beenThere

    def giveReward(self, newState, action):
        if self.isTerminalState(newState):
            return 0

        resultingPosition = [self.agentPosition[0] + self.actionSpace[action][0],
                             self.agentPosition[1] + self.actionSpace[action][1]]

        actionToQuadrant = {'U': 2, 'D': 6, 'L': 4, 'R': 0,
                            'UL': 3, 'UR': 1, 'DL': 5,
                            'DR': 7}  # Table to translate actions into "angles" - See getQuadrant() for more info

        newQuadrant = actionToQuadrant[action]  # Quadrant we actually going

        if self.getDist(resultingPosition, self.negative) == 0:  # If negative person is stepped on
            return steppedOnNeg

        return self.setReward(newState, newQuadrant, resultingPosition)


    def step(self, action):
        resultingState = self.getState(action)  # Calculate new state based on action taken
        reward = self.giveReward(resultingState, action)  # Give reward based on action taken

        if reward == 0:  # If robot at goal, return state = (0,0), reward = 0, Finished = True.
            return (0, 0, False, 0), reward, True, None
        elif reward == steppedOnNeg:
            return self.state, reward, False, None

        self.setAgentPosition(action, resultingState)  # Change the current state and return new state, reward, finish..
        return resultingState, reward, self.isTerminalState(self.state), None

    def reset(self):  # Dont mind this, it is changed later on anyways!!!!!!!!! But it's used in testQL.py
        self.goal = [randint(0, self.w), randint(0, self.h)]
        self.agentPosition = [randint(0, self.w), randint(0, self.h)]
        self.gl = 0
        if self.getDist(self.agentPosition, self.goal) > 0:
            return tuple([self.getDist(self.agentPosition, self.goal), self.getQuadrant(self.agentPosition, self.goal),
                          False, 0])
        else:
            self.goal = [self.w, self.h]
            self.agentPosition = [0, 0]
            return tuple([self.getDist(self.agentPosition, self.goal), self.getQuadrant(self.agentPosition, self.goal),
                          False, 0])


    def actionSpaceSample(self):  # Choose a random action
        removedActions = self.possibleActions[:]
        position = self.agentPosition

        for i in self.possibleActions:
            if position[0] == 0:  # If all the way left, cant go more left
                if 'L' in i:
                    removedActions.remove(i)
            elif position[0] == 9:
                if 'R' in i:
                    removedActions.remove(i)
            if position[1] == 0:
                if 'U' in i and i in removedActions:
                    removedActions.remove(i)
            elif position[1] == 9:
                if 'D' in i and i in removedActions:
                    removedActions.remove(i)

        return np.random.choice(removedActions)


def maxAction(Q, state, actions, position):  # Choose best action from current state
    # want to take agents estimate of present value of expected future rewards for state
    # and all possible actions. Also want to take the maximum of that
	removedActions = actions[:]
	for i in actions:
		if position[0] == 0:  # If all the way left, cant go more left
			if 'L' in i:
				removedActions.remove(i)
		elif position[0] == 9:
			if 'R' in i:
				removedActions.remove(i)

		if position[1] == 0:
			if 'D' in i and i in removedActions:
				removedActions.remove(i)
		elif position[1] == 9:
			if 'U' in i and i in removedActions:
				removedActions.remove(i)

	# print(position, removedActions)
	values = np.array([Q[state, a] for a in removedActions])
	act = 0


	if len(values) > 0:
		action = np.argmax(values)
		act = removedActions[action]
	else:
		act = np.random.choice(actions)

	return act


if __name__ == '__main__':
    width = 9  # Width of map
    height = 9  # Height of map

    env = GridWorld(width, height)  # Create new GridWorld class

    # model hyperparameters
    learning_rate = 0.1  # Determines to what extent newly acquired info overrides old info.
    discount = 1.0  # Determines the importance of future rewards.
    EPS = 1.0  # Determines randomness of action taking - High in the beginning low at the end.

    Q = {}
    # Create a value of state action pairs
    for state in env.stateSpace:
        for action in env.possibleActions:
            Q[state, action] = 0

    # Create 0 distance for goal..
    for action in env.possibleActions:
        Q[(0, 0, False, 0), action] = 0

    numGames = 500000  # Number of iterations
    stopLearning = numGames * 0.9  # Stop Learning after 80%
    totalRewards = np.zeros(numGames)  # Keeps track of total rewards
    rend = False  # Rend iteration or not

    count = 0
    flag = 0

    for i in range(numGames):
        # print("Iteration number: ", i-1, " went ouside: ", env.gl)
        if i % 5000 == 0:  
            print('starting game ', i, " Learning Rate: ", EPS)


        done = False  # Robot reaches goal
        epRewards = 0  # Amount of rewards for iteration

        observation = env.reset()  # Reset environment - Returns Current state

        # Change what happened in reset (this could actually replace what is in env.reset()) but what is in reset its used in testQL.py
        # in Every iteration it changes robot and goal to opposite corners for a deeper search...
        if count >= 0:
            combi = [[(0, 0), (9, 9)], [(0, 9), (9, 0)]]
            if count > len(combi) - 1:
                count = 0
                flag += 1

            env.goal = [combi[count][flag % 2][0], combi[count][flag % 2][1]]
            env.agentPosition = [combi[count][(flag + 1) % 2][0], combi[count][(flag + 1) % 2][1]]


            combi2 = [(0, 1), (2, 7), (8, 9)]
            ind = randint(0, 2)
            env.negative[0] = randint(combi2[ind][0], combi2[ind][1])
            if ind == 1:
                env.negative[1] = randint(0, 9)
            elif env.goal == [0, 0] or env.goal == [9, 9]:
                if ind == 0:
                    env.negative[1] = randint(2, 9)
                else:
                    env.negative[1] = randint(0, 7)
            else:
                if ind == 2:
                    env.negative[1] = randint(2, 9)
                else:
                    env.negative[1] = randint(0, 7)

            nearby = False
            quadrant2 = 0
            negative_dist = env.getDist(env.agentPosition, env.negative)

            if negative_dist < 1.5:
                nearby = True
                quadrant2 = env.getQuadrant(env.agentPosition, env.negative)

            observation = tuple([env.getDist(env.agentPosition, env.goal), env.getQuadrant(env.agentPosition, env.goal),
                                 nearby, quadrant2])
            count += 1

        while not done:
            if len(env.history) < 20:
                env.history.append(env.agentPosition)
            else:
                env.history.append(env.agentPosition)
                env.history.pop(0)

            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions, env.agentPosition) if rand < (1 - EPS) \
                else env.actionSpaceSample()  # Take best action or random action based on EPS

            # Get new state and reward from environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward  # Amount of rewards for this iteration

            action_ = maxAction(Q, observation_, env.possibleActions, env.agentPosition)  # Choose maxAction

            # Update Q-Table with new knowledge
            Q[observation, action] = Q[observation, action] + learning_rate * (reward + \
                                                                               discount * Q[observation_, action_] - Q[
                                                                                   observation, action])
            observation = observation_  # Environment has changed state (state = new_state)


        if EPS - 2 / numGames > 0:
            EPS -= 1 / stopLearning  # Lower Randomness after each iteration to start taking Best action instead of random action.
        else:
            EPS = 0
        totalRewards[i] = epRewards

    with open('Qtable_2Emotions_500k_fversion.pkl',
              'wb') as f:  # Save QTable as 'Qtable.pkl' you can change name to not overwrite older versions
        pickle.dump([Q, totalRewards], f)

    plt.plot(totalRewards)  # Plot learning
    plt.savefig('8Action_500k_Iterations_2Emotions_fversion.png')  # Save plot
    plt.show()  # Show plot
