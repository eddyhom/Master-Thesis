import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import floor, sqrt
import pickle
from random import randint


width = 100


class GridWorld(object):
    def __init__(self, w, h):  # w and h are the width and height of the space
        self.w = w  # Width
        self.h = h  # Height

        self.agentPosition = [0, 0]  # Robots position in x, y coordinates
        self.goal = [self.w, self.h]  # Positive Person's position in x, y coordinates

        states = [] # All the discrete states
        for i in range(self.w + 1):  # Create states from all the distances along the width
            for j in range(self.h + 1): # Create states from all the distances along the height
                dx = self.goal[0] - i # Distance to goal from the i:th position
                dy = self.goal[1] - j # Distance to goal from the j:th position
                for quadrant in range(0, 8): # Angle to goal from robot according to the x-plane and -y-plane (negative y is positive here) -See getQuadrant()
                    dist = round(sqrt(dx ** 2 + dy ** 2), 2) #Calculate distance with pythagoras
                    if dx == 0 or dy == 0:
                        if quadrant % 2 == 0: #If pair
                            if (dist, quadrant) not in states and dist > 0: #Add the states to state list, if they're not repeated and if dist is not 0.
                                states.append((dist, quadrant))  # State space distance to goal (for now every meter)
                    else:
                        if quadrant % 2 == 1:
                            if (dist, quadrant) not in states and dist > 0: #Add the states to state list, if they're not repeated and if dist is not 0.
                                states.append((dist, quadrant))  # State space distance to goal (for now every meter)


        for sta in states:
            print(sta)
        print(len(states))

        self.stateSpace = states # Here are all the states saved
        self.state = states[0]  #Initial state

        self.possibleActions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']  # Possible actions that robot can take
        self.actionSpace = {'U': [0, -1], 'D': [0, 1],
                            'L': [-1, 0], 'R': [1, 0],
                            'UL': [-1, -1], 'UR': [1, -1],
                            'DL': [-1, 1], 'DR': [1, 1]}  # How to take each action

    def isTerminalState(self, state): #Return True if is final state, else return False
        if abs(state[0] - 0) <= 0.01: #if distance to goal is ~0, robot is in goal
            return True
        else:
            return False

    def setAgentPosition(self, action, newState):  # Takes the new action as input and changes robots position and current state based on new position
        self.agentPosition = [self.agentPosition[0] + self.actionSpace[action][0],
                              self.agentPosition[1] + self.actionSpace[action][1]]
        self.state = newState #tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])

    def offGridMove(self, action):  #Calculate new position, if new position outside map return True, else False
        position = [self.agentPosition[0] + self.actionSpace[action][0],
                    self.agentPosition[1] + self.actionSpace[action][1]]

        return not (0 <= position[0] <= self.w and 0 <= position[1] <= self.h)

    def getQuadrant(self, newPosition): # Get angle to goal from new position
        #Angles are divided in 8 options, same options as possibleActions Right = R, Up-Right = UR, etc..
        #The numbers below represent the Directions.
        # 0 = R, 1 = UR, 2 = U,  3 = UL,  4 = L,   5 = DL,  6 = D,   7 = DR  --- In actions
        # 0 = 0, 1 = 45, 2 = 90, 3 = 135, 4 = 180, 5 = 225, 6 = 270, 7 = 315 --- In Angles

        pos = [self.goal[0] - newPosition[0], self.goal[1] - newPosition[1]]

        if pos[0] > 0:
            if pos[1] > 0:
                return 7
            elif pos[1] < 0:
                return 1
            else:
                return 0
        elif pos[0] < 0:
            if pos[1] > 0:
                return 5
            elif pos[1] < 0:
                return 3
            else:
                return 4
        elif pos[0] == 0:
            if pos[1] >= 0:
                return 6
            elif pos[1] < 0:
                return 2

    def getDist(self, position): #Get distance from robot to goal with pythagoras
        dx = self.goal[0] - position[0]
        dy = self.goal[1] - position[1]

        return round(sqrt(dx ** 2 + dy ** 2), 2)

    def getState(self, newState): #Calculate new state by calculating distance and angle.
        resultingPosition = [self.agentPosition[0] + self.actionSpace[newState][0],
                             self.agentPosition[1] + self.actionSpace[newState][1]]

        return tuple([self.getDist(resultingPosition), self.getQuadrant(resultingPosition)])

    def giveReward(self, newState, action):
        actionToQuadrant = {'U': 2, 'D': 6, 'L': 4, 'R': 0, \
                            'UL': 3, 'UR': 1, 'DL': 5, 'DR': 7} #Table to translate actions into "angles" - See getQuadrant() for more info

        quadrant = newState[1]  # Quadrant we should go
        newQuadrant = actionToQuadrant[action]  # Quadrant we actually going

        if not self.isTerminalState(newState):
            if quadrant == newQuadrant: #If we are going in the right direction - give small punishment
                return -1  # Best Case Scenario where it goes a straight direction
            else: #If we're not going in the right direction - give bigger punishment
                return -2
        else: # If we're at Goal give punishment 0
            return 0

    def step(self, action):
        resultingState = self.getState(action) #Calculate new state based on action taken

        reward = self.giveReward(resultingState, action) #Give reward based on action taken
        if reward == 0: #If robot at goal, return state = (0,0), reward = 0, Finished = True.
            return (0, 0), 0, True, None

        if not self.offGridMove(action):  # If not moving out of boundaries
            self.setAgentPosition(action, resultingState) # Change the current state and return new state, reward, finish..
            return resultingState, reward, \
                   self.isTerminalState(resultingState), None
        else: # If moving out of boundaries, keep same state give greater punishment..
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), \
                   -7, \
                   self.isTerminalState([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), None

    def reset(self): #Dont mind this, it is changed later on anyways!!!!!!!!! But it's used in testQL.py
        self.goal = [randint(0, self.w), randint(0, self.h)]
        self.agentPosition = [randint(0, self.w), randint(0, self.h)]
        if self.getDist(self.agentPosition) > 0:
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])
        else:
            self.goal = [self.w, self.h]
            self.agentPosition = [0, 0]
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])

    def render(self, win): #Draws the robot and goal in a map
        robot_size = 40

        for event in pygame.event.get(): #Close window if "X" is clicked
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        pygame.draw.rect(win, (255, 255, 255), (50, 50, 400, 400)) #Draws working space in white
        pygame.draw.rect(win, (0, 0, 255), (50+self.agentPosition[0]*40, 50+self.agentPosition[1]*40, robot_size, robot_size))#draws robot in blue
        pygame.draw.rect(win, (0, 255, 0), (50+self.goal[0]*40, 50+self.goal[1]*40, robot_size, robot_size)) #draws goal in green

        #If you wanna draw negative person uncomment line below and adapt the names to your own - draw negative person in red
        ##pygame.draw.rect(win, (255, 0, 0), (50+self.negative[0]*40, 50+self.negative[1]*40, robot_size, robot_size)) #draws goal

        pygame.display.update()
        return True

    def actionSpaceSample(self): #Choose a random action
        return np.random.choice(self.possibleActions)


def maxAction(Q, state, actions): #Choose best action from current state
    # want to take agents estimate of present value of expected future rewards for state
    # and all possible actions. Also want to take the maximum of that
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    width = 9 #Width of map
    height = 9 #Height of map
    env = GridWorld(width, height) #Create new GridWorld class

    # model hyperparameters
    learning_rate = 0.1  # Determines to what extent newly acquired info overrides old info.
    discount = 1.0  # Determines the importance of future rewards.
    EPS = 1.0 # Determines randomness of action taking - High in the beginning low at the end.

    Q = {}
    # Create a value of state action pairs
    for state in env.stateSpace:
        for action in env.possibleActions:
            Q[state, action] = 0

    # Create 0 distance for goal..
    for action in env.possibleActions:
        Q[(0, 0), action] = 0

    #print(Q)

    numGames = 200000 #Number of iterations
    stopLearning = numGames * 0.8  # Stop Learning after 80%
    totalRewards = np.zeros(numGames)  # Keeps track of total rewards
    rend = False # Rend iteration or not

    count = 0
    count2 = 0
    flag = 0

    for i in range(numGames):
        if i % 500 == 0:
            print('starting game ', i)
        if i % 4999 == 0: #Rend every 2500 iterations
            print('starting game ', i)
            print("This is epReward: ", totalRewards[i - 1])
            print("This is EPS: ", EPS)

            rend = True
            pygame.init()
            win = pygame.display.set_mode((500, 500))

            pygame.display.set_caption("First Game")
        else:
            rend = False
            pygame.quit()

        done = False # Robot reaches goal
        epRewards = 0 # Amount of rewards for iteration

        observation = env.reset()  # Reset environment - Returns Current state

        #Change what happened in reset (this could actually replace what is in env.reset()) but what is in reset its used in testQL.py
        #in Every iteration it changes robot and goal to opposite corners for a deeper search...


        if count2 == 0:
            if count >= 0:
                combi = [[(0, 0), (9, 9)], [(0, 9), (9, 0)]]
                if count > len(combi)-1:
                    count = 0
                    flag += 1

                env.goal = [combi[count][flag % 2][0], combi[count][flag % 2][1]]
                env.agentPosition = [combi[count][(flag+1) % 2][0], combi[count][(flag+1) % 2][1]]
                observation = tuple([env.getDist(env.agentPosition), env.getQuadrant(env.agentPosition)])
                count += 1
                count2 += 1
        else:
            count2 -= 1


        while not done:
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1 - EPS) \
                else env.actionSpaceSample()  # Take best action or random action based on EPS

            # Get new state and reward from environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward # Amount of rewards for this iteration

            action_ = maxAction(Q, observation_, env.possibleActions) #Choose maxAction

            # Update Q-Table with new knowledge
            Q[observation, action] = Q[observation, action] + learning_rate * (reward + \
                                                                               discount * Q[observation_, action_] - Q[
                                                                                   observation, action])
            observation = observation_  # Environment has changed state (state = new_state)

            if rend:
                pygame.time.delay(50)
                rend = env.render(win)


        if EPS - 2 / numGames > 0:
            EPS -= 1 / stopLearning #Lower Randomness after each iteration to start taking Best action instead of random action.
        else:
            EPS = 0
        totalRewards[i] = epRewards


    with open('Qtable14.pkl', 'wb') as f: # Save QTable as 'Qtable.pkl' you can change name to not overwrite older versions
        pickle.dump([Q, totalRewards], f, protocol=2)

    plt.plot(totalRewards) #Plot learning
    plt.savefig('8Action_250k_Iterations_v1.2.png') #Save plot
    plt.show() #Show plot
