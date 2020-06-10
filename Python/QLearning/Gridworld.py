import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import floor, sqrt

import os
from time import sleep

width = 100


class GridWorld(object):
    def __init__(self, w, h): #m and n, shape of the grid
        self.w = w  # Width
        self.h = h  # Height
        self.direction = 'E' #Different directions are East 'E', West 'W', North 'N', South 'S'

        self.agentPosition = [0, 0]  # Robots position
        self.goal = [self.w, self.h] # Positive Person

        states = []
        for i in range(self.w+1): #Still working on this!!!!!!
            for j in range(self.h+1):
                dx = self.goal[0] - i
                dy = self.goal[1] - j
                for quadrant in range(0, 4):
                    dist = round(sqrt(dx**2 + dy**2), 2)
                    if (dist, quadrant) not in states:
                        states.append((dist, quadrant))   # State space distance to goal (for now every meter)

        self.stateSpace = states

        self.possibleActions = ['U', 'D', 'L', 'R']  # Possible actions that robot can take
        self.actionSpace = {'U': [0, -1], 'D': [0, 1],
                            'L': [-1, 0], 'R': [1, 0]}  # How to take each action


    def isTerminalState(self, state):
        #print(state)
        #print("Here im taking away 0: ", state[0]-0)
        if abs(state[0]-0) <= 0.01:
         #   print(True)
            return True
        else:
          #  print(False)
            return False


    def setAgentPosition(self, action): #Takes the new action as input
        self.agentPosition = [self.agentPosition[0] + self.actionSpace[action][0],
                          self.agentPosition[1] + self.actionSpace[action][1]]

    def offGridMove(self, action): #Change input if needed
        position = [self.agentPosition[0] + self.actionSpace[action][0],
                          self.agentPosition[1] + self.actionSpace[action][1]]

        return not (0 <= position[0] <= self.w and 0 <= position[1] <= self.h)

    def getQuadrant(self, newPosition):
        pos = [self.goal[0] - newPosition[0], self.goal[1] - newPosition[1]]

        if pos[0] >= 0:
            if pos[1] >= 0:
                return 3
            else:
                return 0
        else:
            if pos[1] >= 0:
                return 2
            else:
                return 1 #Returns in what quadrant the goal is with respect to the robot

    def getDist(self, position):
        dx = self.goal[0] - position[0]
        dy = self.goal[1] - position[1]

        return round(sqrt(dx ** 2 + dy ** 2), 2)

    def getState(self, newState):
        resultingPosition = [self.agentPosition[0] + self.actionSpace[newState][0],
                          self.agentPosition[1] + self.actionSpace[newState][1]]

        return tuple([self.getDist(resultingPosition), self.getQuadrant(resultingPosition)])

    def giveReward(self, newState):
        if not self.isTerminalState(newState):
            currentDist = self.getDist(self.agentPosition)
            if newState[0] < currentDist:
                return -1
            else:
                return -5
        else:
            return 0

    def step(self, action):
        resultingState = self.getState(action)

        reward = self.giveReward(resultingState)

        if not self.offGridMove(action): #If not moving off grid
            self.setAgentPosition(action)
            return resultingState, reward, \
                   self.isTerminalState(resultingState), None
        else:
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), \
                   reward, \
                   self.isTerminalState([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), None

    def reset(self):
        self.agentPosition = [0, 0]  # Robots position
        return tuple([self.getDist([0, 0]), self.getQuadrant([0, 0])])

    def render(self):
        x, y = 0, 0  # starting position
        w = width / self.m  # width of each cell
        border = w - 2
        robot_size = border / 3
        robot_pos = w/2 - robot_size / 2

        for row in self.grid:
            for col in row:
                if col == 0:
                    pygame.draw.rect(win, (255, 255, 255), (x + 1, y + 1, border, border))
                elif col == 1:
                    pygame.draw.rect(win, (255, 255, 255), (x + 1, y + 1, border, border))
                    pygame.draw.rect(win, (250, 0, 0), (x + robot_pos, y + robot_pos, robot_size, robot_size))
                elif col == 2:
                    pygame.draw.rect(win, (0, 128, 0), (x + 1, y + 1, border, border))

                pygame.display.update()
                x = x + w  # move right
            y = y + w  # move down
            x = 0  # rest to left edge

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    #want to take agents estimate of present value of expected future rewards for state
    #and all possible actions. Also want to take the maximum of that
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]

def checkIfDuplicates_1(routes):
    dups = {tuple(x) for x in routes if routes.count(x) > 1}
    print(dups)
    print(len(dups))

if __name__ == '__main__':
    # map magic squares to their connecting square
    width = 9
    height = 9
    env = GridWorld(width, height)
    # model hyperparameters
    learning_rate = 0.1 #Determines to what extent newly acquired info overrides old info.
    discount = 1.0 #Determines the importance of future rewards.
    EPS = 1.0

    Q = {}
    #Want to find a value of state action pairs
    for state in env.stateSpace:
        for action in env.possibleActions:
            Q[state, action] = 0

    print(Q)
    print(len(Q))


    numGames = 50000
    totalRewards = np.zeros(numGames) #Keeps track of total rewards
    rend = False

    for i in range(numGames):
        print('starting game ', i)
        if i % 100 == 0:
            print('starting game ', i)
            #rend = True
        #    pygame.init()
        #    win = pygame.display.set_mode((env.w, env.h))
        #    pygame.display.set_caption("First Game")
        #else:
            #rend = False
            #pygame.quit()

        done = False
        epRewards = 0
        observation = env.reset() #Reset environment - Returns Agent Position....

        while not done:
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample() #Else make a random choice

            #Get new state and reward from environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            if env.agentPosition == env.goal:
                print("Agent position :", env.agentPosition, "and Done is: ", done)

            action_ = maxAction(Q, observation_, env.possibleActions)

            #Update Q-Table with new knowledge
            Q[observation, action] = Q[observation, action] + learning_rate*(reward + \
                        discount*Q[observation_, action_] - Q[observation, action])
            observation = observation_ #Environment has changed state (state = new_state)

            #if rend:
                #pygame.time.delay(50)
                #env.render()

        if EPS - 2 / numGames > 0:
            #EPS = math.sqrt(EPS)
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
        print(epRewards)

    plt.plot(totalRewards)
    plt.show()
