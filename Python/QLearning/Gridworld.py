import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import floor, sqrt

import os
from time import sleep

width = 600


class GridWorld(object):
    def __init__(self, w, h): #m and n, shape of the grid
        self.w = w  # Width
        self.h = h  # Height
        self.stateSpace = [i for i in range(floor(sqrt(self.h * self.h + self.w * self.w)))]  # State space distance to goal (for now every meter)

        self.actionSpace = {'U': [0, -1], 'D': [0, 1],
                            'L': [-1, 0], 'R': [1, 0]}  # How to take each action
        '''UL': -self.m-1,  'UR': -self.m+1, 'DL': self.m-1, 'DR': self.m+1}'''
        self.possibleActions = ['U', 'D', 'L', 'R']  # Possible actions that robot can take

        self.agentPosition = [0, 0]  # Robots position

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y

    def setState(self, state): #Takes the new state as input
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0 #0 denotes an empty square
        self.agentPosition = state #Agent position is the new state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1 #1 represents the column

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

    def step(self, action):
        x, y = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.offGridMove(resultingState, self.agentPosition): #If not movinf off grid
            self.setState(resultingState)
            return resultingState, reward, \
                   self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, \
                   self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.grid[-1][-1] = 2
        return self.agentPosition

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

if __name__ == '__main__':
    # map magic squares to their connecting square
    grids = 11
    env = GridWorld(grids, grids)
    # model hyperparameters
    learning_rate = 0.1 #Determines to what extent newly acquired info overrides old info.
    discount = 1.0 #Determines the importance of future rewards.
    EPS = 1.0

    Q = {}
    #Want to find a value of state action pairs
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames) #Keeps track of total rewards
    rend = False

    for i in range(numGames):
        print('starting game ', i)
        if i % 5000 == 0:
            print('starting game ', i)
            rend = True
            pygame.init()
            win = pygame.display.set_mode((width, width))
            pygame.display.set_caption("First Game")
        else:
            rend = False
            pygame.quit()

        done = False
        epRewards = 0
        observation = env.reset() #Reset environment

        while not done:
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample()

            #Get new state and reward from environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            '''if epRewards < -2000:
                epRewards = -3000
                done = True'''

            action_ = maxAction(Q, observation_, env.possibleActions)

            #Update Q-Table with new knowledge
            Q[observation, action] = Q[observation, action] + learning_rate*(reward + \
                        discount*Q[observation_, action_] - Q[observation, action])
            observation = observation_ #Environment has changed state (state = new_state)

            if rend:
                pygame.time.delay(50)
                env.render()

        if EPS - 2 / numGames > 0:
            #EPS = math.sqrt(EPS)
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
        print(epRewards)

    plt.plot(totalRewards)
    plt.show()