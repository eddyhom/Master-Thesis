import numpy as np
import matplotlib.pyplot as plt
import pygame
import math

import os
from time import sleep

width = 600


class GridWorld(object):
    def __init__(self, m, n):
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove((self.m * self.n) - 1)
        self.stateSpacePlus = [i for i in range(self.m * self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1, 'UL': -self.m - 1, 'UR': -self.m + 1, 'DL': self.m - 1, 'DR': self.m + 1}
        self.possibleActions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']
        # dict with magic squares and resulting squares
        self.agentPosition = 0

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y

    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid - its not gonna be in stateSpacePlu
        if newState not in self.stateSpacePlus:  # Go over 1st row or under m:th row
            #print("movement off grid")
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            #print("movement off grid")
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            #print("movement off grid")
            return True
        else:
            #print("movement INSIDE")
            return False

    def step(self, action):
        if len(action) == 1:
            resultingState = self.agentPosition + self.actionSpace[action]

            reward = -1 if not self.isTerminalState(resultingState) else 0
            if not self.offGridMove(resultingState, self.agentPosition):
                self.setState(resultingState)
                return resultingState, reward, \
                       self.isTerminalState(resultingState), None
            else:
                return self.agentPosition, reward, \
                       self.isTerminalState(self.agentPosition), None
        else:
            #print("This is action: " + action[0] + str(len(action)))
            resultingState = self.agentPosition + self.actionSpace[action[0]]

            if not self.offGridMove(resultingState, self.agentPosition):
                resulState = resultingState
                resultingState = resultingState + self.actionSpace[action[1]]
                reward = -1 if not self.isTerminalState(resultingState) else 0

                if not self.offGridMove(resultingState, resulState):
                    self.setState(resultingState)
                    return resultingState, reward, \
                           self.isTerminalState(resultingState), None
                else:
                    return self.agentPosition, reward, \
                           self.isTerminalState(self.agentPosition), None
            else:
                reward = -1
                return self.agentPosition, reward, \
                       self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.grid[-1][-1] = 2
        return self.agentPosition

    def render(self):

        w = width / (self.m + 2)  # width of each cell
        x, y = w, w  # starting position
        border = w - 2
        robot_size = border / 3
        robot_pos = w / 2 - robot_size / 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        for row in self.grid:
            for col in row:
                if col == 0:
                    pygame.draw.rect(win, (0, 0, 0), (x + 1, y + 1, border, border))
                elif col == 1:
                    pygame.draw.rect(win, (0, 0, 0), (x + 1, y + 1, border, border))
                    pygame.draw.rect(win, (250, 0, 0), (x + robot_pos, y + robot_pos, robot_size, robot_size))
                elif col == 2:
                    pygame.draw.rect(win, (0, 128, 0), (x + 1, y + 1, border, border))

                x = x + w  # move right
            y = y + w  # move down
            x = w  # rest to left edge
        pygame.display.update()
        return True

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)


def maxAction(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]


if __name__ == '__main__':
    # map magic squares to their connecting square
    grids = 11
    env = GridWorld(grids, grids)
    # model hyperparameters

    ALPHA = 0.01  # Learning Rate
    GAMMA = 1.0  # Importance of future rewards
    EPS = 1.0  # Randomness of choosing from the Q-table.

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    rend = False

    for i in range(numGames):
        print('starting game ', i)
        if (i + 1) % 5000 == 0:
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
        observation = env.reset()
        while not done:
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1 - EPS) \
                else env.actionSpaceSample()

            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation, action] = Q[observation, action] + ALPHA * (reward + \
                                                                       GAMMA * Q[observation_, action_] - Q[
                                                                           observation, action])
            observation = observation_

            if rend:
                pygame.time.delay(50)
                rend = env.render()

        if EPS - 2 / numGames > 0:
            EPS -= 1 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards



    plt.plot(totalRewards)
    plt.show()
