import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import floor, sqrt
import pickle
from random import randint


width = 100


class GridWorld(object):
    def __init__(self, w, h):  # m and n, shape of the grid
        self.w = w  # Width
        self.h = h  # Height
        self.direction = 'E'  # Different directions are East 'E', West 'W', North 'N', South 'S'

        self.agentPosition = [0, 0]  # Robots position
        self.goal = [self.w, self.h]  # Positive Person

        states = []
        for i in range(self.w + 1):  # Still working on this!!!!!!
            for j in range(self.h + 1):
                dx = self.goal[0] - i
                dy = self.goal[1] - j
                for quadrant in range(0, 8):
                    dist = round(sqrt(dx ** 2 + dy ** 2), 2)
                    if (dist, quadrant) not in states and dist > 0:
                        states.append((dist, quadrant))  # State space distance to goal (for now every meter)

        self.stateSpace = states
        self.state = states[0]

        self.possibleActions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']  # Possible actions that robot can take
        self.actionSpace = {'U': [0, -1], 'D': [0, 1],
                            'L': [-1, 0], 'R': [1, 0],
                            'UL': [-1, -1], 'UR': [1, -1],
                            'DL': [-1, 1], 'DR': [1, 1]}  # How to take each action

    def isTerminalState(self, state):
        # print(state)
        # print("Here im taking away 0: ", state[0]-0)
        if abs(state[0] - 0) <= 0.01:
            #   print(True)
            return True
        else:
            #  print(False)
            return False

    def setAgentPosition(self, action):  # Takes the new action as input
        self.agentPosition = [self.agentPosition[0] + self.actionSpace[action][0],
                              self.agentPosition[1] + self.actionSpace[action][1]]
        self.state = tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])

    def offGridMove(self, action):  # Change input if needed
        position = [self.agentPosition[0] + self.actionSpace[action][0],
                    self.agentPosition[1] + self.actionSpace[action][1]]

        return not (0 <= position[0] <= self.w and 0 <= position[1] <= self.h)

    def getQuadrant(self, newPosition):
        pos = [self.goal[0] - newPosition[0], self.goal[1] - newPosition[1]]

        if pos[0] > 0:
            if pos[1] > 0:
                return 3
            elif pos[1] < 0:
                return 0
            else:
                return 4
        elif pos[0] < 0:
            if pos[1] > 0:
                return 2
            elif pos[1] < 0:
                return 1
            else:
                return 6
        else:
            if pos[1] > 0:
                return 7
            elif pos[1] <= 0:
                return 5

    def getDist(self, position):
        dx = self.goal[0] - position[0]
        dy = self.goal[1] - position[1]

        return round(sqrt(dx ** 2 + dy ** 2), 2)

    def getState(self, newState):
        resultingPosition = [self.agentPosition[0] + self.actionSpace[newState][0],
                             self.agentPosition[1] + self.actionSpace[newState][1]]

        return tuple([self.getDist(resultingPosition), self.getQuadrant(resultingPosition)])

    def giveReward(self, newState, action):
        actionToQuadrant = {'U': 2, 'D': 6, 'L': 4, 'R': 0, \
                            'UL': 3, 'UR': 1, 'DL': 5, 'DR': 7}

        quadrant = newState[1]  # Quadrant we should go
        newQuadrant = actionToQuadrant[action]  # Quadrant we actually going

        x = 6
        reward = []
        for i in range(1, 5):
            reward.append(((quadrant + i) % 8, (quadrant + i + x) % 8))
            x -= 2

        if not self.isTerminalState(newState):
            if quadrant == newQuadrant:
                return -1  # Best Case Scenarion where it goes a straigh direction
            else:
                difference = [item for item in reward if item[0] == newQuadrant or item[1] == newQuadrant]
                ind = reward.index(difference[0]) + 2
                return -5#1 * ind
        else:
            return 0

    def step(self, action):
        resultingState = self.getState(action)

        reward = self.giveReward(resultingState, action)
        if reward == 0:
            return (0, 0), 0, True, None

        if not self.offGridMove(action):  # If not moving off grid
            self.setAgentPosition(action)
            return resultingState, reward, \
                   self.isTerminalState(resultingState), None
        else:
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), \
                   -7, \
                   self.isTerminalState([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)]), None

    def reset(self):
        self.goal = [randint(0, self.w), randint(0, self.h)]
        self.agentPosition = [randint(0, self.w), randint(0, self.h)]
        if self.getDist(self.agentPosition) > 0:
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])
        else:
            self.goal = [self.w, self.h]
            self.agentPosition = [0, 0]
            return tuple([self.getDist(self.agentPosition), self.getQuadrant(self.agentPosition)])

    def render(self, win):
        robot_size = 40

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False


        pygame.draw.rect(win, (255, 255, 255), (50, 50, 400, 400))
        pygame.draw.rect(win, (250, 0, 0), (50+self.agentPosition[0]*40, 50+self.agentPosition[1]*40, robot_size, robot_size))
        pygame.draw.rect(win, (0, 255, 0), (50+self.goal[0]*40, 50+self.goal[1]*40, robot_size, robot_size))



        pygame.display.update()
        return True


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)


def maxAction(Q, state, actions):
    # want to take agents estimate of present value of expected future rewards for state
    # and all possible actions. Also want to take the maximum of that
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
    learning_rate = 0.1  # Determines to what extent newly acquired info overrides old info.
    discount = 1.0  # Determines the importance of future rewards.
    EPS = 1.0

    Q = {}
    # Want to find a value of state action pairs
    for state in env.stateSpace:
        for action in env.possibleActions:
            Q[state, action] = 0

    for action in env.possibleActions:
        Q[(0, 0), action] = 0

    numGames = 100000
    stopLearning = numGames * 0.8  # Stop Learning after 80%
    totalRewards = np.zeros(numGames)  # Keeps track of total rewards
    rend = False

    count = 0
    flag = 0

    for i in range(numGames):
        if i % 2499 == 0:
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

        done = False
        epRewards = 0

        observation = env.reset()  # Reset environment - Returns Agent Position....
        if count >= 0:
            combi = [[(0, 0), (9, 9)], [(0, 9), (9, 0)]]
            if count > len(combi)-1:
                count = 0
                flag += 1

            env.goal = [combi[count][flag % 2][0], combi[count][flag % 2][1]]
            env.agentPosition = [combi[count][(flag+1) % 2][0], combi[count][(flag+1) % 2][1]]
            observation = tuple([env.getDist(env.agentPosition), env.getQuadrant(env.agentPosition)])
            count += 1

        while not done:
            # print(done)
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1 - EPS) \
                else env.actionSpaceSample()  # Else make a random choice

            # Get new state and reward from environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)

            # Update Q-Table with new knowledge
            Q[observation, action] = Q[observation, action] + learning_rate * (reward + \
                                                                               discount * Q[observation_, action_] - Q[
                                                                                   observation, action])
            observation = observation_  # Environment has changed state (state = new_state)

            if rend:
                pygame.time.delay(50)
                rend = env.render(win)

        if EPS - 2 / numGames > 0:
            # EPS = math.sqrt(EPS)
            EPS -= 1 / stopLearning  # 1 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
        if rend:
            print(epRewards)

    print(Q)

    with open('Qtable.pkl', 'wb') as f:
        pickle.dump([Q, totalRewards], f)

    plt.plot(totalRewards)
    plt.savefig('8Action_250k_Iterations_v1.1.png')
    plt.show()
