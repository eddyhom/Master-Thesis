import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import floor, sqrt
import pickle
import Gridworld

width = 9
height = 9
env = Gridworld.GridWorld(width, height)

f = open('Qtable.pkl', 'rb')
Q, reward = pickle.load(f)

pygame.init()
win = pygame.display.set_mode((500, 500))
pygame.display.set_caption("First Game")

for i in range(100):
    state = env.reset()
    done = False

    while not done:
        action = Gridworld.maxAction(Q, state, env.possibleActions)

        # Get new state and reward from environment
        observation_, _, done, _ = env.step(action)

        state = observation_  # Environment has changed state (state = new_state)

        pygame.time.delay(200)
        rend = env.render(win)

pygame.quit()


