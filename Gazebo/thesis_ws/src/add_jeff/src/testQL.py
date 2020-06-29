#!/usr/bin/env python
import pygame
import pickle
import Gridworld #We use the training code for the class GridWorld

width = 9 #Determine width of space (same as for training)
height = 9  #Determine height of space (same as for training)
env = Gridworld.GridWorld(width, height) #Create a Gridworld object to use its functions..

f = open('Qtable.pkl', 'rb') #Open Qtable file to use what was learned
Q, reward = pickle.load(f) #Set Qtable to Q, ignore reward its not needed here

pygame.init() #Init for visualizatiion
win = pygame.display.set_mode((500, 500)) #Create window to draw on
pygame.display.set_caption("First Game") #Name of window

for i in range(100): #Try 100 random options to test QLearning
    state = env.reset() #Reset environment
    done = False #Determines if robot has reached goal

    while not done: #Go until goal has been reached
        action = Gridworld.maxAction(Q, state, env.possibleActions) #Take best action - previously learned in the training

        # Get new state and reward from environment
        observation_, _, done, _ = env.step(action)

        state = observation_  # Environment has changed state (state = new_state)

        pygame.time.delay(200) # Draw whats happening, higher delay to better observe what's happening
        rend = env.render(win) # Render whats going on.

pygame.quit() #Close window


