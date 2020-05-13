import gym
import numpy as np
import random
import time
#from IPython.display import clear_output

#Create out environment
env = gym.make('FrozenLake-v0')

#Number of colums in table = action space
action_space_size = env.action_space.n
#Number of rows in table = state space in environment
state_space_size = env.observation_space.n

#Initialize table with all zeros
Q_table = np.zeros((state_space_size, action_space_size))
#print(Q_table)

#Set learning parameters
learning_rate = 0.1 #Determines to what extent newly acquired info overrides old info.
discount = .99 #Determines the importance of future rewards.
num_episodes = 10000 #Number of episodes agent will play during training
max_steps_per_episode = 100

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

#create list to contain total rewards and steps per episode
rewardList = []

#Q-learning algorithm
#First loop contains everything happning in each episode.
for episode in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    totalReward = 0
    done = False

    #Second nested loop contains everything that happens for
    #a single timestep within each episode.
    for step in range(max_steps_per_episode):

        #This is used to determine wether the agent will explore or exploit in this timestep.
        exploration_rate_threshhold = random.uniform(0,1)
        if exploration_rate_threshhold > exploration_rate:
            #Agent will exploit env by choosing highest q-value for current state
            action = np.argmax(Q_table[state,:])
        else:
            action = env.action_space.sample() #Explore env and sample action randomly.

        #Choose an action by picking from Q table
        #a = np.argmax(Q_table[state,:] + np.random.randn(1,action_space_size)*(1./(i+1)))

        #Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        #Update Q-Table with new knowledge
        Q_table[state,action] = Q_table[state,action] + learning_rate*(reward + discount*np.max(Q_table[new_state,:]) - Q_table[state,action])
        state = new_state
        totalReward += reward
        if done == True:
            break

    #Exoploration rate decay
    exploration_rate = min_exploration_rate + \
                (max_exploration_rate-min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewardList.append(totalReward)

print("Score over time: " +  str(sum(rewardList)/num_episodes))
#Calculate and print the average reward per 1000 episodes.
reward_per_thousand = np.split(np.array(rewardList), num_episodes/1000)
count = 1000
print("***Average reward per 1000 episodes***")
for r in reward_per_thousand:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

print("\n***Final Q-Table Values***\n")
print(Q_table)



#Agent in live action using the learned knowledge

#For each of the 5 episodes that will be shown
for episode in range(5):
    state = env.reset()
    done = False #Just keeps tracks if episode is done
    print("***Episode ", episode+1,"***\n\n\n")
    time.sleep(1) #Time to read printout

    for step in range(max_steps_per_episode):
        #clear_output(wait=True) #Clears output from current cell. Waits until another printout is available.
        env.render() #Renders current state to the display and shows gamegrid
        time.sleep(0.5)

        action = np.argmax(Q_table[state,:]) #Sets action to highest q-value from table for current state
        new_state, reward, done,_ = env.step(action)

        #If current action did end episode
        if done:
           # clear_output(wait=True)
            env.render() #Renders to show where agent ended up from last timestep
            if reward == 1: 
                print("***You've reached the goal***")
                time.sleep(0.3)
            else:
                print("***You fell through a hole***")
                time.sleep(0.3)
             #   clear_output(wait=True)
                break

        #If last action didn't complete episode, transition to new state
        state = new_state

#When all episodes are done
env.close()