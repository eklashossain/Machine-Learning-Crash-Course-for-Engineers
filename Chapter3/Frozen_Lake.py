# ----------------------Importing Libraries----------------------
import numpy as np
import gymnasium as gym
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections.abc import Sequence


# ----------------------Initiate Environment---------------------
custom_map = ["SFFH",
              "FFHF",
              "HFFF",
              "HFFG"]

env_gym = gym.make('FrozenLake-v1', desc=custom_map, render_mode="rgb_array")

env_gym.reset()
plt.imshow(env_gym.render())


# --------------------------Parameters---------------------------
# Generating Q-table
a_size = env_gym.action_space.n
s_size = env_gym.observation_space.n
# Initializing Q-table with zero
Q_table = np.zeros((s_size, a_size))

# Total number of episodes
num_episodes = 9000
# Maximum number of steps agent is allowed to take within a single episode
maximum_step_each_episode = 90

learn_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

def check_state(state):
    if isinstance(state, Sequence):
        return state[0]
    else:
        return state


# ---------------------------Training----------------------------
# List to hold reward from each episodes
all_rewards = []

# Q-learning algorithm
for episode in range(num_episodes):
    # Initialize new episode params
    state = env_gym.reset()
    state = check_state(state)
    # Done parameter keep to keep track of episode when finish
    finished_flag = False
    rewards_current_episode = 0

    # For each time step within this episode
    for step in range(maximum_step_each_episode):

        # Choosing between exploration and exploitation
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(Q_table[state, :])
        else:
            action = env_gym.action_space.sample()

        outs = env_gym.step(action)
        if len(outs) == 4:
            observation, reward, finished_flag, _ = env_gym.step(action)
        else:
            observation, reward, terminated, truncated, _ = env_gym.step(action)
            finished_flag = terminated

        # Update Q-table for Q(s,a)
        Q_table[state, action] = Q_table[state, action] * (1 - learn_rate) + learn_rate * (
                    reward + discount_rate * np.max(Q_table[observation, :]))

        state = observation
        rewards_current_episode += reward

        if finished_flag == True:
            break

    # Exploration rate decay using exponential decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
        -exploration_decay_rate * episode)

    all_rewards.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(all_rewards), num_episodes / 1000)
count = 1000

print("Average reward summary:")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Updated Q-table
print("Updated Q-table:")
print(Q_table)



# -----------------Iterations of agent playing-------------------
# Watching the agent playing with best actions from the Q-table
for episode in range(4):
    state = env_gym.reset()
    state = check_state(state)
    finished_flag = False
    print("============================================")
    print("EPISODE: ", episode+1)

    for step in range(maximum_step_each_episode):

        action = np.argmax(Q_table[state, :])
        outs = env_gym.step(action)

        if len(outs) == 4:
            observation, reward, finished_flag, _ = env_gym.step(action)
        else:
            observation, reward, terminated, truncated, _ = env_gym.step(action)
            finished_flag = terminated

        if finished_flag:
            plt.imshow(env_gym.render())
            if reward == 1:
                print("The agent reached the Goal")
            else:
                print("The agent fell into a hole")

            print("Number of steps", step)

            break

        state = observation

env_gym.close()