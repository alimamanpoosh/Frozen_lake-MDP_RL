import random

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Create Environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=16), render_mode="human", is_slippery=True)
observation, info = env.reset(seed=42)
action_space_size = env.action_space.n
stat_space_size = env.observation_space.n
qtable = np.zeros((stat_space_size, action_space_size))

num_episodes = 1000
learning_rate = 0.2
alpha = 0.1
gamma = 0.99
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001
max_iter_number = 1000

rewards = []

for episode in range(num_episodes):
    print(episode)
    state = env.reset()[0]
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_iter_number):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done1, done, info = env.step(action)
        max_new_state = np.max(qtable[new_state, :])
        qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + gamma * max_new_state - qtable[state, action])
        total_rewards += reward
        state = new_state
        if done or done1:
            break
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("score:", str(sum(rewards) / total_rewards))


env.reset()

for episode in range(1):
    state = env.reset()
    step = 0
    done = False
    print("Episode:", episode + 1)

    for step in range(max_iter_number):
        action = np.argmax(qtable[state, :])
        new_state, reward, done1, done, info = env.step(action)
        env.render()
        if done:
            print("number of steps", step)
            break
        state = new_state
env.close()
