import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=16), render_mode="human", is_slippery=True)
observation, info = env.reset(seed=42)
action_space_size = env.action_space.n
stat_space_size = env.observation_space.n
qtable = np.zeros((stat_space_size, action_space_size))
num_episodes = 10000
# learning_rate = 0.2
alpha = 0.1
gamma = 0.99
epsilon = 1
# max_epsilon = 1
# min_epsilon = 0.01
# decay_rate = 0.001
max_iter_number = 1000

rewards = []
# Define MDP components
num_states = env.observation_space.n
num_actions = env.action_space.n
trans_prob = np.zeros((num_states, num_actions, num_states))
reward = np.zeros((num_states, num_actions))

# Calculate Transition Probabilities and Rewards(Value iteration)
for s in range(num_states):
    for a in range(num_actions):
        transitions = env.env.P[s][a]
        for p_trans, next_s, r, done in transitions:
            trans_prob[s, a, next_s] += p_trans
            reward[s, a] += r * p_trans

# Normalize Transition Probabilities
for s in range(num_states):
    for a in range(num_actions):
        trans_prob[s, a] /= np.sum(trans_prob[s, a])


# Define Policy Function
def policy(state, q_values, epsilon):
    if np.random.uniform() < epsilon:
        # Random Action
        return np.random.randint(num_actions)
    else:
        # Greedy Action
        return np.argmax(q_values[state])


# Define Q-Learning Algorithm(Q Function)

q_values = np.zeros((num_states, num_actions))
for i in range(num_episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = policy(state, q_values, epsilon)
        next_state, r, done1, done, info = env.step(action)
        q_values[state, action] += alpha * (r + gamma * np.max(q_values[next_state]) - q_values[state, action])
        state = next_state

# Run Trained Agent
observation, info = env.reset()
for i in range(max_iter_number):
    action = policy(observation, q_values, epsilon=0.0)
    observation, reward, done1, done, info = env.step(action)
    if done or done1:
        observation, info = env.reset()

env.close()