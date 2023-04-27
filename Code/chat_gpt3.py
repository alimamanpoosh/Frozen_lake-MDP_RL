import gym
import numpy as np

env = gym.make('FrozenLake-v1')

# Define the policy function
def policy(state, action):
    return 1/3

# Define the value iteration algorithm
def value_iteration(env, gamma=1.0, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            q_vals = []
            for a in range(env.action_space.n):
                q_val = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q_val += prob * (reward + gamma * V[next_state])
                q_vals.append(q_val)
            V[s] = max(q_vals)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Test the policy and value iteration algorithm on the FrozenLake environment
values = value_iteration(env)
policy_values = np.zeros((env.observation_space.n, env.action_space.n))
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q_val = 0
        for prob, next_state, reward, done in env.P[s][a]:
            q_val += prob * (reward + values[next_state])
        policy_values[s][a] = q_val
optimal_policy = np.argmax(policy_values, axis=1)

# Print the optimal policy for each state
print(optimal_policy)
