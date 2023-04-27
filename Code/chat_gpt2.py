import gym
import numpy as np

env = gym.make("FrozenLake-v1")

# Value Iteration function
def value_iteration(env, gamma=1.0, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            q_vals = []
            for a in range(env.nA):
                q_val = sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.P[s][a]])
                q_vals.append(q_val)
            V[s] = max(q_vals)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Policy function
def policy(state, V, gamma=1.0):
    actions = np.zeros(env.nA)
    for a in range(env.nA):
        q_val = sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.P[state][a]])
        actions[a] = q_val
    action = np.argmax(actions)
    return action

# Run Value Iteration
V = value_iteration(env)

# Test Policy function
state = env.reset()
while True:
    action = policy(state, V)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
