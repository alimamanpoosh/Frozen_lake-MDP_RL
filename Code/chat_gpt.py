import gymnasium as gym
import numpy as np

# create the Frozen Lake environment
env = gym.make('FrozenLake-v1')

# get the number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

# initialize the value function and discount factor
V = np.zeros(num_states)
gamma = 0.99

def value_iteration(max_iterations=10000):
    """
    The value iteration algorithm.
    """
    delta = np.inf
    i = 0
    
    # iterate until convergence or maximum iterations reached
    while delta > 1e-8 and i < max_iterations:
        delta = 0
        
        # update the value function for each state
        for s in range(num_states):
            v = V[s]
            q_values = []
            
            # calculate the Q-value for each action in the state
            for a in range(num_actions):
                next_states_rewards = []
                
                # calculate the expected reward for each possible next state
                for p, s_, r, done in env.P[s][a]:
                    next_states_rewards.append(p * (r + gamma * V[s_]))
                
                # calculate the Q-value for the action
                q_values.append(sum(next_states_rewards))
            
            # update the value function for the state
            V[s] = max(q_values)
            
            # store the maximum change in the value function
            delta = max(delta, abs(v - V[s]))
        
        # increment iteration count
        i += 1
    
    # print optimal policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            next_states_rewards = []
            for p, s_, r, done in env.P[s][a]:
                next_states_rewards.append(p * (r + gamma * V[s_]))
            q_values.append(sum(next_states_rewards))
        policy[s] = np.argmax(q_values)
    
    print("Optimal Policy:")
    print(policy.reshape(4,4))

# run the algorithm and print the resulting value function
value_iteration()
print("Value function:")
print(V.reshape(4,4))
