# Import libraries
# now is best
import gym
import numpy as np

# Define the environment and observation space
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
observation_space = env.observation_space.n

# Define the value function
V = np.zeros(observation_space)

# Define the policy function
def policy(state):
    # Initialize all actions with equal probability
    action_prob = np.ones(env.action_space.n) / env.action_space.n
    return action_prob

# Define the value iteration function
def value_iteration(state):
    # Get the optimal value for the current state
    Q = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[state][action]:
            Q[action] += prob * (reward + (1 - done) * V[next_state])
    return np.max(Q)

# Define the number of episodes and maximum number of iterations
num_episodes = 1000
max_iter_number = 100

# Run the value iteration algorithm
for i in range(num_episodes):
    # Reset the environment
    state = env.reset()
    for j in range(max_iter_number):
        # Select an action based on the current policy
        action_prob = policy(state)
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)

        # Update the value function for the current state
        V[state] = value_iteration(state)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)

        # Update the policy and state
        state = next_state

        # If the episode has terminated, break
        if done:
            break

# Print the learned value function
print("Learned value function:\n", V)

# Close the environment
env.close()
