# Import libraries
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

# Define policy function
def policy_function(observation):
    # TODO: implement your own policy function here
    return env.action_space.sample() # example policy, replace with your own


# Create environment 
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=16), render_mode="human", is_slippery=True)

# Reset environment
observation, info = env.reset(seed=42)

max_iter_number = 1000

for _ in range(max_iter_number):

   # Call policy function to select an action 
   action = policy_function(observation)

   # Take a step in the environment based on the selected action
   observation, reward, terminated, truncated, info = env.step(action)

   # If episode is over, reset environment
   if terminated or truncated:
      observation, info = env.reset()

# Close environment
env.close()
