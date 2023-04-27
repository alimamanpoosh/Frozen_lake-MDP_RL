#import library

import gymnasium as gym

from gym.envs.toy_text.frozen_lake import generate_random_map

# TODO
def policy_evaluation(s, a):
    pass

#TODO
def value_iteration(s):
   pass


env = gym.make("FrozenLake-v1", desc=generate_random_map(size=16), render_mode="human", is_slippery=True)

observation, info = env.reset(seed=42)

max_iter_number = 1000

for _ in range(max_iter_number):

   ##################################

   # # TODO # #

   # The action selection (policy) function should appear here

   # which returns an action

   # .sample() returns a random action!

   # Replace this line with a call to your implemented function

   action = env.action_space.sample()

   ##################################

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:

      observation, info = env.reset()

env.close()

