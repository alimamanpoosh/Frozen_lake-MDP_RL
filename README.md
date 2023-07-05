# AI_P1 (Frozen Lake - MDP - SARSA(RL))

## MDP : Value Policy Iteration

This project implements the value iteration algorithm for solving the FrozenLake-v1 environment in OpenAI Gym. The algorithm iteratively updates the value function and finds the optimal policy for the given environment.

### Installation

To run this project, you need to have Python and the gymnasium library installed. You can install the necessary dependencies using pip:

```shell
pip install gymnasium
```

### Usage

To run the value policy iteration algorithm and print the resulting value function, execute the following command:

```shell
python main.py
```

You can modify the maximum number of iterations and other parameters in the `Value_policy_iteration` function inside `main.py`.
## SARSA: FrozenLake-Reinforcement Learning

This part implements the SARSA algorithm to train an agent in the FrozenLake-v1 environment from the OpenAI Gym. The agent aims to find the optimal policy for navigating a frozen lake, where it must reach the goal while avoiding holes. The Q-learning algorithm is used to learn the Q-values for each state-action pair and guide the agent's decision-making process during training.

Feel free to use, modify, and distribute this code for your own purposes.
