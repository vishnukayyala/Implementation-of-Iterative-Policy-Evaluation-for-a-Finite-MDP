
# Implementation of Iterative Policy Evaluation for a Finite Markov Decision Process

## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function V(s) for a fixed random policy in a finite Markov Decision Process (MDP).

## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

**Requirements:**
- Python 3.x
- Gymnasium
- NumPy
- Jupyter Notebook or any Python IDE

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where an agent starts from an initial position and tries to reach a goal while avoiding holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Action 0 | Left |
| Action 1 | Down |
| Action 2 | Right |
| Action 3 | Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and holes |

**Environment layout:**

```text
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

Where:
- S = Starting position
- F = Frozen surface
- H = Hole
- G = Goal

## Problem Statement

The objective is to evaluate a fixed random policy in the FrozenLake-v1 environment using iterative policy evaluation.

The agent follows a random policy in which all four actions have an equal probability of being selected.

The policy is represented as:

```text
P(action | state) = 1/4
```

The goal is to estimate the state-value function V(s).

It represents the expected total discounted reward obtained by starting from a particular state and following the given policy.

The environment's transition probabilities are used to calculate the expected value of each state. The process is repeated until the value estimates converge.

## Theory

A **Markov Decision Process (MDP)** is a mathematical framework used to represent decision-making problems where an agent interacts with an environment.

An MDP consists of:

- **States (S):** All possible positions or situations of the agent.
- **Actions (A):** Available actions that the agent can perform.
- **Transition Probability (P):** Probability of moving from one state to another after taking an action.
- **Reward (R):** Feedback received after performing an action.
- **Discount Factor (gamma):** Determines the importance of future rewards.

### State-Value Function

The state-value function V(s) represents the expected return obtained from state s by following a particular policy.

### Bellman Expectation Equation

Iterative policy evaluation uses the Bellman expectation equation to calculate the value of each state.

```text
V(s) = SUM over actions [ P(a|s) *
       SUM over next states [ P(s'|s,a) *
       (R(s,a,s') + gamma * V(s')) ] ]
```

Where:

| Symbol | Meaning |
|---|---|
| s | Current state |
| a | Selected action |
| s' | Next state |
| P(a\|s) | Probability of selecting an action |
| P(s'\|s,a) | Transition probability |
| R(s,a,s') | Immediate reward |
| gamma | Discount factor |
| V(s) | Value of the current state |

### Iterative Policy Evaluation

The algorithm starts by assigning zero to the value of every state. It then repeatedly updates each state's value using the Bellman expectation equation.

The process continues until the maximum difference between two consecutive value estimates becomes smaller than a predefined threshold.

```text
Delta = MAX over all states |New V(s) - Old V(s)|
```

When:

```text
Delta < theta
```

the algorithm stops because the value function has converged within the specified tolerance.

## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the environment's state transition probabilities.
3. Initialize the value function V(s) = 0 for all states.
4. Define a random policy where each of the four actions has an equal probability of 1/4.
5. Set the discount factor gamma and convergence threshold theta.
6. For every state:
   - Consider all four possible actions.
   - Obtain the transition probabilities, next states, rewards, and terminal information.
   - Calculate the expected value using the Bellman expectation equation.
7. Update the value function for all states.
8. Calculate the maximum change between consecutive iterations.
9. Repeat until the maximum change is smaller than the convergence threshold.
10. Display the number of iterations and the final state-value function as a 4 x 4 grid.

## Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P

# Parameters
gamma = 0.9
theta = 1e-8
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize value function
V = np.zeros(n_states)

# Random policy: equal probability for all actions
policy = np.ones((n_states, n_actions)) / n_actions

# Identify terminal states
desc = env.unwrapped.desc.flatten()
terminal = [
    s for s in range(n_states)
    if desc[s] in (b'H', b'G')
]

# Iterative policy evaluation
iterations = 0

while True:
    delta = 0
    new_V = V.copy()

    for s in range(n_states):
        if s in terminal:
            new_V[s] = 0
            continue

        value = 0

        for a in range(n_actions):
            for prob, next_s, reward, done in P[s][a]:
                next_value = 0 if done else V[next_s]

                value += policy[s, a] * prob * (
                    reward + gamma * next_value
                )

        new_V[s] = value
        delta = max(delta, abs(new_V[s] - V[s]))

    V = new_V
    iterations += 1

    if delta < theta:
        break

# Display results
print("Number of Iterations:", iterations)
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

## Output

```text
Number of Iterations: 49

State-Value Function as 4x4 Grid:

[[0.0045 0.0042 0.0101 0.0041]
 [0.0067 0.0000 0.0263 0.0000]
 [0.0187 0.0576 0.1070 0.0000]
 [0.0000 0.1304 0.3915 0.0000]]
```

The output displays the number of iterations required for convergence and the estimated state-value function for all 16 states.

## Result

Iterative policy evaluation was successfully implemented using the Gymnasium FrozenLake environment.

The state-value function for a fixed random policy was estimated using the Bellman expectation equation. The algorithm repeatedly updated the values until the convergence condition was satisfied.

## Inference

- Iterative policy evaluation estimates the value of each state under a fixed policy.
- The random policy assigns an equal probability of 1/4 to each available action.
- The Bellman expectation equation is used to update state values.
- The discount factor determines the importance of future rewards.
- The convergence threshold controls the accuracy of the estimated values.
- The algorithm stops when consecutive value estimates differ by less than the specified threshold.

## About

Implementation of Iterative Policy Evaluation for a Finite Markov Decision Process using Python, NumPy, and Gymnasium.
