# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description
```
FrozenLake-v1 is a simple Reinforcement Learning environment provided by Gymnasium.

It has a 4 × 4 grid, so there are 16 states.
The agent starts from the Start (S) position.
F represents a safe frozen tile.
H represents a hole. If the agent falls into a hole, the episode ends.
G represents the goal.
The agent has 4 actions: Left, Down, Right, and Up.
The agent gets a reward of 1 when it reaches the goal.
Otherwise, the reward is 0.
In this program, is_slippery=False is used, so the agent moves in the direction it selects without slipping.

```



## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
STEP 1: Initialize the FrozenLake environment, Q-table, and parameters such as learning rate, discount factor, and epsilon.    
STEP 2: Generate a complete episode using the epsilon-greedy policy by selecting actions through exploration and exploitation.   
STEP 3: Store the states, actions, and rewards obtained during the episode.  
STEP 4: Calculate the return (G) by traversing the episode backwards using G=R+γG.   
STEP 5: Update the Q-values using Q(s,a)←Q(s,a)+α[G−Q(s,a)], and reduce epsilon after each episode.   
STEP 6: Repeat for 20,000 episodes, then select the highest Q-value action as the final policy and evaluate it using the average reward and learning curve.  


## Python Program



```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
n_states = env.observation_space.n
n_actions = env.action_space.n

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0

    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        Q[state, action] += alpha * (G - Q[state, action])

    episode_rewards.append(sum(x[2] for x in episode))

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name: Sana Fathima H")
    print("Register Number: 212223240145")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()

```



## Output


### Final Q-table:
<img width="240" height="265" alt="image" src="https://github.com/user-attachments/assets/1a5ca3fb-2a9a-438e-8200-adf7c0111762" />


#### FOR EPISODE 1500
<img width="365" height="420" alt="image" src="https://github.com/user-attachments/assets/0cad461f-4de8-4cb4-ae2c-f91939994e07" />



### Estimated State-Value Function:
<img width="390" height="110" alt="image" src="https://github.com/user-attachments/assets/485f3a29-0bb4-4313-a84f-6f70c2d24fe4" />



#### FOR EPISODE 1500

<img width="360" height="125" alt="image" src="https://github.com/user-attachments/assets/be9c374c-fa2d-4948-b050-026d6b9d2cfb" />




### Learned Policy:
<img width="189" height="78" alt="image" src="https://github.com/user-attachments/assets/ec88a584-30e8-4023-9a19-f70e5d3d0a3e" />

#### FOR EPISODE 1500
<img width="213" height="86" alt="image" src="https://github.com/user-attachments/assets/ae060fe8-239d-4f8b-ba68-5bd844bf149a" />


### Average reward over last 1000 episodes: 


<img width="632" height="421" alt="image" src="https://github.com/user-attachments/assets/b08dd966-ca58-44a8-8c36-14eb75f405af" />

#### FOR EPISODE 1500
<img width="689" height="456" alt="image" src="https://github.com/user-attachments/assets/8c7453be-ac37-419c-b986-8d0566803893" />

## Result

The Monte Carlo Control algorithm successfully learned an improved policy for the FrozenLake environment. The Q-table was updated using returns from complete episodes, and the agent learned actions that help it reach the goal while avoiding holes.


## Inference
Increasing the number of episodes gives the Monte Carlo agent more experience. Therefore, 20,000 episodes generally provide better learning than 1,500 episodes, although the exact results can vary because the algorithm uses random exploration.









