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
Create the FrozenLake-v1 environment with is_slippery=False, so the environment is deterministic.
Initialize a Q-table with zeros.
FrozenLake has 16 states.
There are 4 actions: Left (L), Down (D), Right (R), Up (U).
Start with epsilon = 1.0, meaning the agent initially explores randomly.
Generate an episode using the epsilon-greedy policy.
Store (state, action, reward) for every step.

Traverse the episode backwards and calculate the return:

G
t
	​

=R
t+1
	​

+γG
t+1
	​


Update the Q-value using:

Q(s,a)←Q(s,a)+α[G−Q(s,a)]
Gradually decrease epsilon from 1.0 toward 0.05, reducing exploration and increasing exploitation.

After 1500 episodes, select the action with the highest Q-value for each state:

optimal_policy = np.argmax(Q, axis=1)

Calculate the state-value function using:

V(s) = max Q(s,a)
11. Plot the moving average of rewards to observe the learning progress.


## Python Program
```
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

num_episodes = 1500
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

## Output

```text
Final Q-table:
<img width="522" height="410" alt="image" src="https://github.com/user-attachments/assets/9430b337-e2e3-4de7-9680-34cc316ca711" />




Estimated State-Value Function:
<img width="405" height="187" alt="image" src="https://github.com/user-attachments/assets/54ff9151-4343-4b44-beba-4b8ebf4efb93" />







Learned Policy:

<img width="528" height="175" alt="image" src="https://github.com/user-attachments/assets/7ab1070a-8748-4515-b741-da3e0bdc465f" />

<img width="1045" height="580" alt="image" src="https://github.com/user-attachments/assets/ed75298e-ff2e-4cd5-9093-3f361152e74f" />


## Result
```text
The final output will contain:

Final Q-table: learned Q-values for all 16 states and 4 actions.
Estimated State-Value Function: highest Q-value for each state.
Learned Policy: an action (L, D, R, or U) for each state.
Average reward over the last 1000 episodes: indicates how often the agent successfully reaches the goal.
Learning curve: shows whether the agent's performance improves over training.

Because is_slippery=False, the agent can learn a reliable path to the goal. The learned policy should generally direct the agent around the holes and toward the goal.
```
---

## Inference
```text

The Monte Carlo Control algorithm successfully learns an optimal or near-optimal policy by repeatedly generating complete episodes and updating the Q-values using the returns obtained from those episodes. Initially, the agent explores many actions because epsilon is high. As training progresses, epsilon decreases, causing the agent to increasingly choose actions with higher Q-values.

The learning curve should show an improvement in the average reward as the number of episodes increases. A higher average reward in the last 1000 episodes indicates that the agent has learned to reach the goal more consistently.

Important: the exact Q-table and average reward are dependent on the random seed, so your numerical output may differ each time you run the code. Also, because this is Monte Carlo learning with only 1500 episodes, the Q-values may not be perfectly converged.

```





---

