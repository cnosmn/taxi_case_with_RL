# Taxi Q-Learning Project

This project implements a Q-learning algorithm to train a taxi agent to pick up and drop off passengers in a grid-world environment, demonstrating the power of reinforcement learning in solving sequential decision-making problems.

## Project Description

This application uses a customized version of the classic OpenAI Gym Taxi-v3 environment. In a 10x10 grid world, we train a taxi agent to learn how to efficiently navigate, pick up passengers from specific locations, and deliver them to their destinations while avoiding obstacles.

The project demonstrates fundamental reinforcement learning concepts including:
- State representation and encoding
- Action selection through epsilon-greedy policy
- Q-value updates using the Bellman equation
- Exploration vs. exploitation trade-offs
- Reward shaping and optimization techniques

The environment features customizable obstacles, random starting positions, passenger and destination locations, providing a challenging learning task for the agent. Through Q-learning, the agent gradually learns the optimal policy for navigating the environment and completing the taxi service tasks.

## Files

- `custom_taxi_env.py`: Custom taxi environment implementation
- `q_learning.py`: Main Q-learning algorithm implementation
- `interactive_test_agent.py`: Interactive script for step-by-step testing of the trained model
- `debug_q_learning.py`: Diagnostic tool for troubleshooting environment and model issues
- `test_env.py`: Test the enviroment

## Requirements

```
numpy
matplotlib
gym
```

## Installation

```bash
# Install required libraries
pip install numpy matplotlib gym

# Clone the repository
git clone https://github.com/username/taxi-q-learning.git
cd taxi-q-learning
```

## Usage

### 1. Training the Model

Run the basic Q-learning algorithm:

```bash
python q_learning.py
```

### 2. Testing the Trained Model

Test the trained model interactively:

```bash
python interactive_test_agent.py
```

### 3. Debugging

Diagnose environment and model issues:

```bash
python debug_q_learning.py
```

## Environment Details

### State Space
- Taxi position (row, column)
- Passenger position (row, column)
- Destination position (row, column)
- Whether passenger is in taxi (boolean)

### Action Space
- 0: South (down)
- 1: North (up)
- 2: East (right)
- 3: West (left)
- 4: Pickup passenger
- 5: Dropoff passenger

### Reward Structure
- Each step: -1 point
- Successful pickup: +10 points
- Failed pickup: -10 points
- Successful dropoff (at destination): +20 points
- Failed dropoff: -10 points

## Learning Parameters

- alpha (learning rate): 0.1
- gamma (discount factor): 0.9
- epsilon (exploration rate): 0.1 (basic) / 4.0→0.01 (optimized)

## Algorithm

Q-learning is a model-free reinforcement learning algorithm where Q-values are updated using the Bellman equation:

```
Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max(Q(s')))
```

Where:
- s: current state
- a: selected action
- s': next state
- r: received reward
- α: learning rate
- γ: discount factor

## Development Notes

- Due to the large state space (2 million states), the optimized Q-learning version works more effectively
- Using relative positions to reduce the state space accelerates training
- Epsilon decay strategy improves the exploration/exploitation balance

## Future Improvements

- Deep Q-Network (DQN) implementation
- More complex environments and tasks
- Multi-agent support
- Training acceleration through transfer learning methods

## Contact

[cnosmn14043@gmail.com]
