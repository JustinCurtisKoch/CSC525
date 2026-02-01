# CSC525 Principles of Machine Learning
# Portfolio Project: ML Agents Project


# Connect Four Reinforcement Learning Project
# 
# Training a reinforcement learning agent to play Connect Four using a Kaggle-provided environment.
# 
# The implementation uses:
# - PyTorch for model training
# - TensorBoard for logging and visualization
# 
# Each section is intentionally separated so it can be copied between
# Google Colab cells and a Git repository as the project evolves.

# =========================
# Colab Setup: Install Dependencies
# =========================

# !pip install -q kaggle-environments
# !pip install -q gym==0.25.2

"""
Google Colab includes multiple reinforcement learning libraries with conflicting dependency requirements.
For this project, the Kaggle Connect Four environment was verified to function correctly using Gym 0.25.2, and unused libraries were ignored.
This ensured a stable runtime without impacting training or evaluation.
"""

# =========================
# Imports and Dependencies
# =========================

import numpy as np
import random
from collections import deque

# Kaggle Connect Four environment
from kaggle_environments import make

# PyTorch (to be used for DQN)
import torch
import torch.nn as nn
import torch.optim as optim

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# ==============================
# Cell 4: Inspect Connect Four Environment
# ==============================

"""
Create and return the Kaggle Connect Four environment.
The environment manages the game rules, valid moves,
and win/loss conditions.
"""

from kaggle_environments import make
import numpy as np

# Create the Connect Four environment
env = make("connectx", debug=True)

# Reset environment to get initial observations
env.reset()

# Extract initial observation for player 0
initial_obs = env.state[0].observation

print("=== Environment Overview ===")
print("Environment name:", env.name)
print("Number of agents:", len(env.state))
print()

print("=== Observation Keys ===")
print(initial_obs.keys())
print()

print("=== Board Information ===")
print("Board shape (flattened):", len(initial_obs.board))
print("Board contents:", initial_obs.board)
print()

print("=== Game Configuration ===")
print("Rows:", env.configuration.rows)
print("Columns:", env.configuration.columns)
print("In-a-row to win:", env.configuration.inarow)
print()

print("=== Action Space ===")
print("Valid actions (columns):", list(range(env.configuration.columns)))
print()

print("=== Sample Board as 2D Grid ===")
board_2d = np.array(initial_obs.board).reshape(
    env.configuration.rows, env.configuration.columns
)
print(board_2d)

# ==============================
# State Preprocessing
# ==============================

import torch

def preprocess_observation(observation):
    """
    Converts a Kaggle Connect Four observation into a PyTorch tensor.
    The agent's own pieces are encoded as 1,
    the opponent's pieces as -1,
    and empty spaces as 0.
    """
    board = observation.board
    mark = observation.mark  # 1 or 2

    processed_board = []
    for cell in board:
        if cell == 0:
            processed_board.append(0)
        elif cell == mark:
            processed_board.append(1)
        else:
            processed_board.append(-1)

    return torch.tensor(processed_board, dtype=torch.float32)

# Test preprocessing on initial observation
test_tensor = preprocess_observation(env.state[0].observation)
print("Processed state tensor shape:", test_tensor.shape)
print("Processed state tensor:", test_tensor)

# ==============================
# Deep Q-Network (DQN) Model
# ==============================

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size=42, output_size=7):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Instantiate model and test forward pass
model = DQN()

sample_output = model(test_tensor)
print("Q-values output shape:", sample_output.shape)
print("Q-values:", sample_output)

# ===============================
# Epsilon-Greedy Action Selection
# ===============================

import random
import torch

"""
Returns a list of valid column indices.
A column is valid if the top cell is empty (0).
"""

def get_valid_actions(board, columns=7):
    valid_actions = []
    for col in range(columns):
        if board[col] == 0:
            valid_actions.append(col)
    return valid_actions


def select_action(model, state_tensor, board, epsilon):
    """
    Selects an action using epsilon-greedy strategy.

    Args:
        model: Q-network
        state_tensor: Torch tensor of shape [42]
        board: Raw board list from Kaggle environment
        epsilon: Probability of random exploration

    Returns:
        action (int): selected column
    """
    valid_actions = get_valid_actions(board)

    # Exploration
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # Exploitation
    with torch.no_grad():
        q_values = model(state_tensor)

    # Mask invalid actions
    masked_q_values = q_values.clone()
    for action in range(len(q_values)):
        if action not in valid_actions:
            masked_q_values[action] = -float("inf")

    return torch.argmax(masked_q_values).item()