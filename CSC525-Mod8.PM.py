import os
import sys
import subprocess
import gc
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 1. AUTO-DEPENDENCY INSTALLER
def install_requirements():
    requirements = ["tensorboard", "kaggle-environments"]
    for package in requirements:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

from torch.utils.tensorboard import SummaryWriter
from kaggle_environments import make

# 2. HARDWARE & CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('logs/gen3_marathon')
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')

CONFIG = {
    "LR_GEN3": 0.00005,
    "GAMMA": 0.99,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 100000,         # Expanded for GPU
    "EPSILON_START": 0.20,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY_STEPS": 200000, # Slower decay for higher episode count
    "TARGET_UPDATE_FREQ": 1000,
    "SAVE_EVERY": 1000,
    "NUM_EPISODES": 10000,
    "SHAPED_REWARD_WEIGHTS": {"2_ROW": 0.005, "3_ROW": 0.02}
}

# 3. HELPER FUNCTIONS
def get_epsilon(step):
    if step >= CONFIG["EPSILON_DECAY_STEPS"]: return CONFIG["EPSILON_END"]
    return CONFIG["EPSILON_START"] + (step / CONFIG["EPSILON_DECAY_STEPS"]) * (CONFIG["EPSILON_END"] - CONFIG["EPSILON_START"])

def preprocess_observation(observation):
    board = np.array(observation.board if hasattr(observation, 'board') else observation['board']).reshape(6, 7)
    mark = observation.mark if hasattr(observation, 'mark') else observation['mark']
    ch1, ch2, ch3 = (board == mark), (board == (3 - mark)), (board == 0)
    state = np.stack((ch1, ch2, ch3), axis=0).astype(np.float32)
    return torch.tensor(state).to(device)

def count_n_in_a_row(board_2d, player, n):
    rows, cols = board_2d.shape
    count = 0
    for r in range(rows):
        for c in range(cols - n + 1):
            if all(board_2d[r, c+i] == player for i in range(n)): count += 1
    for r in range(rows - n + 1):
        for c in range(cols):
            if all(board_2d[r+i, c] == player for i in range(n)): count += 1
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(board_2d[r+i, c+i] == player for i in range(n)): count += 1
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(board_2d[r-i, c+i] == player for i in range(n)): count += 1
    return count

def compute_shaped_reward(old_board, new_board):
    old_2d, new_2d = np.array(old_board).reshape(6, 7), np.array(new_board).reshape(6, 7)
    d3 = count_n_in_a_row(new_2d, 1, 3) - count_n_in_a_row(old_2d, 1, 3)
    d2 = count_n_in_a_row(new_2d, 1, 2) - count_n_in_a_row(old_2d, 1, 2)
    return (d3 * CONFIG["SHAPED_REWARD_WEIGHTS"]["3_ROW"] + d2 * CONFIG["SHAPED_REWARD_WEIGHTS"]["2_ROW"])

# 4. MODELS
class ConvDQN(nn.Module):
    def __init__(self):
        super(ConvDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(64*6*7, 128), nn.ReLU(), nn.Linear(128, 7))
    def forward(self, x): return self.fc(self.conv(x))

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d): self.buffer.append((s.cpu(), a, r, ns.cpu(), d))
    def sample(self, b):
        samples = random.sample(self.buffer, b)
        s, a, r, ns, d = zip(*samples)
        return torch.stack(s).to(device), torch.tensor(a).to(device), torch.tensor(r, dtype=torch.float32).to(device), torch.stack(ns).to(device), torch.tensor(d, dtype=torch.float32).to(device)
    def __len__(self): return len(self.buffer)

# 5. MAIN TRAINING LOGIC
def main():
    print(f"Starting Gen 3 on {device}")
    policy_net = ConvDQN().to(device)
    target_net = ConvDQN().to(device)
    replay_buffer = ReplayBuffer(CONFIG["BUFFER_SIZE"])
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG["LR_GEN3"])

    # Load checkpoint logic
    start_ep, g_step = 10001, 0
    ckpt_path = "connect4_ep10000.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        policy_net.load_state_dict(ckpt['model_state_dict'])
        target_net.load_state_dict(ckpt['model_state_dict'])
        print(f"Success: Resuming from {ckpt_path}")

    trainer = make("connectx", debug=False).train([None, "negamax"])
    total_rewards = deque(maxlen=100)
    
    for ep in range(start_ep, start_ep + CONFIG["NUM_EPISODES"]):
        obs = trainer.reset()
        if hasattr(obs, 'observation'): obs = obs.observation
        state = preprocess_observation(obs)
        done, ep_rew = False, 0

        while not done:
            eps = get_epsilon(g_step)
            valid = [c for c in range(7) if obs.board[c] == 0]
            if random.random() < eps:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q = policy_net(state.unsqueeze(0))
                    mask = torch.full((7,), -1e9).to(device)
                    mask[valid] = 0
                    action = (q + mask).argmax().item()

            old_board = obs.board[:]
            next_obs, reward, done, _ = trainer.step(action)
            next_state = preprocess_observation(next_obs)
            
            shaped_rew = reward + compute_shaped_reward(old_board, next_obs.board)
            replay_buffer.push(state, action, shaped_rew, next_state, done)
            
            state, obs, ep_rew, g_step = next_state, next_obs, ep_rew + reward, g_step + 1

            if len(replay_buffer) >= CONFIG["BATCH_SIZE"]:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(CONFIG["BATCH_SIZE"])
                curr_q = policy_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = r_b + CONFIG["GAMMA"] * target_net(ns_b).max(1)[0] * (1 - d_b)
                
                loss = loss_fn(curr_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if g_step % 100 == 0: writer.add_scalar('Loss/train', loss.item(), g_step)

            if g_step % CONFIG["TARGET_UPDATE_FREQ"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        total_rewards.append(ep_rew)
        writer.add_scalar('Reward/Episode', ep_rew, ep)
        
        if ep % 100 == 0:
            avg_rew = np.mean(total_rewards)
            print(f"Ep {ep} | AvgRew {avg_rew:.2f} | Eps {eps:.2f} | Step {g_step}")

        if ep % CONFIG["SAVE_EVERY"] == 0:
            save_path = f"checkpoints/gen3_ep{ep}.pth"
            torch.save({'model_state_dict': policy_net.state_dict(), 'episode': ep, 'global_step': g_step}, save_path)
            print(f"Checkpoint saved: {save_path}")
            torch.cuda.empty_cache() # Keeps GPU memory clean

    writer.close()
    print("Gen 3 Marathon Complete.")

if __name__ == "__main__":
    main()