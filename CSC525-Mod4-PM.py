# test_connect4.py

from kaggle_environments import make

# Step 1: Create the Connect Four environment
env = make("connectx", configuration={"rows": 6, "columns": 7, "inarow": 4})

# Step 2: Define a simple test agent
def simple_agent(obs, config):
    # Select the first available column
    for c in range(config.columns):
        if obs.board[c] == 0:
            return c

# Step 3: Run a single game against a random opponent
result = env.run([simple_agent, "random"])

# Step 4: Print the game result and final board
print("Game result:", result)
print("Final board state:", env.state[0].observation.board)

# Step 5: Render the game (works best in Jupyter or Colab)
try:
    env.render(mode="ipython")
except Exception:
    print("HTML rendering is only available in notebooks.")
