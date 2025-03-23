import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from environments.crypto_trading_env import CryptoTradingEnv

# Load and filter data for sideways regimes
df = pd.read_csv("data/processed/BTCUSDT_H1_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
sideways_df = df[df["Market_Regime"] == "Sideways"].reset_index(drop=True)
train_df = sideways_df.iloc[:int(0.8 * len(sideways_df))].reset_index(drop=True)

# Initialize environment
env = CryptoTradingEnv(train_df)

# A2C model with tuned hyperparameters
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,  # Lowered for stability
    gamma=0.99,
    n_steps=5,  # Frequent updates for A2C
    tensorboard_log="./a2c_logs/"
)

# Validation environment
eval_env = CryptoTradingEnv(train_df.iloc[-1000:].reset_index(drop=True))
eval_callback = EvalCallback(
    eval_env, eval_freq=10000, best_model_save_path="./models/A2C/best", verbose=1
)

# Train
model.learn(total_timesteps=200_000, log_interval=10, callback=eval_callback)

# Save model
os.makedirs("models/A2C", exist_ok=True)
model.save("models/A2C/a2c_crypto_trader")

print("✅ A2C Training Complete! Model Saved.")
