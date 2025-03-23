import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from environments.crypto_trading_env_ddpg import CryptoTradingEnvDDPG

# Load and filter data for bearish regimes
df = pd.read_csv("data/processed/BTCUSDT_H1_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
bearish_df = df[df["Market_Regime"] == "Bearish"].reset_index(drop=True)
train_df = bearish_df.iloc[:int(0.8 * len(bearish_df))].reset_index(drop=True)

# Initialize environment
env = CryptoTradingEnvDDPG(train_df)

# Action noise
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

# Load or create model
model = DDPG.load("models/DDPG/ddpg_crypto_trader_finetuned", env=env) if os.path.exists("models/DDPG/ddpg_crypto_trader_finetuned.zip") else DDPG(
    "MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=0.00005,
    gamma=0.99, buffer_size=100_000, learning_starts=1000, batch_size=128, tensorboard_log="./ddpg_logs/"
)

# Validation environment
eval_env = CryptoTradingEnvDDPG(bearish_df.iloc[int(0.8 * len(bearish_df)):].reset_index(drop=True))
eval_callback = EvalCallback(eval_env, eval_freq=10000, best_model_save_path="./models/DDPG/best", verbose=1)

# Train
model.learn(total_timesteps=50_000, log_interval=10, callback=eval_callback)

# Save
os.makedirs("models/DDPG", exist_ok=True)
model.save("models/DDPG/ddpg_crypto_trader_final")

print("✅ DDPG Final Training Complete! Model Saved.")
