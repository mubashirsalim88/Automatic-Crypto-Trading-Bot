import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from environments.crypto_trading_env_ddpg import CryptoTradingEnvDDPG

# Load and filter for bearish test data
df = pd.read_csv("data/processed/BTCUSDT_H1_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
bearish_df = df[df["Market_Regime"] == "Bearish"].reset_index(drop=True)
test_df = bearish_df.iloc[int(0.8 * len(bearish_df)):].reset_index(drop=True)

# Initialize environment
env = CryptoTradingEnvDDPG(test_df)
obs, _ = env.reset()

# Load model
model = DDPG.load("models/DDPG/ddpg_crypto_trader_final")

# Test loop
done = False
total_reward = 0
step_count = 0
starting_net_worth = env.net_worth
net_worth_history = [starting_net_worth]
actions = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    actions.append(action[0])
    print(f"Step {step_count}: Action Taken: {action}")
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    step_count += 1
    net_worth_history.append(env.net_worth)
    env.render()

# Metrics
final_net_worth = env.net_worth
profit = final_net_worth - starting_net_worth
roi = (profit / starting_net_worth) * 100
returns = np.diff(net_worth_history) / net_worth_history[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24)
peak = np.maximum.accumulate(net_worth_history)
drawdowns = (peak - net_worth_history) / peak
max_drawdown = np.max(drawdowns) * 100

print("\n🔍 **DDPG Evaluation Complete!**")
print(f"📊 Total Steps: {step_count}")
print(f"💰 Starting Net Worth: ${starting_net_worth:.2f}")
print(f"💰 Final Net Worth: ${final_net_worth:.2f}")
print(f"📈 Profit/Loss: ${profit:.2f}")
print(f"📊 ROI: {roi:.2f}%")
print(f"🏆 Total Reward: {total_reward:.2f}")
print(f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
print(f"Action Distribution: {np.histogram(actions, bins=10, range=(-1, 1))}")
