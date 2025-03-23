import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DDPG, A2C
from environments.crypto_trading_env import CryptoTradingEnv
from environments.crypto_trading_env_ddpg import CryptoTradingEnvDDPG
import pickle  # Added for saving data

# Load full test data
df = pd.read_csv("data/processed/BTCUSDT_H1_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
test_df = df.iloc[int(0.8 * len(df)):].reset_index(drop=True)

# Calculate MACD
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df["close"].ewm(span=short_period, adjust=False).mean()
    long_ema = df["close"].ewm(span=long_period, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    return df

test_df = calculate_macd(test_df)
print(f"Test data length: {len(test_df)}")  # 3491

# Initialize environments
env_discrete = CryptoTradingEnv(test_df)
env_continuous = CryptoTradingEnvDDPG(test_df)

# Load trained models
ppo_model = PPO.load("models/PPO/ppo_crypto_trader")
ddpg_model = DDPG.load("models/DDPG/ddpg_crypto_trader_final")
a2c_model = A2C.load("models/A2C/a2c_crypto_trader")

# Ensemble evaluation
obs_discrete, _ = env_discrete.reset()
obs_continuous, _ = env_continuous.reset()
total_reward = 0
step_count = 0
starting_net_worth = env_discrete.net_worth
net_worth_history = [starting_net_worth]
actions = []

while step_count < len(test_df) - 1:
    regime = test_df.loc[step_count, "Market_Regime"]
    rsi = test_df.loc[step_count, "RSI_14"]
    macd = test_df.loc[step_count, "MACD"]
    macd_signal = test_df.loc[step_count, "MACD_Signal"]
    trend_strength = abs(rsi - 50) / 50
    trade_fraction = max(0.33, trend_strength * 0.95)  # 33–95%
    if rsi > 60 and macd > macd_signal:  # Bullish with MACD confirmation
        trade_fraction = min(trade_fraction * 2, 1.0)
    elif rsi < 40 and macd < macd_signal:  # Bearish with MACD confirmation
        trade_fraction = min(trade_fraction * 3, 1.0)

    # Get predictions
    ppo_action, _ = ppo_model.predict(obs_discrete, deterministic=True)
    ddpg_action, _ = ddpg_model.predict(obs_continuous, deterministic=True)
    a2c_action, _ = a2c_model.predict(obs_discrete, deterministic=True)

    # Map to continuous
    ppo_val = 0 if ppo_action == 0 else (1 if ppo_action == 1 else -1)
    a2c_val = 0 if a2c_action == 0 else (1 if a2c_action == 1 else -1)
    ddpg_val = ddpg_action[0]

    # Blend by regime
    if regime == "Bullish":
        final_action = 0.7 * ppo_val + 0.2 * ddpg_val + 0.1 * a2c_val
    elif regime == "Bearish":
        final_action = 0.1 * ppo_val + 0.8 * ddpg_val + 0.1 * a2c_val
    else:  # Sideways
        final_action = 0.2 * ppo_val + 0.2 * ddpg_val + 0.6 * a2c_val

    # MACD override
    if macd > macd_signal and final_action < 0:  # Bullish MACD, no sell
        final_action = max(final_action, 0.0)
    elif macd < macd_signal and final_action > 0:  # Bearish MACD, no buy
        final_action = min(final_action, 0.0)

    # Decide action with stop-loss and profit-taking
    trade_amount = max(env_continuous.balance * trade_fraction, env_continuous.min_trade_amount)
    current_price = test_df.loc[step_count, "close"]
    if abs(final_action) > 0.3:
        if final_action > 0 and env_continuous.balance >= trade_amount:
            action = np.array([max(final_action, 0.7)])  # Buy
        elif final_action < 0 and env_continuous.crypto_held > 0:
            if env_continuous.entry_price:
                if current_price < env_continuous.entry_price * 0.95:  # 5% stop-loss
                    action = np.array([-1.0])
                elif current_price > env_continuous.entry_price * 1.08:  # 8% profit-taking
                    action = np.array([-1.0])
                else:
                    action = np.array([min(final_action, -0.7)])  # Sell
            else:
                action = np.array([min(final_action, -0.7)])
        elif final_action < -0.5 and env_continuous.crypto_held == 0:  # Short
            action = np.array([-1.0])
        else:
            action = np.array([0.0])
    else:
        action = np.array([0.0])

    # Execute
    if action[0] < -0.5 and env_continuous.crypto_held == 0:  # Short
        short_amount = trade_amount / current_price
        env_continuous.crypto_held -= short_amount
        env_continuous.balance += trade_amount
        env_continuous.entry_price = current_price
        reward = 0.01
        print(f"Step {step_count}: 📉 Shorted {short_amount:.6f} at ${current_price:.2f}")
    elif action[0] > 0.7 and env_continuous.crypto_held < 0:  # Cover
        short_amount = -env_continuous.crypto_held
        cover_cost = short_amount * current_price
        profit = (env_continuous.entry_price - current_price) * short_amount
        env_continuous.balance -= cover_cost
        env_continuous.crypto_held = 0
        env_continuous.entry_price = None
        reward = profit / env_continuous.initial_balance
        print(f"Step {step_count}: 📈 Covered {short_amount:.6f} at ${current_price:.2f} (Profit: ${profit:.2f})")
    else:
        obs_continuous, reward, done, _, _ = env_continuous.step(action)

    # Sync
    env_discrete.balance = env_continuous.balance
    env_discrete.crypto_held = env_continuous.crypto_held
    env_discrete.net_worth = env_continuous.net_worth
    env_discrete.entry_price = env_continuous.entry_price
    env_discrete.hold_steps = env_continuous.hold_steps
    env_discrete.last_trade_step = env_continuous.last_trade_step
    env_discrete.current_step = step_count
    env_continuous.current_step = step_count
    obs_discrete = env_discrete._next_observation()
    obs_continuous = env_continuous._next_observation()

    total_reward += reward
    actions.append(action[0])
    print(f"Step {step_count}: Regime: {regime}, Action: {action[0]}")
    env_discrete.render()
    step_count += 1
    net_worth_history.append(env_discrete.net_worth)

    # Save data after each step (overwrite to keep latest)
    with open("research/net_worth_history.pkl", "wb") as f:
        pickle.dump(net_worth_history, f)
    with open("research/actions.pkl", "wb") as f:
        pickle.dump(actions, f)

# Final metrics
final_net_worth = env_discrete.net_worth
profit = final_net_worth - starting_net_worth
roi = (profit / starting_net_worth) * 100
returns = np.diff(net_worth_history) / net_worth_history[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24)
peak = np.maximum.accumulate(net_worth_history)
drawdowns = (peak - net_worth_history) / peak
max_drawdown = np.max(drawdowns) * 100

print("\n🔍 **Ensemble Evaluation Complete!**")
print(f"📊 Total Steps: {step_count}")
print(f"💰 Starting Net Worth: ${starting_net_worth:.2f}")
print(f"💰 Final Net Worth: ${final_net_worth:.2f}")
print(f"📈 Profit/Loss: ${profit:.2f}")
print(f"📊 ROI: {roi:.2f}%")
print(f"🏆 Total Reward: {total_reward:.2f}")
print(f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
print(f"Action Distribution: {np.histogram(actions, bins=10, range=(-1, 2))}")

# Final save (redundant but ensures completion)
with open("research/net_worth_history.pkl", "wb") as f:
    pickle.dump(net_worth_history, f)
with open("research/actions.pkl", "wb") as f:
    pickle.dump(actions, f)
print("Data saved to 'research/net_worth_history.pkl' and 'research/actions.pkl'")
