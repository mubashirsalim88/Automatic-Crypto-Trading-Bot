import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.ticker import PercentFormatter

# Load data from final run
df = pd.read_csv("data/processed/BTCUSDT_H1_processed.csv")
test_df = df.iloc[int(0.8 * len(df)):].reset_index(drop=True)  # 3491 steps
regimes = test_df["Market_Regime"].values

# Load real net_worth_history and actions
with open("research/net_worth_history.pkl", "rb") as f:
    net_worth_history = pickle.load(f)
with open("research/actions.pkl", "rb") as f:
    actions = pickle.load(f)
steps = len(actions)  # 3490
action_indices = [i for i in range(steps) if abs(actions[i]) > 0.3]  # Trades

# Plot 1: Net Worth with Regime Overlay
plt.figure(figsize=(12, 6))
plt.plot(net_worth_history, label="Net Worth", color="black", linewidth=1.5)
colors = {"Bullish": "green", "Bearish": "red", "Sideways": "blue"}
for i in range(steps):
    plt.axvspan(i, i+1, alpha=0.1, color=colors[regimes[i]], zorder=0)
plt.title("Net Worth with Market Regime Overlay", fontsize=14)
plt.xlabel("Step (Hourly)", fontsize=12)
plt.ylabel("Net Worth ($)", fontsize=12)
plt.legend(["Net Worth"] + list(colors.keys()), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("research/net_worth_regimes.png", dpi=300)

# Plot 2: Cumulative Returns with Drawdown
returns = (np.array(net_worth_history[1:]) / net_worth_history[0] - 1) * 100
peak = np.maximum.accumulate(net_worth_history)
drawdowns = (peak - net_worth_history) / peak * 100
plt.figure(figsize=(12, 6))
plt.plot(returns, label="Cumulative Return", color="blue", linewidth=1.5)
plt.fill_between(range(steps), 0, -drawdowns[1:], color="red", alpha=0.3, label="Drawdown")
plt.title("Cumulative Returns and Drawdown", fontsize=14)
plt.xlabel("Step (Hourly)", fontsize=12)
plt.ylabel("Return (%)", fontsize=12)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("research/cumulative_returns.png", dpi=300)

# Plot 3: Action Distribution with Trade Timing
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.hist(actions, bins=10, range=(-1, 2), color="gray", edgecolor="black")
plt.title("Action Distribution and Trade Timing", fontsize=14)
plt.xlabel("Action Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.subplot(2, 1, 2)
plt.scatter(action_indices, [actions[i] for i in action_indices], c=[colors[regimes[i]] for i in action_indices], s=10)
plt.xlabel("Step (Hourly)", fontsize=12)
plt.ylabel("Action Value", fontsize=12)
plt.tight_layout()
plt.savefig("research/actions_trades.png", dpi=300)

# Plot 4: Rolling Sharpe and Drawdown
rolling_returns = np.diff(net_worth_history) / net_worth_history[:-1]
window = 24  # 1-day rolling (H1 data)
rolling_sharpe = [np.mean(rolling_returns[i:i+window]) / np.std(rolling_returns[i:i+window]) * np.sqrt(24) 
                  if i+window <= len(rolling_returns) else np.nan 
                  for i in range(len(rolling_returns))]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rolling_sharpe, label="Rolling Sharpe (24h)", color="purple")
plt.title("Rolling Sharpe Ratio and Drawdown", fontsize=14)
plt.xlabel("Step (Hourly)", fontsize=12)
plt.ylabel("Sharpe Ratio", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.subplot(2, 1, 2)
plt.plot(drawdowns[1:], label="Drawdown", color="red")
plt.xlabel("Step (Hourly)", fontsize=12)
plt.ylabel("Drawdown (%)", fontsize=12)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("research/risk_metrics.png", dpi=300)

print("Plots saved in 'research/' directory.")
