import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class CryptoTradingEnvDDPG(gym.Env):
    def __init__(self, df, initial_balance=10000, risk_factor=0.02, max_trade_fraction=0.7, max_hold_time=30):
        super(CryptoTradingEnvDDPG, self).__init__()

        self.df = df.copy().reset_index(drop=True)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.features = [
            "open", "high", "low", "close", "volume",
            "SMA_20", "SMA_50", "EMA_20", "RSI_14",
            "MACD", "Signal_Line", "ATR_14",
            "balance_norm", "position_value_norm"
        ]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.current_step = 0
        self.entry_price = None
        self.hold_steps = 0

        self.risk_factor = risk_factor
        self.max_trade_fraction = max_trade_fraction
        self.max_hold_time = max_hold_time
        self.min_trade_amount = 50
        self.buy_cooldown = 3
        self.last_trade_step = -self.buy_cooldown

        self.feature_mins = self.df[self.features[:-2]].min()
        self.feature_maxs = self.df[self.features[:-2]].max()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.entry_price = None
        self.hold_steps = 0
        self.last_trade_step = -self.buy_cooldown
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.loc[self.current_step, self.features[:-2]].values
        current_price = self.df.loc[self.current_step, "close"]
        obs = (obs - self.feature_mins) / (self.feature_maxs - self.feature_mins + 1e-6)
        balance_norm = self.balance / self.initial_balance
        position_value_norm = (self.crypto_held * current_price) / self.initial_balance
        obs = np.append(obs, [balance_norm, position_value_norm])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, "close"]
        atr = self.df.loc[self.current_step, "ATR_14"]

        spread = current_price * 0.0005
        slippage = np.random.uniform(-0.0005, 0.0005) * current_price
        execution_price = current_price + spread + slippage

        trend_strength = abs(self.df.loc[self.current_step, "RSI_14"] - 50) / 50
        trade_percentage = np.clip(((action[0] + 1) / 2) * trend_strength, 0.1, self.max_trade_fraction)
        trade_amount = self.balance * trade_percentage
        reward = 0

        if trade_amount < self.min_trade_amount:
            trade_amount = 0

        if (action[0] > 0.2 and self.balance >= trade_amount and trade_amount > 0 and
            self.current_step >= self.last_trade_step + self.buy_cooldown):
            crypto_bought = trade_amount / execution_price
            self.crypto_held += crypto_bought
            self.balance -= trade_amount
            self.entry_price = execution_price  # Set entry price on buy
            self.hold_steps = 0
            self.last_trade_step = self.current_step
            reward += 0.01
            print(f"Step {self.current_step}: 📈 Bought ${trade_amount:.2f} worth at ${execution_price:.2f}")

        elif action[0] < -0.3 and self.crypto_held > 0:
            sell_amount = min(trade_amount / execution_price, self.crypto_held)
            sell_value = sell_amount * execution_price
            profit = sell_value - (sell_amount * self.entry_price if self.entry_price is not None else 0)

            if sell_amount > 0:
                self.balance += sell_value
                self.crypto_held -= sell_amount
                reward += profit / self.initial_balance
                if profit > 0:
                    reward += 0.05  # Bonus for profit
                if self.crypto_held <= 1e-8:  # Threshold for "zero"
                    self.crypto_held = 0
                    self.entry_price = None
                print(f"Step {self.current_step}: 📉 Sold {sell_amount:.6f} at ${execution_price:.2f} (Profit: ${profit:.2f})")

        if self.crypto_held > 0 and self.entry_price is not None:
            max_price = self.df.loc[self.last_trade_step:self.current_step, "close"].max()
            if current_price < max_price - 2 * atr:
                sell_value = self.crypto_held * execution_price
                profit = sell_value - (self.crypto_held * self.entry_price)
                self.balance += sell_value
                self.crypto_held = 0
                self.entry_price = None
                reward += profit / self.initial_balance
                print(f"Step {self.current_step}: 🔒 Trailing Stop at ${execution_price:.2f} (Profit: ${profit:.2f})")
            else:  # Only check stop-loss if trailing stop didn’t trigger
                stop_loss_multiplier = 3 + trend_strength
                if current_price < self.entry_price - stop_loss_multiplier * atr:
                    sell_value = self.crypto_held * execution_price
                    profit = sell_value - (self.crypto_held * self.entry_price)
                    self.balance += sell_value
                    self.crypto_held = 0
                    self.entry_price = None
                    reward -= abs(profit) / self.initial_balance
                    print(f"Step {self.current_step}: 🛑 Stop-Loss Triggered at ${execution_price:.2f} (Profit: ${profit:.2f})")

        if self.hold_steps >= self.max_hold_time and self.crypto_held > 0:
            sell_value = self.crypto_held * execution_price
            profit = sell_value - (self.crypto_held * self.entry_price)
            self.balance += sell_value
            self.crypto_held = 0
            self.entry_price = None
            reward -= 0.1
            print(f"Step {self.current_step}: ⚠️ Forced Exit at ${execution_price:.2f} (Profit: ${profit:.2f})")

        if self.crypto_held > 0 and self.entry_price is not None:
            if current_price > self.entry_price:
                reward += 0.02 * (current_price - self.entry_price) / self.entry_price
        elif self.crypto_held == 0 and trend_strength > 0.5:
            reward -= 0.002

        new_net_worth = self.balance + (self.crypto_held * execution_price)
        reward += 1.5 * (new_net_worth - self.net_worth) / self.initial_balance
        self.net_worth = new_net_worth
        if self.crypto_held > 0:
            self.hold_steps += 1

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
              f"Crypto Held: {self.crypto_held:.6f}, Net Worth: ${self.net_worth:.2f}")
