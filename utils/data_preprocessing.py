import pandas as pd
import numpy as np
import os

# Load raw data
raw_data_path = "data/raw/BTCUSDT_H1.csv"
processed_data_path = "data/processed/BTCUSDT_H1_processed.csv"

df = pd.read_csv(raw_data_path)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Handle missing values
df = df.dropna()

# Calculate Technical Indicators
def calculate_indicators(df):
    # Simple Moving Average (SMA)
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_50"] = df["close"].rolling(window=50).mean()

    # Exponential Moving Average (EMA)
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR) for volatility-based stop-loss
    df["High-Low"] = df["high"] - df["low"]
    df["High-Close"] = abs(df["high"] - df["close"].shift(1))
    df["Low-Close"] = abs(df["low"] - df["close"].shift(1))
    df["True_Range"] = df[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean()

    return df

df = calculate_indicators(df)

# Market Regime Classification (Bullish, Bearish, Sideways)
def classify_market_regime(df):
    df["Market_Regime"] = np.where(df["SMA_20"] > df["SMA_50"], "Bullish", "Bearish")
    df.loc[(df["SMA_20"].between(df["SMA_50"] - 100, df["SMA_50"] + 100)), "Market_Regime"] = "Sideways"
    return df

df = classify_market_regime(df)

# Drop NaN rows (caused by rolling indicators)
df = df.dropna()

# Save processed data
os.makedirs("data/processed", exist_ok=True)
df.to_csv(processed_data_path, index=False)

print(f"✅ Data preprocessing complete! Saved to {processed_data_path}")
