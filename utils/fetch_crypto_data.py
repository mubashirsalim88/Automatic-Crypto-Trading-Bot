import ccxt
import pandas as pd
import time
import os

# Replace with your actual API keys (keep these secure!)
api_key = "x6xZIHM1eHH14zXT0QpqgwcKYiGBN7MCui2WXJ5moycoIOzIylklrrU68sTG8YEJ"
api_secret = "ne4v8clMMGk4GVsUk4Lsl3k0vglrP7xURGgO4lxeO9YE4oKOgpPSR7Kp99qNvEZY"

# Initialize Binance API
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'adjustForTimeDifference': True},
    'rateLimit': 1200,  # Respect API rate limits
})

# Set trading pair & timeframe
symbol = "BTC/USDT"
timeframe = "1h"  # Changed from 15m to 1h
limit = 500  # Max candles per request

# Set data save path
raw_data_path = "data/raw"
os.makedirs(raw_data_path, exist_ok=True)
filename = f"{raw_data_path}/BTCUSDT_H1.csv"

# Convert timeframe to milliseconds
tf_map = {'1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000, '1d': 86400000}

# Set data range (Approx. 2 years for research efficiency)
since = binance.parse8601("2023-01-01T00:00:00Z")  # Adjust as needed

all_candles = []
fetch_count = 0  # Track number of requests

print(f"Fetching historical OHLCV data for {symbol} ({timeframe})...")

while True:
    try:
        candles = binance.fetch_ohlcv(symbol, timeframe, since, limit)
        if not candles:
            break  # Stop if no more data

        all_candles.extend(candles)
        since = candles[-1][0] + tf_map[timeframe]  # Move to next batch
        fetch_count += 1

        print(f"Fetched {len(all_candles)} candles so far... (Batch {fetch_count})")
        time.sleep(1)  # Avoid API rate limits

        # Stop if we have around 2 years of data (365 * 24 * 2 ≈ 17500 candles)
        if len(all_candles) >= 17500:
            break  

    except Exception as e:
        print(f"Error: {e}")
        break

# Convert to DataFrame
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert to datetime

# Save to CSV
df.to_csv(filename, index=False)
print(f"✅ Data collection complete! {len(df)} rows saved to {filename}")
