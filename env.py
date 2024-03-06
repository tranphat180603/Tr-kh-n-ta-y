import pandas as pd
import gymnasium as gym
import gym_trading_env
import numpy
# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)
# df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

# Create the feature : ( close[t] - close[t-1] )/ close[t-1]
df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]

# Create the feature : volume[t] / max(*volume[t-7*24:t+1])
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

df.dropna(inplace= True) # Clean again !
# Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG) FULL VOLUME, GIÁ TRỊ NÀY CÓ THỂ NẰM TRONG KHOẢNG DECIMAL NẾU MÌNH ASSIGN LẠI CHO NÓ
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )
# Run an episode until it ends :
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    #Chọn policy ở chỗ này, policy hiện tại đang là random
    position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    observation, reward, done, truncated, info = env.step(position_index)
# At the end of the episode you want to render
env.unwrapped.save_for_render(dir = "render_logs")
