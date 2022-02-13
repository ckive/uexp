from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
from ue.uexp.dataprocessing.func import *
from ue.uexp.dataprocessing.ta import *
from ue.uexp.models.util import *

#======TA==============
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.wrapper import *
import ta
#=========================
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (20, 10)

import warnings
warnings.filterwarnings('ignore')

# get data
### get data (set config params)
ticker_list = ["BTCUSDT"]
start_date = "2022-02-03"
end_date = "2022-02-04"
time_interval = "1s"

p = BinanceProcessor("binance")
df = p.download_data(ticker_list, start_date, end_date, time_interval)

df.drop(["tic"], axis=1, inplace=True)
df.columns = ["Open","High","Low","Close","Volume"]

### INDICATORS
window = 30 # a choice

# TSI Indicator
df["momentum_tsi"] = TSIIndicator(
    close=df["Close"], window_slow=25, window_fast=13, fillna=True
).tsi()/100 #scale 0-1


def stationize(series):
    return np.log(series / series.shift(1))

df['log_ret'] = stationize(df['Close'])

df.drop(["Open","High","Low", "Volume"], axis=1, inplace=True)

HORIZON = 1
WINDOW = 30

# Add windowed columns
for i in range(WINDOW-1): # Shift values for each step in WINDOW_SIZE
  df[f"log_ret+{i+2}"] = df["log_ret"].shift(periods=i+2)
df.head(50)