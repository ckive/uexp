#======Core============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping
import torch
#======Darts===========
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.dataprocessing import Pipeline
#======UEXP============
from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
from ue.uexp.dataprocessing.func import *
#from ue.uexp.dataprocessing.ta import *
#from ue.uexp.models.util import *
#======TA==============
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.wrapper import *
import ta

import warnings
warnings.filterwarnings('ignore')

### get data (set config params)
ticker_list = ["BTCUSDT"]
start_date = "2021-12-30"
end_date = "2021-12-31"
time_interval = "1s"

p = BinanceProcessor("binance")
df = p.download_data(ticker_list, start_date, end_date, time_interval)

df.drop("tic", axis=1, inplace=True)

df.columns = ["Open","High","Low","Close","Volume"]

def stationize(series):
    return np.log(series / series.shift(1))

# Log return
df['log_ret'] = stationize(df['Close'])

# Relative Strength Index (RSI)
df["momentum_rsi"] = RSIIndicator(
close=df["Close"], window=14, fillna=True
).rsi()/100 #scale 0-1

# Stoch RSIs (StochRSI)
indicator_srsi = StochRSIIndicator(
close=df["Close"], window=14, smooth1=3, smooth2=3, fillna=True
)
df["momentum_stoch_rsi"] = indicator_srsi.stochrsi()

# Money Flow Index
df["volume_mfi"] = MFIIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], fillna=True
        ).money_flow_index()/100 #norm to 0-1

# SMAs
df["trend_sma_fast"] = SMAIndicator(
    close=df["Close"], window=12, fillna=True
).sma_indicator()
# EMAs
df["trend_ema_fast"] = EMAIndicator(
    close=df["Close"], window=12, fillna=True
).ema_indicator()

N = 30
df.drop(index=df.index[:N], 
        axis=0, 
        inplace=True)
df.dropna(inplace=True)

dfts = TimeSeries.from_dataframe(df)

# Train set
train_df = dfts[:int(0.8*df.shape[0])]

scaler_target_tr = Scaler()
# [0] because singular
scaled_target = scaler_target_tr.fit_transform(train_df['Close'])

cov_ts = [train_df['Open'], train_df['High'], train_df['Volume'], 
               train_df['log_ret'], train_df['trend_sma_fast'], train_df['trend_ema_fast']]
# the other indicators are already scaled

scaler_cov_tr = Scaler()
pipeline_train = Pipeline([scaler_cov_tr])
transformed_train = pipeline_train.fit_transform(cov_ts)

# Val set
val_df = dfts[int(0.8*df.shape[0]):]

scaler_target_val = Scaler()
scaled_val_target = scaler_target_val.fit_transform(val_df['Close'])

multiple_ts_val = [val_df['Open'], val_df['High'], val_df['Volume'], 
               val_df['log_ret'], val_df['trend_sma_fast'], val_df['trend_ema_fast']]
# the other indicators are already scaled

scaler_cov_val = Scaler()
pipeline_val = Pipeline([scaler_cov_val])
transformed_val= pipeline_val.fit_transform(multiple_ts_val)

model = NBEATSModel(input_chunk_length=60, output_chunk_length=15, random_state=42)
from darts import concatenate

tr_covs = concatenate(transformed_train, axis="component")
val_covs = concatenate(transformed_val, axis="component")

#series = 
model.fit(
    series=scaled_target, 
    past_covariates=tr_covs,
    val_series=scaled_val_target, 
    val_past_covariates=val_covs,
    epochs=50, 
    verbose=True)

# training was for horizon=15, if predict here horizon>15 then need historic_future_covs
pred_train = model.predict(series=scaled_target, n=15)

# scale back:
pred_train_scaled = scaler_target_tr.inverse_transform(pred_train)

# PLOT AND SAVE
plt.figure(figsize=(10, 6))
# prev train
train_df['Close'][-100:].plot(label="past prices")
#actual
val_df['Close'][:15].plot(label="actual prices")
# pred
pred_train_scaled.plot(label="pred prices")

plt.savefig('exp1')


# # Historical Forecast
# hist_for = model.historical_forecasts(
#     series=scaled_target,
#     start=0.95, #just the last 5 percent
#     forecast_horizon=15,
#     stride=5,
#     retrain=False,
#     verbose=True,
# )
# # scale back:
# hist_for = scaler.inverse_transform(hist_for)