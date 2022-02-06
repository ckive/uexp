from ue.uexp.models.nbeats import NBeatsBlock
from ue.uexp.models.util import evaluate_preds
from ue.uexp.benchmarking.simpleplot import plot_time_series
from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
import tensorflow as tf


### get data (set config params)
ticker_list = ["BTCUSDT"]
start_date = "2021-12-01"
end_date = "2021-12-31"
time_interval = "1m"

p = BinanceProcessor("binance")
df = p.download_data(ticker_list, start_date, end_date, time_interval)
p.add_technical_indicator(["stochrsi"], use_stockstats=True)
df = df[['time','close']]

### import model, run
HORIZON = 1
WINDOW_SIZE = 7

# Set up dummy NBeatsBlock layer to represent inputs and outputs
dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE, 
                                       theta_size=WINDOW_SIZE+HORIZON, # backcast + forecast 
                                       horizon=HORIZON,
                                       n_neurons=128,
                                       n_layers=4)

# Create dummy inputs (have to be same size as input_size)
dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0) # input shape to the model has to reflect Dense layer input requirements (ndim=2)
dummy_inputs

# Pass dummy inputs to dummy NBeatsBlock layer
backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
# These are the activation outputs of the theta layer (they'll be random due to no training of the model)
print(f"Backcast: {tf.squeeze(backcast.numpy())}")
print(f"Forecast: {tf.squeeze(forecast.numpy())}")