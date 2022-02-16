from ue.uexp.models.BaseModel import BaseModel
from ue.uexp.models.model1 import Model_1
from ue.uexp.models.util import evaluate_preds
from ue.uexp.benchmarking.plots import plot_time_series
from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
import tensorflow as tf


### get data (set config params)
ticker_list = ["BTCUSDT"]
start_date = "2021-12-30"
end_date = "2021-12-31"
time_interval = "1m"

p = BinanceProcessor("binance")
df = p.download_data(ticker_list, start_date, end_date, time_interval)
df = df[['time','close']]

### import model, run
HORIZON = 1
WINDOW_SIZE = 7

model = Model_1(df, HORIZON, WINDOW_SIZE)
model.build_model()
model.fit_model()

# get results and evaluate
print("evaluate model")
model.evaluate_model()
print("evaluate best model")
model.evaluate_model(best=True)


# see predictions (saves plot)
test_set_preds = model.make_preds()
model.visualize_pred(test_set_preds)


# evaluate predictions
against_test_set = model.evaluate_preds()
print(against_test_set)


print("done")
