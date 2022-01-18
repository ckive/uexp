import pandas as pd
import numpy as np
from ..benchmarking.simpleplot import plot_time_series

# should rename this to data processor xD

class BaseModel:
    def __init__(self, data):
        # default horizon and window
        self.horizon = 1
        self.window = 7

        df = self.clean_data(data)
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split(df)

        prices = df['close'].to_numpy()
        full_windows, full_labels = self.make_windows(prices, self.window, self.horizon)
        self.train_windows, self.test_windows, self.train_labels, self.test_labels = self.make_train_test_splits(full_windows, full_labels)

    def clean_data(self, df):
        """[summary]
        assumes |time|open|high|low|close|adj_close|volume|
        makes ohlc stationary, time->pd.datetime, normalizes volume
        df.index <- time

        Args:
            df ([pd.DataFrame]): [data to clean]
        """
        
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df['close'] = self.stationize(df['close'])

        # for multivariate (OPEN LATER)
        # for cols in ["open","high","low","close"]:
        #     df[cols] = self.stationize(df[cols])
        # df['volume'] = (df['volume'] - np.min(df['volume'])) / (np.max(df['volume']) - np.min(df['volume']))
        df.dropna(inplace=True)
        return df

    # NOT FOR WINDOWS, USE WINDOWS INSTEAD
    def train_test_split(self, df, pct=0.8):
        """[summary]
        pct amount for training, 1-pct amount for validation
        using *correct* method of train-test split for time-series data

        Args:
            df ([pd.DataFrame]): [cleaned data, if 2D only time and price, if more cols, multivariate]
        """
        ### NOTE: univariate for now (bc i dun get the multi one from tutorial yet)
        assert df.shape[1] == 1

        # Get bitcoin date array
        timesteps = df.index.to_numpy()
        #prices = df["Price"].to_numpy()

        # Create train and test splits the right way for time series data
        split_size = int(0.8 * len(df)) # 80% train, 20% test        

        # Create train data splits (everything before the split)
        X_train, y_train = timesteps[:split_size], df[:split_size]

        # Create test data splits (everything after the split)
        X_test, y_test = timesteps[split_size:], df[split_size:]

        return X_train, y_train, X_test, y_test

    def naive_model(self, df):
        # number of timesteps to predict into the future
        #horizon =
        # number of timesteps from past used to predict horizon
        #window = 

        #e.g. use 1 week to predict tomorrow's price, h=1, w=7

        """
        naive model 0 training
        y_hat_t = y_t-1
        
        prediction at time t is equal to value at t-1
        """
        x_train, y_train, x_test, y_test = self.train_test_split(df)

        naive_forecast = y_test[:-1]
        self.show(x_train, y_train, label='Train Data')
        #self.show(timesteps=x_test, values=y_test, label="Test data")
        self.show(timesteps=x_test[1:], values=naive_forecast, format="-", label="Naive forecast");

        #zoom in look to compare forecast and true
        offset = len(x_train) # only look at prediction period 
        self.show(timesteps=x_test, values=y_test, start=offset, label="Test data")
        self.show(timesteps=x_test[1:], values=naive_forecast, format="-", start=offset, label="Naive forecast");


    def stationize(self, series):
        """[summary]
        helper of clean_data to make series stationary

        Args:
            series ([pd.core.series.Series]): [a column of df]
        """
        return np.log(series / series.shift(1))

    def show(self, x, y, label):
        plot_time_series(x, y, label=label)


    def get_labelled_windows(self, x, horizon=1):
        """
        Creates labels for windowed dataset.

        E.g. if horizon=1 (default)
        Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
        """
        return x[:, :-horizon], x[:, -horizon:]

    # Create function to view NumPy arrays as windows 
    def make_windows(self, x, window_size=7, horizon=1):
        """
        Turns a 1D array into a 2D array of sequential windows of window_size.
        """
        # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
        window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
        # print(f"Window step:\n {window_step}")

        # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
        window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
        # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

        # 3. Index on the target array (time series) with 2D array of multiple window steps
        windowed_array = x[window_indexes]

        # 4. Get the labelled windows
        windows, labels = self.get_labelled_windows(windowed_array, horizon=horizon)

        return windows, labels

    # Make the train/test splits
    def make_train_test_splits(self, windows, labels, test_split=0.2):
        """
        Splits matching pairs of windows and labels into train and test splits.
        """
        split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
        train_windows = windows[:split_size]
        train_labels = labels[:split_size]
        test_windows = windows[split_size:]
        test_labels = labels[split_size:]
        return train_windows, test_windows, train_labels, test_labels
  