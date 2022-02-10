import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path,
                              model_name),  # create filepath to save model
        verbose=0,  # only output a limited amount of text
        save_best_only=True)  # save only the best model to file


#import problem debugging
def print_hi():
    print("hi")


def evaluate_preds(y_true, y_pred):
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(
        y_true,
        y_pred)  # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy()
    }


# MASE implemented courtesy of sktime - https://github.com/alan-turing-institute/sktime/blob/ee7a06843a44f4aaec7582d847e36073a9ab0566/sktime/performance_metrics/forecasting/_functions.py#L16
def mean_absolute_scaled_error(y_true, y_pred):
    """
	Implement MASE (assuming no seasonality of data).
	"""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season = tf.reduce_mean(
        tf.abs(y_true[1:] - y_true[:-1])
    )  # our seasonality is 1 day (hence the forward_horizoning of 1 day)

    return mae / mae_naive_no_season


class WindowGenerator():

    def __init__(self,
                 data,
                 back_window=30,
                 forward_window=1,
                 forward_horizon=1,
                 splitpct=(0.7, 0.9),
                 embargo=False,
                 label_columns=None):

        # Work out the window parameters.
        self.back_window = back_window
        self.forward_window = forward_window
        self.forward_horizon = forward_horizon

        self.total_window_size = back_window + forward_horizon

        self.input_slice = slice(0, back_window)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.forward_window
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]

        # Store the raw data.
        self.train_df = data[:int(splitpct[0] * len(data))]
        self.val_df = data[int(splitpct[0] * len(data)):int(splitpct[1] *
                                                            len(data))]
        self.test_df = data[int(splitpct[1] * len(data)):]

        #Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i
                for i, name in enumerate(label_columns)
            }
        self.column_indices = {
            name: i
            for i, name in enumerate(self.train_df.columns)
        }

        # Model datasets
        self.train_ds = self.make_dataset(self.train_df)
        self.val_ds = self.make_dataset(self.val_df)
        self.test_ds = self.make_dataset(self.test_df)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([
                labels[:, :, self.column_indices[name]]
                for name in self.label_columns
            ],
                              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.back_window, None])
        labels.set_shape([None, self.forward_window, None])

        return inputs, labels

    # def plot(self, model):
    #     #only predict next 10 seconds
    #     y_preds = model(self.test_ds[:10])

    # RIGHT NOW, PREDICTING AND PLOTTING NEXT 10 SECONDS
    def plot_preds(self, model, plot_col='log_ret', max_subplots=10):
        inputs, labels = self.example

        y_preds = model(self.test_ds[:10])

        # for

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices,
                     inputs[n, :, plot_col_index],
                     label='Inputs',
                     marker='.',
                     zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices,
                        labels[n, :, label_col_index],
                        edgecolors='k',
                        label='Labels',
                        c='#2ca02c',
                        s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X',
                            edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e',
                            s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [s]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    def get_example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train_ds))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            #f'Label column name(s): {self.label_columns}'
        ])
