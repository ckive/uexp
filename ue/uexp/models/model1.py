import tensorflow as tf
from tensorflow.keras import layers
from .util import create_model_checkpoint, evaluate_preds
from ue.uexp.benchmarking.plots import plot_time_series
from .BaseModel import BaseModel
import matplotlib.pyplot as plt

"""
We're going to start by keeping it simple, model_1 will have:

A single dense layer with 128 hidden units and ReLU (rectified linear unit) activation
An output layer with linear activation (or no activation)
Adam optimizer and MAE loss function
Batch size of 128
10 epochs
"""

class Model_1(BaseModel):
    def __init__(self, df, horizon=1, window=7):
        # ingest data, clean, make windows, save train/test windows/labels
        super().__init__(df)
        self.horizon = horizon
        self.window = window

    def build_model(self, name="model_1", loss_type="mae", optimizer=tf.keras.optimizers.Adam()):
        # Set random seed for as reproducible results as possible
        # tf.random.set_seed(42)

        # Construct model
        self.model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(self.horizon, activation="linear") # linear activation is the same as having no activation                        
        ], name="model_1_dense") # give the model a name so we can save it

        # Compile model
        self.model.compile(loss=loss_type,
                        optimizer=optimizer)


    #def fit_model(self, train_windows, train_labels, test_windows, test_labels)
    def fit_model(self):
        # Fit model
        self.model.fit(x=self.train_windows,
                    y=self.train_labels, 
                    epochs=100,
                    verbose=1,
                    batch_size=128,
                    validation_data=(self.test_windows, self.test_labels),
                    callbacks=[create_model_checkpoint(model_name=self.model.name)]) # create ModelCheckpoint callback to save best model

    def evaluate_model(self, best=False):
        if best:
            best_loaded = tf.keras.models.load_model("model_experiments/model_1_dense")
            best_loaded.evaluate(self.test_windows, self.test_labels)
        else:
            self.model.evaluate(self.test_windows, self.test_labels)

    def make_preds(self, x=None):
        if x:
            #passing new data for making preds on
            forecast = self.model.predict(x)
        else:
            #default case
            forecast = self.model.predict(self.test_windows)
        return tf.squeeze(forecast) #into 1D
    
    def evaluate_preds(self, y_true=None, y_pred=None):
        if not y_true and not y_pred:
            #using self.test_windows and self.test_labels (default case)
            return evaluate_preds(y_true=tf.squeeze(self.test_labels), y_pred=self.make_preds())
        else:
            #make sure arrs are 1D (check later, now assume true)
            # workflow if default: use make_preds first, then pass as y_pred here
            return evaluate_preds(y_true=y_true, y_pred=y_pred)

    def visualize_pred(self, preds):
        """[summary]
        saves to model_experiments/{}-{}-{}.png
        """
        plt.figure(figsize=(10, 7))
        # Account for the test_window offset and index into test_labels to ensure correct plotting
        plot_time_series(timesteps=self.X_test[-len(self.test_windows):], values=self.test_labels[:, 0], label="Test_data")
        plot_time_series(timesteps=self.X_test[-len(self.test_windows):], values=preds, format="-", label="model_1_preds")
        
        plt.savefig("model_experiments/output.png")