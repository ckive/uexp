import os
import tensorflow as tf

# Create a function to implement a ModelCheckpoint callback with a specific filename 
def create_model_checkpoint(model_name, save_path="model_experiments"):
	return tf.keras.callbacks.ModelCheckpoint(
		filepath=os.path.join(save_path, model_name), # create filepath to save model
		verbose=0, # only output a limited amount of text
		save_best_only=True) # save only the best model to file

#import problem debugging
def print_hi():
	print("hi")

def evaluate_preds(y_true, y_pred):
	# Make sure float32 (for metric calculations)
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)

	# Calculate various metrics
	mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
	mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
	rmse = tf.sqrt(mse)
	mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
	mase = mean_absolute_scaled_error(y_true, y_pred)
	
	return {"mae": mae.numpy(),
			"mse": mse.numpy(),
			"rmse": rmse.numpy(),
			"mape": mape.numpy(),
			"mase": mase.numpy()}

# MASE implemented courtesy of sktime - https://github.com/alan-turing-institute/sktime/blob/ee7a06843a44f4aaec7582d847e36073a9ab0566/sktime/performance_metrics/forecasting/_functions.py#L16
def mean_absolute_scaled_error(y_true, y_pred):
	"""
	Implement MASE (assuming no seasonality of data).
	"""
	mae = tf.reduce_mean(tf.abs(y_true - y_pred))

	# Find MAE of naive forecast (no seasonality)
	mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shifting of 1 day)

	return mae / mae_naive_no_season