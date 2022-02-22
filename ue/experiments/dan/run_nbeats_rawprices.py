#======Core============
import pandas as pd
import numpy as np
import os, sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
#======PyForecast======
from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE
#======UEXP============
from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
from ue.uexp.dataprocessing.func import *
from ue.uexp.dataprocessing.ta import *
from ue.uexp.models.util import *
from ue.uexp.benchmarking.gifify import build_gif

import warnings

warnings.filterwarnings('ignore')


def run_nbeats_rawprices_experiment(start,
                                    end,
                                    contract,
                                    max_encoder_length=60,
                                    max_prediction_length=20,
                                    prediction_length=30,
                                    **training_params):
    """Summary or Description of the Function

    Parameters:
        argument1 (int): Description of arg1

    Returns:
        int:Returning value

   """
    exp_name = "{}_{}_of_{}_mel-{}_mpl-{}_for_{}".format(
        start, end, contract, max_encoder_length, max_prediction_length,
        prediction_length)
    basepath = os.path.join('experiment_plots', exp_name)
    os.makedirs(basepath, exist_ok=True)

    time_interval = "1s"
    p = BinanceProcessor("binance")
    df = p.download_data([contract], start, end, time_interval)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    df['time_idx'] = df.index
    df.drop(['open', 'high', 'low', 'volume', 'tic'], inplace=True, axis=1)
    df['series'] = 0
    df['time'] = pd.to_datetime(df['time'])
    ensure_pred_len = max_prediction_length + prediction_length
    training_cutoff = df["time_idx"].max() - ensure_pred_len

    #TSDS
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="close",
        categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
        group_ids=["series"],
        # only unknown variable is "close" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["close"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=training_cutoff + 1)

    # ^ Generate dataset with different underlying data but same variable encoders and scalers, etc.
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True,
                                              batch_size=batch_size,
                                              num_workers=0)
    val_dataloader = validation.to_dataloader(train=False,
                                              batch_size=batch_size,
                                              num_workers=0)

    # Fitting
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     #i dont think 0.1 is possible atm.
    #     stopping_threshold=0.1,
    #     patience=5,
    #     verbose=False,
    #     mode="min")

    trainer = pl.Trainer(
        auto_lr_find=True,
        max_epochs=100,
        # devices="auto",
        # accelerator="auto",
        # strategy="ddp",
        gpus=1,
        gradient_clip_val=0.01,
        default_root_dir="model_checkpoints",
        # callbacks=[early_stop_callback],
        #limit_train_batches=30, # :O WTF! this is why it's under performing, only 31 epochs..............
    )

    net = NBeats.from_dataset(
        training,
        learning_rate=1e-2,  #bigger LR from original 4e-3
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0,
    )

    # find optimal lr and use it (WTF FAILS)
    # lr_finder = trainer.tuner.lr_find(net,
    #                                   train_dataloader=train_dataloader,
    #                                   val_dataloaders=val_dataloader,
    #                                   min_lr=1e-5)
    # net.hparams.lr = lr_finder.suggestion()

    # trainer.tune(net,
    #              train_dataloader=train_dataloader,
    #              val_dataloaders=val_dataloader,
    #              )

    trainer.fit(
        net,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Eval
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NBeats.load_from_checkpoint(best_model_path)

    #error
    # Predicting on validation set
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_model.predict(val_dataloader)
    (actuals - predictions).abs().mean()

    # look at random samples in validation set prediction
    raw_predictions, x = best_model.predict(val_dataloader,
                                            mode="raw",
                                            return_x=True)

    for idx in range(prediction_length):  # plot X examples
        imgpath = "step{}.png".format(idx)
        basepath = os.path.join('experiment_plots', exp_name)
        best_model.plot_prediction(x,
                                   raw_predictions,
                                   idx=idx,
                                   add_loss_to_title=True).savefig(
                                       os.path.join(basepath, imgpath))
    print("experiment complete!")
    #make gif!
    #build_gif(basepath, "result.gif")


# This solves the "An attempt ... boostrapping phase"
# https://github.com/pytorch/pytorch/issues/5858
# NOTE: muliple GPU right now FAILS... which sucks...
if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     exit()
    # contract = sys.argv[1]
    # mel = sys.argv[2]
    # run_nbeats_rawprices_experiment(start="2021-12-30",
    #                                 end="2021-12-31",
    #                                 contract=contract,
    #                                 max_encoder_length=mel,
    #                                 max_prediction_length=10,
    #                                 prediction_length=60)

    run_nbeats_rawprices_experiment(start="2022-02-01",
                                    end="2022-02-02",
                                    contract="BTCUSDT",
                                    max_encoder_length=120,
                                    max_prediction_length=10,
                                    prediction_length=60)

# TIP: nohup python run_nbeats_rawprices.py &
# to run throughout the night
