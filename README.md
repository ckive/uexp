

(TO WRITE BETTER)

# How to use:
Understanding structure of repo:
```
.
│   README.md
│   run_experiment.py
|   ...    
│
└───ue
│   └───experiments
│   │   |
|   |   └───dan
|   |       |   m1_btc.py
|   |       |   experiment.ipynb
|   |       |   ...
|   |   | 
|   |   └───...
│   │
│   └───uexp
│       │   
|       └───benchmarking (tools)
|       |
|       └───data (for storing downloaded data)
|       |
|       └───dataprocessing (tools)
|       |
|       └───models (where to add new models)
│           │   BasicModel.py
│           │   model1.py
│   
└───model_experiments (contains saved models and plots)
    │   model_1_dense
    │   output.png
```
make a directory in ue/experiments with your name, you can write scripts and make notebooks in your directory
see m1_btc.py for an example

build your models in ue/uexp/models. see model1.py as an example

you can run your script that's in your folder from root dir by running
python run_experiment.py [dir_name] [filename]
*note* no need for extension in filename

You can try 
```
python run_experiment.py dan m1_btc
```
to see an example. It should print out training, evaluation losses and evaluation metrics as well as save a model and image to .model_experiments

Currently model is using tensorflow (bc much taken inspiration from (Inspired by)), will want to use pytorch uniformly. Please help with making your own models and experimenting with example models given by (Inspired by). They list many models that are implemented in the notebook. They also mention many 3rd party libraries (such as Facebook Kats, Linkedin Greykite etc), feel free to find a way to integrate them well with this repo.

# Todo
- bug: need to run twice to get good graph? first run graph is flat... but better to do number 2 first
- from diffed log returns back to prices for plotting
- implement the other types of models, especially...
    - NBEATS
    - leadlag method: series_x -> series_y for some horizon

# Resources:
Time series analysis + modeling
https://www.youtube.com/watch?v=s3XH7fTHMb4
https://www.youtube.com/watch?v=Prpu_U5tKkE

Great book for theory
https://otexts.com/fpp3/

Inspired by 
https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb

TradingView Basics by Eagan
https://docs.google.com/document/d/1fAzeJ1TtDGklCbdRWKJ-ymn-_MfDqDQ0aNnwGS5P5ww/edit?usp=sharing
