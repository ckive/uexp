import matplotlib.pyplot as plt

# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """[summary]
    plots 

    Args:
        timesteps ([np.array]): [array of timesteps (x axis)]
        values ([np.array]): [array of value (y axis)]
        format ([str]): [style of plot, default "."]
        start ([int]): [data to clean]
        end ([0]): [data to clean]
        label ([str]): [data to clean]
    """

    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Price ($)")
    if label:
        plt.legend(fontsize=14) # make label bigger
    plt.grid(True)